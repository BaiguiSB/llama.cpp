IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

上游贡献规则见 [AGENTS.md](AGENTS.md)。若改动要提交上游, 必须先读它。本文件记录本 fork 的私有信息: 目标环境、已审计结论、当前改造方案。
(2026-09-25 精简: 被取代的版本演进过程(v3/v4 详情、已闭环验证清单、方案 B 原设计、证伪叙事全文)已删, 需要时查 git 历史与 ~/report/*.md。)

## 本 fork 定位与环境(必读)

- 私有 fork, 只为作者一台机器服务, 不用考虑其他平台/架构:
  Ubuntu 24.04, 1x Tesla V100 32GB (sm_70, cc=700, Volta, 80 SM, L2 6MB, HBM ~900GB/s), CUDA 12.8
- V100 相关判定: `turing_mma_available()` 恒 false, `volta_mma_available()` 恒 true (common.cuh:360-366)
- **Volta 的 FP16 tensor core 在 FP32 累加时半速**: FP16 累加 ~112 TFLOPS, FP32 累加 ~56 TFLOPS。
  算效率时分母要按累加类型选(GEMM 走 `CUBLAS_COMPUTE_16F` 满速, FA 内核 FP32 累加半速)
- 用户保证量化 KV 只用对称组合: `q4_0-q4_0` 或 `q8_0-q8_0` (K 与 V 同类型), 不用混搭
- 改动只需在这台 V100 上正确且快, 不背其它硬件的包袱


## nsys 剖面方法学 (2026-09-20)

```sh
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node -f true -o /tmp/prof \
  ./build/bin/llama-bench -m "$MODEL" -ngl 99 -p 2048 -n 0 -fa 1 -d 10240 -ctk q8_0 -ctv q8_0
nsys stats --report cuda_gpu_kern_sum --format table --force-export=true -o /tmp/out /tmp/prof.nsys-rep
```

纯 CPU 解析(不占 GPU)。`nsys stats` 输出落到 `<前缀>_<report名>.txt`, 终端只打印 "PROCESSED";
首次先转同名 `.sqlite`(160MB 约需几分钟)。环境无 `sqlite3` CLI, 用 python3 的 `sqlite3` 模块查
`CUPTI_ACTIVITY_KIND_KERNEL`(join `StringIds`; `demangledName` 带 `void ` 前缀, 用子串匹配)。

**时间线切分(关键)**: 报告混了预热/深度填充/正式测量三段, 直接全量求和会得出错误占比。
用 FA 内核形状切: 每次 `llama_decode(n)` = `n/ubatch` 个 burst, 每 burst 含 16 次
`flash_attn_ext_f16`(16 层 full attention), burst 内时长随深度单调增长。
pp2048 的 44 burst = 4 预热 + 20 深度填充(10240) + 20 正式测量(5 rep x 4 ubatch);
tg512 以 gemm/dequant 内核消失为分水岭(decode 段为 0)。
`llama-bench -d` 语义: 每 rep 先 `llama_memory_clear` 再恢复深度状态(首次全量, 之后走
`llama_state_seq_set_data` 缓存), 只有 `test_prompt(n_prompt)` 被计时。

## PP 画像: 每 512-token ubatch 718.5 ms (690 t/s @ d10240)

| 阶段 | ms/ubatch | 占比 |
|---|---:|---:|
| cutlass F16 GEMM (`s884gemm_f16_128x128`) | 299.2 | 41.6% |
| **权重反量化 -> F16** (`dequantize_block_*`) | **169.2** | **23.5%** |
| `flash_attn_ext_f16` (MMA + staging) | 136.4 | 19.0% |
| `gated_delta_net_cuda` (GDN) | 46.1 | 6.4% |
| 其余 (convert_unary / silu / norm / concat / memcpy) | ~64 | 8.9% |

GPU busy 94.5%, 每 ubatch 2864 次内核发射。

- **GEMM 已打满, 不要去动**: 92 TFLOPS = V100 FP16 TC 峰值(~112)的 82%。
  `batched_mul_mat_traits<GGML_TYPE_F16>` 用 `CUBLAS_COMPUTE_16F` (ggml-cuda.cu:1397) → FP16 累加满速通道。
- **病灶: 权重反量化 23.5%, 每 ubatch 全量重做**: V100 无 int8 TC, dp4a 打不过 F16 TC,
  ggml-cuda.cu:1876 走 `ggml_cuda_mul_mat_cublas`, 1441-1465 每次 `mul_mat` 申请整块 F16 暂存并调
  `ggml_get_to_fp16_cuda`, 不复用。实测读 13451 + 写 39951 MiB = **52.1 GiB/ubatch = 2.9x 模型体积的
  纯格式转换**(佐证: q4_K grid=348160 每 ubatch 105.7 次 vs GGUF 同形状张量恰 107 个)。内核仅
  330-578 GB/s(可期 ~850, 余量 1.5-2.5x); 块配置 grid=348160 x **32 线程**, 每 warp 只处理一个
  256 元素超块(读 144B/写 512B), 访存并行度太低。
- **FA 19% 是占用率瓶颈而非带宽**: grid=192, **smem 67584B → 每 SM 只驻留 1 block(4/64 warp)**,
  17.1 TFLOPS = FP32 累加峰值(56)的 31%。全程仅 ~13-15 GB/s HBM 流量(带宽的 1.5%) →
  **消除 staging 省不下时间**(旧 P0 实测 -0.7% 证实)。

## TG 画像: 每 decode step 38.5 ms (25.98 t/s @ d10240)

| 阶段 | ms/step | 占比 | 备注 |
|---|---:|---:|---|
| `mul_mat_vec_q` 全家 | 28.36 | 73.6% | GEMV 读权重; Q4_K/Q5_K/Q6_K/Q8_0 |
| `flash_attn_ext_vec` + combine | 2.62 | 6.8% | O(n_kv), 已被 XQA-TC 接管(见下) |
| `rms_norm_f32` | 1.51 | 3.9% | 305 次/step |
| `quantize_q8_1` | 1.14 | 3.0% | 433 次/step, 与 mmv 严格一对一(每 matmul 单独量化激活, 无复用) |
| GDN + ssm_conv | 0.60 | 1.6% | |
| 其余小算子 + GPU 空转 | 4.30 | 11.2% | 空转 1.7ms 为 graph replay 间隙 |

每 step 1988 次内核发射, 98.4% 在 CUDA graph 内(launch 开销已压住), 但内核自身延迟受限(grid 1~48 block)。

- **权重带宽已近天花板**: 每 token 读 17.76 GiB(18.01 减去只做 gather 的 token_embd) ÷ 28.36 ms
  = 656 GB/s(峰值 900 的 73%)。天花板 21.9 ms → **单看 mmv 最好 ~45 t/s**。
  decode 空间 = 非 mmv 的 8.5 ms/step(22%) + mmv 自身效率差。
- **VEC 的 GQA 冗余**(历史第二大头): grid=(1,13,24), GQA 6:1, 同一 KV 头被 6 个 Q 头各完整读一遍;
  深度 10752 时 2.25 GB/step ÷ 2.62 ms = 858 GB/s 已打满 HBM, 64K 外推 15.6 ms/step(31%),
  与 server 实测逐项吻合。**→ 已由 XQA-TC 内核解决(KV 单读), 见专节**。

## server 分阶段计时 (2026-09-20 实测)

server 内置分阶段计时(server-common.h `server_stage`), 阶段 `prefill`/`decode`/`draft`/`verify`/
`sample`/`detok`(前两个是 target 前向, 按 batch 内 `is_prompt` 拆分)。三处输出: API
`timings.stages.<stage>`、`/metrics` `stage_seconds_total{stage=...}`(需 `--metrics`)、日志 `stages = ...`。

```sh
curl -s localhost:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"...","n_predict":128,"cache_prompt":true,"ignore_eos":true}' | jq .timings
curl -s localhost:8080/metrics | grep '^llamacpp:stage_'
```

测试配置: Qwen3.8-27B Q4_K_M + mtp-Q4_0, `-fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp
--spec-draft-n-max 3 --parallel 1 --ctx-size 64000`

长上下文稳态 decode step(~58 ms): forward 50.0(**86.6%**) / draft(MTP) 5.5 / verify 1.5 / 其余 1.2。
MTP 每 step 产 2.33 token(接受率 0.439, 按位置 0.664/0.388/0.250), draft+verify 花 6.9 ms 换
1.33 个额外 token(≥2.0x)。**即使 draft+verify 优化到 0 也只到 ~46 t/s(+16%) → 重心在 target forward。**

prefill 线性/二次分解(14252 token prompt, 7 点拟合, 残差 <0.4%; 单点 14252 = 20457 ms/697 t/s,
其中 prefill stage 17827 ms = 87%):

```
T(N) = 0.334 + 1.1127e-3*N + 1.847e-8*N^2   (秒; 线性=权重/MLP, 二次=注意力)
```

注意力占比: 4096→6.0%(792 t/s), 14252→18.8%(712 t/s), 32K→34.5%, 64K→51.4%(外推)。
prefill 期间 GPU 99-100% util, 但受限的是 GEMM+反量化(65%), 不是 FA(19%)。

**阶段覆盖不到的部分**: 缓存命中时 `prompt_ms` 的 75-85% 不是计算 —— 14252 token 时 `prompt_ms`
359.5 ms 但 prefill stage 只有 66.6 ms, 差的 293 ms = checkpoint 恢复/KV 回滚/建 batch/sampler init
(见下节审计); 另每 step 约 1.2 ms(2%)在 stage 外。

方法学坑: µs 级阶段(detok/sample)单次采样不可信(CPU 调度噪声, 同 prompt 读数差 700x), 用累计值或
中位数; 计时必须在 `yield_to_queue` 的 lambda 内(外层会把队列排空等待算进来); 不能按"是否 sync 过"
过滤 `llama_decode` 计时(大 prompt batch 它是阻塞的, 过滤后 prefill 只剩 0.3%, 见 commit `efded36ad`)。

## server context checkpoint 审计 (2026-09-21)

hybrid 模型(GDN recurrent + KV)的 checkpoint 存 **GDN 状态**(KV 自己靠 seq_rm/seq_add 滚)。
三份存放: `slot.prompt.checkpoints`(槽内跨请求, server-task.h:569)、prompt cache 深拷贝(槽换出后,
默认 `--cache-ram 8192`)、`slot.spec_ckpt`(投机解码每步重写, **本机配置下不触发**)。

生成点唯一: `create_checkpoint()`(server-context.cpp:2339, 调用点 3683), 在 prompt 处理的 batch 边界、
`llama_decode()` **之前**创建(不含当前 batch)。门控靠故意制造 batch 边界(`checkpoint_offsets =
{4+n_ubatch, 4}`) + user 消息起点(间距 ≥ `checkpoint_min_step` 默认 8192) → **14252 token prompt
产生 2 个 checkpoint(n_tokens 13736/14248)**(上游 a7b3dee7a / PR#20288 本意)。

体积与代价: `PARTIAL_ONLY` → 只写 mem_recr(llama-memory-hybrid.cpp:190); S 3 MiB + R 120 KiB /层,
F32(llama-model.cpp:2578-2579 写死) x 48 层 = **149.6 MiB/checkpoint**(宿主内存)。序列化 = 48 层 x2 张量
= **96 次独立 `ggml_backend_tensor_get`**(每次 cudaMemcpyAsync+Sync, ggml-cuda.cu:792-798), 完全串行
≈ **15 ms**, 在主循环线程上(期间 GPU 空转); restore 对称。淘汰策略会同 `n_tokens` 旧项直接擦掉重建
(重复请求重存 300 MiB, 纯冗余)。

与本机配置交互: `--spec-draft-n-max 3` → `common_context_can_seq_rm` 判 **RS**(common.cpp:1589) →
`n_rollback<=3` 时每 decode step 不存 checkpoint; 代价转移到 recurrent cache: `n_rows = mem_size*(1+3)`
= 4 行 ≈ **599 MiB 显存**, 且 graph 每步搬状态行(`build_rs` llama-graph.cpp:3478-3504, ~0.3 ms/step)。
无 SWA → `[TAG_CHECKPOINTS_FIX_POS_MIN]` 已知缺陷不适用。

与 293 ms 的关系: checkpoint 只解释约 45 ms(2 create + 1 restore); 剩余大头候选 `prompt_save`/`prompt_load`
—— NONE flags = **全量**(KV 496 + GDN 150 ≈ 646 MiB 往 + 646 MiB 回 @14252), 只在槽被 LRU 换出 /
`f_keep<0.5` 时发生(1666-1686)。现成埋点(server `-v`): `prompt cache update took %.2f ms`(1685)、
`created context checkpoint`(2398)、`restored context checkpoint`(3425)。

可动杠杆(按性价比): ① `llama_io_write_host` 自带 TODO(llama-context.cpp:2586 "batch tensor_get"):
96 次串行 sync 打包成一次 `cudaMemcpy2D`/批量 1D(纯拷贝路径, PPL 可逐位对拍); ② 同 `n_tokens`+同
`id_task` 跳过重存(GDN 状态只依赖前缀, 需先验证逐位相同); ③ `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`
(llama.h:914)已实现但 server 未用: 数据留显存省全部 D2H/H2D, 约束每 seq 一份+单 range → 只适配
`spec_ckpt`; ④ GDN 状态降精度(149.6 MiB 可减半, 需单独评估数值)。
未解决: 293 ms 精确拆分 —— 需 create/restore 两侧加计时后跑一次 14252+cache_prompt(占 GPU, 先问)。

## CUDA FlashAttention 子系统结构

入口与路由: `ggml/src/ggml-cuda/fattn.cu`
- `ggml_cuda_get_best_fattn_kernel()`: 按 cc 与形状选内核; `ggml_cuda_flash_attn_ext_get_alloc_size()`: 为 F16 staging 预留图 buffer
- 四个内核:
  - **XQA-TC** (`fattn-xqa.cuh`/`fattn-xqa.cu`, 本 fork 新增): V100 decode/verify 专用, n_tokens 1-4 + q8_0 直读 + GQA 6:1 打包(见专节)
  - **VEC** (`fattn-vec.cuh`): 一个 block 一个 Q 头, CUDA core + dp4a, 原生直读量化 K/V
  - **TILE** (`fattn-tile.cuh`): Q tile 进 shared, `flash_attn_tile_load_tile` 装 K/V 到 KV_tmp, CUDA core half2 FMA; 本 fork 已加 q8_0 直读(见 P1)
  - **MMA_F16** (`fattn-mma-f16.cuh`): tensor core, stream-K 调度; 只认 F16
- 公共装载/转换: `fattn-common.cuh` 的 `launch_fattn()`(need_f16_K/V 决定是否全量转 F16 落 HBM staging), 现成 `dequantize_V_q4_0/q8_0/...`(输出 half2)
- KV cache 写入: `ggml-cuda/set-rows.cu` (`set_rows_cuda_quant`, 支持 q4_0/q8_0 等)

## 已审计结论: 量化 KV 的 HBM 往返 (2026-09)

TILE(F16 路径)/MMA_F16 下, 每次 FA 算子执行(每 step 每 layer)把当前全部 K/V(整个 view 含 padding)
反量化成 F16 写入 HBM staging(KQV 输出张量后的 extra buffer), 内核再从 staging 读; 非增量。
路由(fattn.cu volta 分支): 有效 batch = `Q->ne[1] * gqa_ratio_eff`(gqa_ratio 的最大 2 幂因子, 上限 8),
`<=2`→VEC(直读无往返), `<=16`→TILE, 其余→MMA_F16。staging 按 n_kv 全量 F16(K+V)预留 →
长上下文时 ≈ F16 cache 尺寸, 量化 KV 显存净收益变负。n_kv 被 pad 到 256 倍数(llama-kv-cache.cpp `get_n_kv`)。
**修正(nsys)**: "5 B/el 往返"是理论比值而非性能瓶颈 —— FA 全程仅 ~14 GB/s。**消除 staging 只省显存,
不省时间**(旧 P0 实测 -0.7% 证实); 显存收益是这条线唯一的价值。
**量化 KV 启用 Hadamard rotation(llama-kv-cache.cpp, `ggml_is_quantized` 门控, QuaRot 式), 是量化 KV
质量的关键, 任何改造不得破坏。**

## XQA-TC 内核: GQA 6:1 去重的 decode 内核 (2026-09-24/25 落地, 现状 = v5)

按 flash-attention-v100 xqa_tc 设计实现(不移植代码, MMA 用 nvcuda::wmma): 一个 block 负责一个
(KV 头, KV 分片), 6 个 Q 头 x n_tokens 打包成 ceil(6n/8) 个 M=8 WMMA tile(尾部零填充), **KV 只读一次**。

- 文件: `fattn-xqa.cuh`(内核+launcher) / `fattn-xqa.cu`(gate+入口); fattn.cu 5 处接入
  (enum, Volta 分支最前, alloc 不留 f16 staging, dispatch, include)
- gate(`ggml_cuda_fattn_xqa_supported`): D=256 + q8_0-q8_0 + gqa_ratio==6 + n_tokens∈[1,4] +
  Q/K/V ne[3]==1 + 无 ALiBi/softcap/sinks + mask F16 共享或 null + n_kv%128==0 + V 布局同 K;
  env `GGML_CUDA_FA_XQA_FALLBACK=1` 强制回退(同二进制 A/B 正控制; 日志已在 e21063f2f 移除)
- 结构: 256 线程; QK 与 PV 均走 wmma 8x32x16(A row_major ld=272 / B ld=144, 行首 32B 对齐)。
  QK = 4 consumer warp 各 32-token 切片; softmax = 8 warp 行轮转(r = warp + j*8)标量在线 max/sum,
  P 以 f16 写入 sP[8NT][144](wmma A 布局); PV = warp 0-3 各认 V 面板 32 维切片, fragment 每 tile
  从零累加, store_matrix_sync 落 sO f32 后由行 warp 标量折叠(panel 0 顺带在线 rescale, 不依赖
  fragment lane 映射)。**KV 按 128 维 panel 分批进 smem**(QK 累加器跨 2 panel 存活, V 两面板两趟 PV;
  整块 256 维需 64KB+ 会爆 smem, 参考库 136/144 步长即 panel 行步长)
- smem: sP[8NT][144] half + sO[8NT][128] f32(S 先落此处, softmax 消费后复用为 PV chunk) +
  sQ[8NT][272] half + sKV[128][144] half; NT=1 共 47616B → 2 block/SM, NT=2/3 → 1 block/SM,
  launch_bounds (256, NT>=2?1:2)
- 数值: S 保持 f32, P 过一次 f16, 不再与 VEC 逐位一致(PPL 极接近, 已验证);
  边界组零填充是 wmma 正确性必需(PV 的 A 按 16 列整块读, softmax P 写入门控 t<ceil16(tv), 见 9f5b6e237)
- 分片: pb = min(2*nsm/H_kv, ntiles) = 40(V100), grid (1,40,4)=160 block 恰 1 波; pb>1 走
  `flash_attn_combine_results`(与 VEC/TILE 同一 dst_tmp/dst_meta ABI, 逐位镜像 VEC 写出公式)
- 路由: n=1(普通 decode 与 MTP draft)原走 VEC、n=2..4(MTP verify)原走 TILE q8 直读, 现**全部进 XQA**;
  prefill(>4 token)仍 MMA_F16+staging, 多序列/其他形状原路不动
- 寄存器门槛(政策: 无 spill 即可, NT>=2 预算 255 不必自限): cuobjdump NT=1/2/3 = REG 128/143/168,
  STACK/LDL/STL 全 0

### 当前性能 (v5 = TC-PV 后, 2026-09-25; 详见 ~/report/xqa_full_v5/v5_verdict.md)

- 内核 @64K: **301.41µs**(4.69ns/token, 有效带宽 464GB/s = 峰值 52%, 距 DRAM 下限 1.88x);
  combine 5.57µs; FA 降至 step 的 13.6%
- 停顿谱: long_sb 3.99(28%) > barrier 2.70 > lg_throttle 2.00 —— **内存延迟 42% 第一瓶颈**
  (16 warp/SM 藏不住装载延迟); XU 26.8% 为最忙计算管线(int8→f16 转换)
- e2e(d64000, -ntgs 1,2,4 三树): XQA **28.48/48.05/73.23** t/s, 对回退(VEC/TILE-q8)
  **+25.8/+11.5/+12.8%**, 对 vanilla +25.7/+21.0/+22.0%。收益随深度与行数放大:
  d10240 仅 +2.3%(FA 占 step <5%, 对 XQA vs TILE 无鉴别力, 胜负在 >=32K 深度)
- 版本演进(d64000 v1): 初版 27.52(+21.4% 对 VEC) → spill 修复/截断削减(e02e1ead9) 27.26 →
  TC-PV wmma 化(1a4252b9a+899db4b1c) 28.48; 内核 398.6→301.4µs(warp-inst −50%)
- 验证**全部闭环**: cuobjdump/ncu 运行时无 spill 双确认、64K e2e 无倒退、PPL 极接近、
  server MTP 实跑正常(verify 批 1-4 token, NT=1/2/3 全走 XQA; JSON acceptance 0.798 / 自由文本
  0.352 —— "结构化>>自由文本"健康签名, 行映射若错会塌方; 严格 logits 对拍不再必要。
  server step 时间含 batch4 verify + 3x draft + 采样, 与 bench 工况不可直比)。
  历史报告: ~/report/xqa_analysis.md(v2)、xqa_v3_verdict.md(v3)、xqa_full_v4/(v4, 已被 v5 取代)

### FA 线 backlog 与证伪清单 (2026-09-25)

**backlog(按 v5 停顿谱排序)**: ① int8→f16 转换指令经济性: PRMT+I2F 2.5 条/元素 → ~1.2,
参考库 E4M3_SHARED_LUT 的 q8_0 同构方案 = 256 项 int8→half LUT 进 smem, LDS+HMUL2 替换, 用 LSU 换
XU 压力; ② combine 并行化(5.57µs, 24 块 latency-bound, 参考库 split_reduce_dim_tile 的 D-tile 切法);
③ bank conflict(4.53M, f16 P 的 STS.U16 2-way, 非首项)。FA 打磨到 DRAM 下限的 e2e 增量 v1 约 +4-7%,
与 lm_head/其余项量级相当 —— FA branch 最大单项已收割。

**已证伪/已关闭, 勿重试**:
- **装载流水线(参考库 QK_SW_PIPELINE 结构; 两轮: 全 NT 与 NT>=2 专用重试)**: 64K e2e -2.0~-3.9%。
  机理: q8_0 反量化生产者不可约延迟链 ≈ 全局载入 ~500cyc + 4 slice 反量化串行 ~320 ≈ 820cyc,
  而 64 维拆分把消费窗砍半(NT=3 时 C/2≈450 < 820) → 流水线 = 同一条延迟链付 3 次换一个减半的窗口;
  生产者 unroll 2→4 零响应(MLP 不是瓶颈)。参考库该路径默认启用条件是 **fp16 KV**
  (flash_decode_paged.cu:5526, 生产者为纯拷贝零计算), 与 q8_0 反量化生产者不可比 ——
  唯一逃逸路径(生产者无反量化)对 q8_0 结构性不可得。**v5 串行 + 8 warp 全员装载 + (NT=1) pf
  寄存器预取即最优装载点**。教训: 移植参考库设计前核对其 env 默认的**使用点门控**(条件可藏在
  use 处而非 env 函数里)。被否版本曾存档 /tmp/xqa_qk_pipeline{,_nt2plus}_rejected.patch(重启或已失)
- ~~STS bank conflict 主线~~: ptxas 自动把 8xSTS.32 合并成 2xSTS.128, 残余冲突 1-3%
- (1,2,4) 小形状固定开销(25.6µs vs 理想 27.5 基线不可达): n_kv=256 时每 split 恰 1 全满 tile,
  截断机制不生效; 64K 每 block 12.55 tile 摊薄, 无 e2e 影响, 关闭
- 已落地修复(不必重做): spill 修复 + softmax 32 列组截断 + NT>=2 launch_bounds 放宽(e02e1ead9);
  P 尾列零填充 t<ceil16(tv) + pf 预取打包 18→10 reg(9f5b6e237; 该修复的 PPL 独立效应未跑,
  预期微小移动或不动, 皆可接受)

### GQA=8 派生内核 fattn-xqa8 (2026-09-25 代码完成, 待编译 + GGUF 验证)

目标模型 Qwen3.6-35B-A3B (qwen35moe, HF config 已逐项核对): head_dim 256, 16 Q 头/2 KV 头 = GQA 8,
40 层 = 30 GDN + 10 full-attn (KV 仅 10.6 KiB/token), MTP 1 层, max_pos 262K; build_attn 调用形式
(qwen35moe.cpp:342) 与 qwen35 逐字相同, gate 前提(无 ALiBi/softcap/sinks, F16 共享 mask)成立。

零修改加法派生: `fattn-xqa8.{cuh,cu}` = 原文件包 `namespace fattn_xqa8` + 仅 5 处数值改动
(q_head0/head 乘 8, rows=8n, gate ratio==8, nt=(8n+7)/8 == n_tokens) + case 4; fattn.cu 5 处 11 行
纯插入(include/enum 301/路由/alloc/dispatch, 0 行删除)。fattn-xqa.{cuh,cu} 一字未动, GQA=6 路径按编译
单元隔离逐位不变, 无需回归验证。两文件故意不重缩进保持可 diff 对拍: 今后 fattn-xqa 的任何修复
必须机械同步到 fattn-xqa8 (双改契约)。唯一非数值差异: pf 分支的 `dequantize_V_q8_0_regs` 调用带
`fattn_xqa8::` 限定 —— half* 实参的 ADL 会在 fattn.cu TU(含两头文件)把 fattn-xqa.cuh 的全局孪生
拉成同签名二义(首建实踩), 同步修复时必须保留该限定; 其余重名符号均为函数指针取值/常量引用, 无 ADL。

结构差异: NT == n_tokens 且 M=8 tile 恒满载(零填充循环成死代码, 保留)。NT=4 为新增实例
(smem 79872B < 96K opt-in, 1 block/SM, launch_bounds 的 NT>=2 分支已覆盖; REG 预估 190-200,
cuobjdump 无 spill 是硬门槛待实测)。gridDim.z=2 -> pb=min(80,ntiles): NT=1 时 (1,80,2)=160 block
恰 1 波, NT>=2 为 2 波(与现模型同状况)。env 开关与 GQA=6 共用(GGML_CUDA_FA_XQA_FALLBACK, 两 gate
按 ratio 互斥); 回退路径不同于 GQA=6: gqa_ratio_eff=8 > 2 -> 不走 VEC, n<=2 落 TILE staging,
n>=3 落 MMA_F16 ((256,256) q8 tile 无 ncols2=8 实例), XQA8 的对照优势包含整个 staging 往返。

待办(用户侧): 编译(CMake GLOB 需重新 configure); cuobjdump 查 NT=1..4 REG/STACK/LDL/STL;
GGUF 到手后 fallback A/B + PPL 对拍 + MTP acceptance 签名闭环(口径同 v5 验证链)。

## TILE q8_0 直读 (P1, 2026-09-18 落地, 分支 v100/tile-q8-direct)

定位: 曾服务 MTP verify 与多序列 decode; XQA 落地后主要作回退路径与对照基线。
- 改法: fattn-tile.cuh `flash_attn_tile_load_tile` 加 type_KV/elem0(K 与 V 共用此函数), q8_0 分支复用
  fattn-common.cuh `dequantize_V_q8_0<half,2*cpy_ne>` 寄存器反量化写 shared; iter_KQ/iter/kernel 透传
  type_K/type_V, q8_0 时 stride 保持字节单位; q8_0 行(34B 块)无法 half2 指针前移, K 尾段由 elem0 定位
  (与 F16 指针前移互斥, 不重复计账)
- 提交: 4cddb5514(内核装载) + 0d9531b73(分派/显存 + 实例文件)。实例 (256,256) x ncols2∈{1,2} x
  ncols1∈{1,2,4,8,16} 共 9 组, 文件 fattn-tile-instance-dkq256-dv256-q8_0.cu(CMake GLOB 自动收编,
  新文件需重新 configure); 扩容 = 加 DECL + 放宽 `ggml_cuda_fattn_tile_q8_supported` 两处检查;
  ncols2=4/8 与 ncols1=32 暂回退 staging
- 结构要点: `supported()`(fattn-tile.cu)是分派与 get_alloc_size **共用的唯一判定源**, staging 恰好在
  被使用时才预留 —— 结构性规避 mma 分支的 supported/alloc 镜像失配 bug 类
- A/B(2026-09-19): 载具 llama-batched-bench **必须加 -kvu**(默认 split KV 按序列拆 ubatch, FA 退化
  1 行走 VEC, 踩坑实录): `-fa on -ctk q8_0 -ctv q8_0 -kvu -c 16896 -npp 4096 -ntg 128 -npl 1,2,4`
  vs vanilla 树。pl=1 持平(VEC 对照); pl=2 +1.3%; pl=4 +2.4%(版本内重复性 <0.15%, 真实收益);
  PP 持平✓。机制: 消除 staging 理论省 ~2.5ms@880GB/s, 实测省 1.39ms(56%), 缺口 = 窄加载+反量化
  指令开销。收益 ∝ n_kv(64K 时 +8%级)。路径进入的 fprintf 确认已移除(e21063f2f), A/B 改用跨二进制对照

## 目标模型: Qwen3.8-27B (qwen35, 带视觉与 MTP)

(HF config 的 model_type 即 "qwen3_5"/Qwen3_5ForConditionalGeneration; 本地权重为 Qwen3.8-27B 系,
注意力画像与 Qwen3.5-27B 一致 —— 同一架构的两种称呼)

注意力画像(决定所有 FA 优化):
- 64 层 = 48 层线性注意力(GDN, 无 KV cache) + 16 层 full attention(标准 KV cache + FA)
- full attn: 24 Q 头 / 4 KV 头 / head_dim 256 → gqa_ratio=6, gqa_ratio_eff=2; max_position 262K
- MTP 1 层(qwen35.cpp 有 graph_mtp); KV 每 token: F16 64KB / q8_0 33.5KB / q4_0 17.8KB
  (只有 16 层, 比常规模型小 4 倍)
- 补充: full_attention_interval=4; vocab 248320; hidden 5120; 无 ALiBi(max_bias=0, 分派 gqa_opt 成立);
  partial_rotary_factor 0.25 + mrope_interleaved section [11,11,10]; 线性注意力 16 k 头/48 v 头 x 128 维

### 本地模型文件

主目录 `~/nvme/llama_models/Qwen3.8-27B-Uncensored-GGUF/`:
XQA/decode 线 bench 目录 `/home/baigui/nvme/llama_models/Qwen3.8-27B-Uncensored-GGUF/`:
`Qwen3.8-27B-Uncensored-Q4_K_M.gguf`(16 GiB, 主模型) + `mmproj-...-f16.gguf`(889 MiB)。
对照基线树: `/home/baigui/nvme/llama.cpp`(上游 vanilla, 供 A/B 对拍构建)。

**LynnStyle 主模型是混合 UD 式量化, 不是统一 Q4_K_M**(解析 GGUF 头, 18.01 GiB / 26.90 B 参数,
与 llama-bench 表头逐位吻合): Q8_0 62 个/3.01 GiB, Q4_K 108/5.66, Q5_K 172/3.28, Q6_K 156/6.04,
F32(norm/bias) 353/0.01。`output.weight (5120,248320)` **已是 Q6_K**(994.63 MiB); `token_embd` Q4_K。
关键 shape(dequant kernel grid = 元素数/256, 供 nsys 对照): ffn_gate/up (5120,17408) 与
ffn_down (17408,5120) → 348160; attn_qkv (5120,10240) → 204800; attn_q (5120,12288) → 245760;
ssm_out (6144,5120) / attn_gate (5120,6144) → 122880。

### 该模型的 FA 路由(现状)

- **n=1..4 的 decode/verify/draft → XQA-TC**(2026-09-24 起, 见专节)
- prefill(ubatch 512) → MMA_F16 + staging(grid 192, smem 67584B → 1 block/SM); staging 流量二次方
  增长, 64K 时 ~10.7GB/ubatch 与 tensor core 计算同级, full-attn prefill 被拖慢 1.7-2x
- 历史画像(XQA 前): 单序列 decode → VEC(grid (1,13,24), GQA 6x 冗余); MTP verify/多序列 → TILE;
  多序列仅限 -kvu unified KV(默认 split KV 每 FA 退化 1 行走 VEC)

## 优化优先级 (2026-09-25 现行)

| 优先级 | 动作 | 预期 | 状态/依据 |
|---|---|---|---|
| **P0** | `dequantize_block_*` 提速(提高每线程工作量 / streaming store 绕 write-allocate) | prefill **+10~15%** | `v100/perfill-pipeline` 线攻击中(capped-grid 持久化内核; CAP 扫描证伪重叠路线=寄存器堆互斥, WIP) |
| **P1** | FA prefill 占用率(降 smem 或调 nbatch 让每 SM 驻留 >1 block) | prefill **+10%** | 未动; smem 67584B 卡成 1 block/SM, 4/64 warp, 31% TC |
| P3 | 融合 433 次 `quantize_q8_1` + 305 次 `rms_norm`(grid 1~48 block, 纯延迟) | decode +3~4% | 未动 |
| ~~P2~~ | VEC 的 GQA 去重 | — | **已由 XQA-TC 兑现**(d64000 +25.8%) |
| ~~P0.5~~ | lm_head 量化 | — | **关闭**: output.weight 已是 Q6_K(1.2-1.5 ms/step, 3.5-4%; nsys v5 佐证 mul_mat_vec_q<Q6_K> 长尾恰 1/step); q6→q4 仅省 ~0.5ms 且有输出层质量风险, 不值 |

decode 剩余大头 = `mul_mat_vec_q` 家族 ~72-80% busy(量化权重流, 433 次/step 小 launch 多为
latency-bound); 下一个可选审计 = ncu 该家族达成带宽(合并/加宽有工程空间但回报递减)。

**已证伪/已否决**:
- **MMA_F16 量化直读(旧 P0, 2026-09-17, 分支 `v100/mma-q8-direct`)**: 实测 **-0.7% 负优化** ——
  FA 不吃带宽(全程 ~14 GB/s), 瓶颈是占用率, 消除 staging 只省显存。提交链(7556cb465 e8446f649
  fcfb3833a e76a66dca)与实例细节见该分支 git 历史。**方法学教训(重要)**: q8_0 反量化链与 convert.cu
  `dequantize_block_q8_0_f16` 是同一条 `__hmul2`, PPL 逐位一致是设计预期 → **PPL 对拍既不能证明也
  不能证伪路径进入**; 正控制 = 同一二进制设/不设 `GGML_CUDA_FA_MMA_QUANT_FALLBACK=1` 对比 compute
  buffer 大小(应差一个 staging)与 eval 时间
- 方案 A(放宽 VEC 路由给 GQA decode, 赌 L2 去重): 只作对照实验, 最坏比现状差 60%, 否决
- fp8_e4m3 KV: ggml 无 F8 类型, 端到端新类型工程量是方案 B 的 2-4 倍, 只省 6% 显存, 搁置
  (若将来做: Hadamard rotation 需手动接线, fa-v100 `kernel/fp8_kv_utils.cuh` 的软件转换可抄)

## flash-attention-v100/ 参考库(只读)

来自 V100 优化版 vLLM 的 FA 库, 放在仓库根目录仅供查阅。torch/ATen 依赖, paged KV, 不参与构建,
**不要链接或移植代码**。
- 它的 decode 正是"内核内读 fp8 反量化", prefill 用显式 fp8->F16 HBM bridge
- 可借鉴设计: smem bank conflict padding 步长(264/136), 按固定 GQA 比值定制 WMMA M=8 tile
  (6 头+零填充 —— XQA 的设计来源), sawtooth 分区路由, E4M3_SHARED_LUT(FA backlog ①),
  split_reduce_dim_tile(backlog ②)
- 高度特化(GROUP_SIZE=6/D=256/固定页数 784/1616/MTP5), 形状不匹配时 fallback 是无 GQA 打包的
  标量内核, 比 llama TILE 弱; 移植不划算

## 编辑历史

- 2026-09-25: base 线(nsys 剖面/PP/TG 画像, server 基准, checkpoint 审计)与 xqa 线(XQA-TC, tile q8)
  合流; 同日全文精简(681→约 270 行), 删除被取代的过程记录, 细节靠 git 历史与 ~/report/*.md
