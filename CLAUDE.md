IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

上游贡献规则见 [AGENTS.md](AGENTS.md)。若改动要提交上游, 必须先读它。本文件记录本 fork 的私有信息: 目标环境、已审计结论、当前改造方案。

## 本 fork 定位与环境(必读)

- 私有 fork, 只为作者一台机器服务, 不用考虑其他平台/架构:
  Ubuntu 24.04, 1x Tesla V100 32GB (sm_70, cc=700, Volta, 80 SM, L2 6MB, HBM ~900GB/s), CUDA 12.8
- V100 相关判定: `turing_mma_available()` 恒 false, `volta_mma_available()` 恒 true (common.cuh:360-366)
- **Volta 的 FP16 tensor core 在 FP32 累加时半速**: FP16 累加 ~112 TFLOPS, FP32 累加 ~56 TFLOPS。
  算效率时分母要按累加类型选(GEMM 走 `CUBLAS_COMPUTE_16F` 满速, FA 内核 FP32 累加半速)
- 用户保证量化 KV 只用对称组合: `q4_0-q4_0` 或 `q8_0-q8_0` (K 与 V 同类型), 不用混搭
- 改动只需在这台 V100 上正确且快, 不背其它硬件的包袱

## Build / Bench / Verify

```sh
# 配置(原生 V100 构建)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
# FA VEC 内核的量化组合, 默认已含 q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16 (ggml/CMakeLists.txt ~207)
# 如需扩展: -DGGML_CUDA_FA_QUANTS="q8_0-q8_0;q4_0-q4_0;f16-f16;bf16-bf16"

cmake --build build --target llama-bench llama-perplexity llama-cli -j

# decode 性能对比(fa 默认开; 对比 ctk/ctv f16 vs q8_0 vs q4_0)
./build/bin/llama-bench -m <model.gguf> -ngl 99 -p 512 -n 128 -fa 1 -ctk q8_0 -ctv q8_0

# 精度对拍(改造前后 perplexity 应逐位一致或极接近)
./build/bin/llama-perplexity -m <model.gguf> -ngl 99 -f <wiki.txt> -fa 1 -ctk q8_0 -ctv q8_0

# 显存/带宽观测
nvidia-smi dmon -s um
```

## nsys 剖面 (2026-09-20, 内核级真值)

```sh
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node -f true -o /tmp/prof \
  ./build/bin/llama-bench -m "$MODEL" -ngl 99 -p 2048 -n 0 -fa 1 -d 10240 -ctk q8_0 -ctv q8_0
nsys stats --report cuda_gpu_kern_sum --format table --force-export=true -o /tmp/out /tmp/prof.nsys-rep
```

纯 CPU 解析(不占 GPU)。`nsys stats` 的输出落到 `<前缀>_<report名>.txt`, 终端只打印 "PROCESSED";
首次会先把 `.nsys-rep` 转成同名 `.sqlite`(160MB 报告约需几分钟)。
环境无 `sqlite3` CLI, 用 python3 的 `sqlite3` 模块查 `CUPTI_ACTIVITY_KIND_KERNEL`
(join `StringIds`;注意 `demangledName` 带 `void ` 前缀, 用子串匹配而非前缀匹配)。

**时间线切分方法(关键)**: 报告混了预热 / 深度填充 / 正式测量三段, 直接对全量求和会得出错误占比。
用 FA 内核的形状切:每次 `llama_decode(n)` = `n/ubatch` 个 burst, 每个 burst 含 16 次
`flash_attn_ext_f16`(16 层 full attention), 且 burst 内时长随深度单调增长。
pp2048 的 44 个 burst = 4 预热 + 20 深度填充(10240) + 20 正式测量(5 rep x 4 ubatch)。
tg512 用 gemm/dequant 是否消失判断分水岭(decode 段这些内核完全为 0)。

`llama-bench -d` 语义: 每个 rep 先 `llama_memory_clear` 再恢复深度状态(首次跑全量, 之后走
`llama_state_seq_set_data` 缓存), 只有 `test_prompt(n_prompt)` 被计时。

---

## PP 画像: 每 512-token ubatch 718.5 ms (690 t/s @ d10240)

| 阶段 | ms/ubatch | 占比 |
|---|---:|---:|
| cutlass F16 GEMM (`s884gemm_f16_128x128`) | 299.2 | 41.6% |
| **权重反量化 -> F16** (`dequantize_block_*`) | **169.2** | **23.5%** |
| `flash_attn_ext_f16` (MMA + staging) | 136.4 | 19.0% |
| `gated_delta_net_cuda` (GDN) | 46.1 | 6.4% |
| 其余 (convert_unary / silu / norm / concat / memcpy) | ~64 | 8.9% |

窗口内 GPU busy 94.5%, 每 ubatch 2864 次内核发射。

### GEMM 已经打满, 不要去动它

2.75e13 FLOPs / 0.2992 s = **92 TFLOPS** = V100 FP16 TC 峰值(~112)的 **82%**。
`batched_mul_mat_traits<GGML_TYPE_F16>` 用 `CUBLAS_COMPUTE_16F` (ggml-cuda.cu:1397) → FP16 累加,
走的正是满速通道。

### 病灶: 权重反量化占 1/4, 且每个 ubatch 全量重做

V100 无 int8 tensor core, `ggml_cuda_should_use_mmq` 的 dp4a 路径打不过 F16 TC, 所以
ggml-cuda.cu:1876 走 `ggml_cuda_mul_mat_cublas`。该路径在 ggml-cuda.cu:1441-1465 每次
`mul_mat` 都 `src0_alloc.alloc(ggml_nelements(src0))` 申请整块 F16 暂存并调
`ggml_get_to_fp16_cuda`(`dequantize_block_*`)。**该转换按 ubatch 重做, 不复用**。

实测流量(按 grid x 块字节精确累加, 每 512-token ubatch):

```
读 13451 MiB + 写 39951 MiB = 52.1 GiB
```

模型本身才 18.01 GiB —— **每 512 token 搬运 2.9 倍模型体积做纯格式转换**。
佐证: `dequantize_block_q4_K` grid=348160(= (5120,17408) FFN 张量)每 ubatch 触发 105.7 次,
而 GGUF 里该形状的 Q4_K 张量恰好 **107** 个, 一个不多一个不少。

内核效率仅 330-578 GB/s(是否 write-allocate 决定取上取下的界), 对 ~850 GB/s 可期值有
**1.5-2.5x 余量**。块配置是 grid=348160 x **32 线程**, 每 warp 只处理一个 256 元素超块
(读 144B / 写 512B), 访存并行度太低。

### FA: 19%, 且是占用率瓶颈而非带宽瓶颈

grid=192x1x1, block=32x4, **smem=67584B → smem 限制每 SM 只能驻留 1 个 block(4 warp / 64)**。
17.1 TFLOPS = Volta FP32 累加峰值(56 TFLOPS)的 **31%**。

**FA 全程只有 ~13-15 GB/s 的 HBM 流量**(每层读 q8_0 KV 12.8MB + 写/读 F16 staging 96MB,
16 层共 ~1.75GB / 136ms), 是带宽的 1.5%。这解释了 P0 为什么是负优化: **消除 staging 省不下时间**。

---

## TG 画像: 每 decode step 38.5 ms (25.98 t/s @ d10240)

| 阶段 | ms/step | 占比 | 备注 |
|---|---:|---:|---|
| `mul_mat_vec_q` 全家 | 28.36 | 73.6% | GEMV, 读权重; 类型 Q4_K/Q5_K/Q6_K/Q8_0 |
| `flash_attn_ext_vec` + combine | 2.62 | 6.8% | 见下, O(n_kv) 随上下文放大 |
| `rms_norm_f32` | 1.51 | 3.9% | 305 次/step |
| `quantize_q8_1` | 1.14 | 3.0% | 433 次/step, 每次 2.6 µs |
| GDN + ssm_conv | 0.60 | 1.6% | |
| 其余小算子 | 2.60 | 6.8% | |
| GPU 空转 | 1.7 | 4.4% | graph replay 间隙 |

**每 step 1988 次内核发射**, 其中 98.4% 在 CUDA graph 内(launch 开销已被压住),
但这些内核自己是延迟受限的(grid 只有 1~48 个 block)。
`quantize_q8_1` 与 `mul_mat_vec_q` 数量**严格一对一**(11278 : 11278 每秒)——
每个 matmul 单独量化一次激活, 无复用。

### 权重带宽已接近天花板

每 token 读 **17.76 GiB**(18.01 GiB 减去只做 gather 的 `token_embd` 682 MiB)/ 28.36 ms
= **656 GB/s**(理论峰值 900 的 73%)。天花板 = 17.76 GiB / 870 GB/s = **21.9 ms**
→ **单看 mmv 最好也只到 ~45 t/s**。decode 的空间在:非 mmv 的 8.5 ms/step(22%)+ mmv 自身的效率差。

### VEC 的 GQA 冗余: 长上下文下的第二大头

grid=`(1, 13, 24)` —— z=24 个 Q 头, y=13 个 KV 分段。GQA 是 6:1,
**同一个 KV 头被 6 个 Q 头各完整读一遍**:

- 深度 10752:KV 23.4 MB/layer x6 冗余 x16 层 = 2.25 GB/step ÷ 2.62 ms = **858 GB/s → 已打满 HBM**
- 即 VEC 是带宽受限的, 2.62 ms 里约 5/6 是纯冗余

该曲线是 O(n_kv)。外推到 CLAUDE.md 记录的 64K 上下文:
`50 ms ≈ mmv 28.4 + FA 15.6 + 其余 6` —— 与 server 分阶段实测的 50.0 ms **逐项吻合**。

| 上下文 | VEC ms/step | 占 step |
|---|---:|---:|
| 10 K(本次) | 2.6 | 6.8% |
| 64 K(server 实测) | 15.6 | **31%** |

---

## 性能基准: server 分阶段耗时 (2026-09-20 实测)

server 已内置分阶段计时 (server-common.h `server_stage`), 三处输出:
- API `timings.stages.<stage>` = `{n, ms, ms_per_call}`
- `/metrics` `stage_seconds_total{stage=...}` / `stage_runs_total{stage=...}`
- 日志 `stages = prefill = ... ms x N (... ms/run), ...`

阶段: `prefill` / `decode` / `draft` / `verify` / `sample` / `detok`。前两个是 target
前向, 按 batch 内 `is_prompt` 拆分。取数(server 需带 `--metrics`):

```sh
curl -s localhost:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"...","n_predict":128,"cache_prompt":true,"ignore_eos":true}' | jq .timings
curl -s localhost:8080/metrics | grep '^llamacpp:stage_'
```

### 测试配置与结果

Qwen3.8-27B Q4_K_M + mtp-Q4_0, `-fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp
--spec-draft-n-max 3 --parallel 1 --ctx-size 64000`

单个 decode step (稳态 ~58 ms, 长上下文):

| 阶段 | ms/step | 占比 |
|---|---|---|
| decode forward | 50.0 | **86.6%** |
| draft (MTP) | 5.5 | 9.4% |
| verify | 1.5 | 2.6% |
| 其余 (建 batch / 采样 / 发响应) | 1.2 | 2.0% |

per token: forward 20.5-24.4 ms, draft 2.2-2.7, verify 0.65, detok ~0.001
→ 合计 23.8-27.8 ms/token (实测 35-41 t/s)。

MTP: 每 step 产出 2.33 token (接受率 0.439, mean len 2.30; 按位置 0.664/0.388/0.250),
draft+verify 花 6.9 ms 换 1.33 个额外 token, 对照估算 ≥2.0x (无 spec 的对照尚未实测)。
**即使 draft+verify 优化到 0 也只到 ~46 t/s (+16%), 优化重心应在 target forward。**

prefill 线性/二次分解 (14252 token 的 prompt, 7 个进度点拟合, 残差 <0.4%):

```
T(N) = 0.334 + 1.1127e-3*N + 1.847e-8*N^2   (秒, N = prompt token 数)
       ↑常量      ↑线性(权重/MLP)    ↑二次(注意力)
```

| N | 注意力占比 | 实测速率 |
|---|---|---|
| 4096 | 6.0% | 792 t/s |
| 14252 | 18.8% | 712 t/s |
| 32000 (外推) | 34.5% | — |
| 64000 (外推) | 51.4% | — |

单点: 14252 token = 20457 ms (697 t/s), 其中 prefill stage 17827 ms (87%)。

prefill 期间 nvidia-smi 全程 99-100% util / ~240W → GPU 受限属实(nsys 侧 busy 94.5%)。
**但受限的是 GEMM + 反量化(合计 65%), 不是 FA(19%)** —— 早期"佐证 FA 优化方向"的推断
已被 nsys 内核级数据推翻, 见上文 PP 画像。

### 阶段覆盖不到的部分 (优化时注意)

- **缓存命中时 `prompt_ms` 的 75-85% 不是计算**: 14252 token 上下文时 `prompt_ms`
  359.5 ms 但 prefill stage 只有 66.6 ms, 差的 293 ms 是 checkpoint 恢复 / KV 回滚 /
  建 batch / sampler init。目前无 stage 覆盖, 长上下文 + cache_prompt 场景值得单独看。
  checkpoint 那部分已审计, 见下文 "server 端 context checkpoint 生成逻辑"。
- 每 step 约 1.2 ms (2%) 在 stage 外: `pre_decode` + `post_decode` + 队列 drain。

### 方法学 (踩过的坑)

- 微秒级阶段 (detok/sample) 单次采样不可信: 同一 prompt 读数在 0.0006 与 0.46 ms/token
  之间跳 (700x), 是 CPU 调度噪声。用累计值或中位数。
- 计时必须在 `yield_to_queue` 的 lambda 内: 该函数在 work() 返回后还会等队列 worker
  排空, 在外层计时会把队列等待算进来。
- 不能按 "是否 sync 过" 过滤 `llama_decode` 计时: 大 prompt batch 它是阻塞的, 过滤会让
  prefill 只剩 0.3% (修正后 87%)。详见 commit `efded36ad`。

## server 端 context checkpoint 生成逻辑 (2026-09-21 审计)

用途: 保存"不可回滚"的那部分记忆, 供 prompt 前缀部分复用(divergence 落在已缓存区间内)时快速回退。
本模型是 hybrid(GDN recurrent + KV), 所以 checkpoint 存的是 **GDN 状态**, KV 自己靠 seq_rm/seq_add 滚。

三份存放, 生命期不同:

| 结构 | 位置 | 生命期 |
|---|---|---|
| `slot.prompt.checkpoints` (`std::list<common_prompt_checkpoint>`) | server-task.h:569 | 槽内跨请求 (cache_prompt) |
| prompt cache 里的**深拷贝**(外加 `data.main/drft` 全量 state) | server-task.cpp:1806 | 槽被换出后(默认 `--cache-ram 8192`) |
| `slot.spec_ckpt`(单个) | server-context.cpp:257 | 投机解码每步重写 |

生成点只有两个:
1. `create_checkpoint()` server-context.cpp:2339, **唯一调用点 3683** —— 在 prompt 处理的 batch 边界上,
   创建于 `llama_decode()` **之前**(见 3680 注释), 所以 checkpoint 不含当前 batch
2. `slot.spec_ckpt.update_tgt()` server-context.cpp:3130 —— **本机配置下不触发**(见下)

门控(3509-3684): `n_ctx_checkpoints>0`(默认 32) + COMPLETION 任务 + (`seq_rm_type ∈ {FULL,RS}` 或 `n_swa>0`)
+ 非 mtmd; 然后靠**故意制造 batch 边界**给 checkpoint 留位置 —— `checkpoint_offsets = {4+n_ubatch, 4}` (3614),
以及 user 消息起点(距上一个 ≥ `checkpoint_min_step`, 默认 8192)。真正落盘还要 `is_user_start || near_prompt_end`
+ `pos_min >= 0` + 间距检查。
→ **一条 14252 token 的完整 prompt 产生 2 个 checkpoint: n_tokens = 13736 与 14248**
(上游 a7b3dee7a / PR#20288 "make 2 checkpoints near the end of the prompt" 的本意)。

内容与体积(精确, 代码 + GGUF 元数据推):
- `update_tgt(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)` → llama-memory-hybrid.cpp:190 **跳过 KV, 只写 mem_recr**
- `n_embd_s = ssm_d_state*ssm_d_inner = 128*6144 = 786432` (3 MiB) / `n_embd_r = 3*(6144+2*16*128) = 30720` (120 KiB),
  两者都是 **F32** (llama-model.cpp:2578-2579 写死) → 48 层 = **每个 checkpoint 149.6 MiB**, 存宿主内存
- `data_dft` 空(draft ctx 是普通 KV cache, PART 类型, 只认 `== FULL`);`data_spec` 空(MTP 的 spec impl 没有 `get_state`)

搬运代价(按 PCIe3 估): 一次序列化 = 48 层 x 2 张量 = **96 次独立 `ggml_backend_tensor_get`**, 每次
`cudaMemcpyAsync + cudaStreamSynchronize` (ggml-cuda.cu:792-798) → 完全串行、无流水, 单次 ≈ **15 ms**,
且发生在主循环线程上(期间 GPU 空转)。restore 对称。

淘汰(2342-2382): 列表快满时删非本任务且距前一项 < min_step 的中间项 → 队首淘汰到有空位 →
**同 `n_tokens` 的旧项直接擦掉重建**(重复相同请求会重存 300 MiB, 纯冗余)。
恢复(3398-3433): 从新到旧找第一个 `pos_max <= pos_next` 且(`pos_min < pos_min_thold` 或 `pos_min == 0`),
`load_tgt` 后 `n_past` 退到 `pos_max`(保证至少重算 1 token);`pos_max > pos_next` 的项全部作废(3439)。

与本机配置的交互: `--spec-draft-n-max 3` → `n_rs_seq = 3` → `common_context_can_seq_rm` 直接判 **RS**
(common.cpp:1589), 于是:
- `use_ckpt_tgt` 要求 `== FULL` 或 `n_rollback > n_rs_seq`; 实测 `n_rollback <= 3` → **每 decode step 不存 checkpoint**,
  投机回滚走原生快照
- 代价转移到 recurrent cache: `n_rows = mem_size * (1 + n_rs_seq)` = **4 行** (llama-memory-recurrent.cpp:96)
  → **RS 缓冲 ≈ 599 MiB 显存**(S 576 + R 22.5);checkpoint 只读其中 1 行, 但 graph 每步还要搬状态行
  (`build_rs` llama-graph.cpp:3478-3504 的 `get_state_rows` + `cpy`;`--parallel 1` 时 1 行/层 ≈ 150 MiB/step 读写, 估 ~0.3 ms)
- 本模型无 SWA → `[TAG_CHECKPOINTS_FIX_POS_MIN]` (2388) 那个已知缺陷不适用

与"缓存命中时 prompt_ms 75-85% 不是计算"的关系: checkpoint 只解释约 **45 ms**(2 次 create = 300 MiB D2H
+ 1 次 restore = 150 MiB H2D);剩余候选大头是 `prompt_save`/`prompt_load` —— 走 NONE flags = **全量**
(KV 496 MiB + GDN 150 MiB ≈ 646 MiB 往 + 646 MiB 回 @14252), 只在槽被 LRU 换出 / `f_keep < 0.5` 时发生(1666-1686)。
三条现成埋点(server 加 `-v`): `prompt cache update took %.2f ms` (1685)、`created context checkpoint ... size = %.3f MiB` (2398)、
`restored context checkpoint ...` (3425)。另注: 3128-3133 有一段**被注释掉的** checkpoint 计时, 上游也曾怀疑此处。

可动杠杆(按性价比):
1. `llama_io_write_host` 自带 TODO (llama-context.cpp:2586 "batch tensor_get"): 96 次串行 sync 打包成一次
   `cudaMemcpy2D` / 批量 1D, 省掉同步开销(纯拷贝路径改动, PPL 可逐位对拍)
2. 同 `n_tokens` + 同 `id_task` 的 checkpoint 可跳过重存(GDN 状态只依赖前缀, 数据应逐位相同 —— 需先验证)
3. `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (llama.h:914) 已实现但 server 未用: 数据留显存, 省掉全部 D2H/H2D;
   约束是每 seq 只能一份、只能单 range → 只适配 `spec_ckpt`
4. GDN 状态 F32 存(149.6 MiB): 降精度可减半, 需单独评估数值

未解决: 293 ms 的精确拆分 —— 需在 create/restore 两侧加计时(与 `server_stage` 对齐)后跑一次
14252 token + `cache_prompt` 场景(要占 GPU, 需先问)。

## CUDA FlashAttention 子系统结构

入口与路由: `ggml/src/ggml-cuda/fattn.cu`
- `ggml_cuda_get_best_fattn_kernel()`: 按 cc 与形状选内核
- `ggml_cuda_flash_attn_ext_get_alloc_size()`: 为 F16 staging 预留图 buffer 空间
- 三个内核:
  - **VEC** (`fattn-vec.cuh`): 一个 block 一个 Q 头, CUDA core + dp4a, 唯一原生直读量化 K/V
  - **TILE** (`fattn-tile.cuh`): Q tile 进 shared, `flash_attn_tile_load_tile` 装 K/V 到 KV_tmp, CUDA core half2 FMA; 只认 F16
  - **MMA_F16** (`fattn-mma-f16.cuh`): tensor core, stream-K 调度; 只认 F16
- 公共装载/转换: `fattn-common.cuh` 的 `launch_fattn()` (need_f16_K/V 决定是否全量转 F16 落 HBM staging), 以及现成的 `dequantize_V_q4_0/q8_0/...` (输出 half2, VEC 在用)
- KV cache 写入: `ggml-cuda/set-rows.cu` (`set_rows_cuda_quant`, 支持 q4_0/q8_0 等)

## 已审计结论: 量化 KV 的 HBM 往返问题 (2026-09)

现象属实: TILE/MMA_F16 路径下, 每次 FA 算子执行(每 step 每 layer)都把当前全部 K/V(整个 view, 含 padding)反量化成 F16 写入 HBM staging(KQV 输出张量后面的 extra buffer), FA 内核再从 staging 读。非增量, 每次全量。

关键代码:
- fattn.cu `ggml_cuda_flash_attn_ext_get_alloc_size()`: TILE/MMA_F16 强制 `need_f16_K = need_f16_V = true`
- fattn-common.cuh `ggml_cuda_flash_attn_ext_get_f16_extra_data()` + `launch_fattn()`: staging 布局与 to_fp16 全量转换
- V100 decode 路由 (fattn.cu volta 分支): 有效 batch = `Q->ne[1] * gqa_ratio_eff`(gqa_ratio 的最大 2 幂因子, 上限 8):
  `<=2` -> VEC(量化直读, 无往返); `<=16` -> TILE(往返); 其余 -> MMA_F16(往返)
  所以 gqa_ratio 为 4 的倍数的模型, 单 token decode 落 TILE, 每 step 每 layer 全量转换
- 显存病灶: staging 按 n_kv 全量 F16(K+V)预留, 长上下文时 staging ~= F16 cache 尺寸, 量化 KV 显存净收益变负甚至倒贴
- 带宽: 现状 5 B/el(读量化 1 + 写 F16 2 + 读 F16 2), F16 cache 2 B/el, 目标 1.06 (q8_0) / 0.56 (q4_0) B/el
- 佐证细节: n_kv 被 pad 到 256 倍数 (llama-kv-cache.cpp `get_n_kv`), VEC 对齐条件恒满足
- 量化 KV 会启用 Hadamard rotation (llama-kv-cache.cpp, `ggml_is_quantized` 门控, QuaRot 式), 这是量化 KV 质量的关键, 改造不得破坏

**⚠ 修正 (2026-09-20 nsys)**: 上表的 5 B/el 是**理论比值, 不是性能瓶颈**。实测 prefill FA
全程仅 ~13-15 GB/s(带宽的 1.5%), VEC 在长上下文下是**因 GQA 冗余**(6 读 1)而非 staging
打满带宽。**消除 staging 只省显存, 不省时间** —— 已由 P0 实测 -0.7% 证实。
显存收益依然成立, 是这条结论目前唯一的价值。

## 方案 B: TILE 内核直读量化 KV (未实施)

核心思想: decode 是显存带宽瓶颈, 消灭 staging 往返, 让 TILE 在装载时反量化直接进 shared memory, 下游计算路径不动。
**定位(2026-09-20 修正)**: 主要价值在**显存**(staging 预留消失), 而非速度;
服务 MTP verify 与多序列 decode(**普通 decode 走 VEC, 用不上**)。

改造点(按依赖顺序):
1. `fattn-tile.cuh` `flash_attn_tile_load_tile()` (~377-425): 加 K/V 类型模板参数, 把"16B F16 拷贝"换成"读量化块 -> 寄存器反量化 -> 写 half2 进 KV_tmp"。K 和 V 共用此函数(两处调用), 改一处全生效
2. 反量化直接复用 fattn-common.cuh 的 `dequantize_V_q8_0/dequantize_V_q4_0`(现成, 输出 half2); q8_0 装载: 每线程 8B(8 个 int8) + 2B scale(scale 冗余取被 L1 吸收)
3. `fattn-tile.cuh` kernel 模板与实例化 switch (~1148-1234) 加 `type_K/type_V`(先 q8_0, 后 q4_0)
4. fattn.cu `ggml_cuda_flash_attn_ext_get_alloc_size()`: TILE 支持的类型时 `need_f16_* = false`(staging 消失, decode 图显存立刻省出)
5. `launch_fattn` 调用点(fattn-tile.cuh ~1166 等处)按类型传 `need_f16_K/V = false`
6. fattn.cu `ggml_cuda_fattn_kv_type_supported()` / 路由无需大改(V100 GQA decode 本来就落 TILE)

布局事实(不用过度担心对齐):
- q8_0 块 34B; per-head 行 = head_dim/32 * 34B(128 维时 136B), 8B 对齐; 整 token 行(1088B)16B 对齐
- warp 32 线程 x 8B 连续 = 256B = 8 个 32B sector, 合并访存零浪费; 顶多放弃 16B 宽载入

数值与验证:
- 反量化到 shared 的值与 staging 路径逐位等价(同样 d*q -> F16), 可直接对拍 perplexity
- llama-bench 看 decode t/s; nvidia-smi 看显存回落

非目标/边界:
- prefill 仍走 MMA_F16 + staging
- VEC 不动; K/V cache 是独立张量, K 与 V 的反量化路径都要实现
- 已否决/搁置: 方案 A(放宽 VEC 路由给 GQA decode, 赌 L2 去重, 只作对照实验, 最坏比现状差 60%); fp8_e4m3 KV(ggml 无 F8 类型, 端到端新类型工程量是方案 B 的 2-4 倍, 只省 6% 显存; 若将来做, Hadamard rotation 需手动接线, fa-v100 的软件转换可抄)

## 目标模型: Qwen3.8-27B (qwen35, 带视觉与 MTP)

注意力画像(决定所有 FA 优化):
- 64 层 = 48 层线性注意力(GDN, 无 KV cache) + 16 层 full attention(标准 KV cache + FA)
- full attn: 24 Q 头 / 4 KV 头 / head_dim 256 -> gqa_ratio=6, gqa_ratio_eff=2; max_position 262K
- MTP 1 层(qwen35.cpp 有 graph_mtp); KV 每 token: F16 64KB / q8_0 33.5KB / q4_0 17.8KB(只有 16 层, 比常规模型小 4 倍)

### 本地模型文件 (2026-09-20 更新路径)

目录已迁移, 原 `/home/baigui/nvme/models/Qwen3.8-27B/` **已不存在**。现位于
`/home/baigui/nvme/llama_models/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2/Q4-LynnStyle/`:

| 文件 | 大小 | 用途 |
|---|---:|---|
| `Qwen3.8-27B-EfficientThink-SimPO-Q4-LynnStyle.gguf` | 18.01 GiB | 主模型(nsys 本次测量所用) |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1.56 GiB | MTP draft |
| `mmproj-Qwen3.8-27B-Q4_K_M.gguf` | 498 MiB | 视觉 |
| `dflash2-qwen38-27b-Q4_K_M.gguf` | 1.06 GiB | DFlash2 |

**它是混合 UD 式量化, 不是统一 Q4_K_M**(解析 GGUF 头得到, 总计 18.01 GiB / 26.90 B 参数,
与 llama-bench 表头逐位吻合):

| 类型 | 张量数 | 体积 |
|---|---:|---:|
| Q8_0 | 62 | 3.01 GiB |
| Q4_K | 108 | 5.66 GiB |
| Q5_K | 172 | 3.28 GiB |
| Q6_K | 156 | 6.04 GiB |
| F32 (norm/bias) | 353 | 0.01 GiB |

关键 shape(反量化 kernel 的 grid = 元素数/256, 可直接对照 nsys 报告):
`ffn_gate/up (5120,17408)` / `ffn_down (17408,5120)` -> grid 348160;
`attn_qkv (5120,10240)` -> 204800; `attn_q (5120,12288)` -> 245760;
`ssm_out (6144,5120)` / `attn_gate (5120,6144)` -> 122880;
`output.weight (5120,248320)` **已经是 Q6_K**(994.63 MiB); `token_embd (5120,248320)` Q4_K。

补充 shape 事实: full_attention_interval=4; vocab 248320; hidden 5120; 无 ALiBi(max_bias=0,
分派 gqa_opt 成立); partial_rotary_factor 0.25 + mrope_interleaved section [11,11,10];
线性注意力 16 k 头/48 v 头 x 128 维。

对照基线树: `/home/baigui/nvme/llama.cpp`(上游 vanilla, 供 A/B 对拍构建)。

### 该模型在 V100 上的 FA 路由(已由 nsys 实测确认)

- 单序列 decode: 有效 batch 1x2=2 -> **VEC**(grid `(1,13,24)`), 量化直读, 无 staging
  (通用警告"gqa%4 落 TILE"对 6:1 不适用)。但见上文 GQA 冗余, 长上下文下它是第二大开销
- MTP verify(k>=2 个 draft)与多序列 decode: 有效 batch >=4 -> TILE -> staging 往返
- prefill(ubatch 512): MMA_F16 -> staging, grid 192x1x1 / smem 67584B

### 优化优先级 (2026-09-20 nsys 重排)

| 优先级 | 动作 | 预期 | 依据 |
|---|---|---|---|
| **P0** | `dequantize_block_*` 提速(提高每线程工作量 / streaming store 绕 write-allocate) | prefill **+10~15%** | 52.1 GiB/ubatch 只跑出 330-578 GB/s; 32 线程/块;重叠路线已证伪, 这是 dequant 唯一剩余方向(见下文专节) |
| **P1** | FA prefill 占用率(降 smem 或调 nbatch 让每 SM 驻留 >1 block) | prefill **+10%** | smem 67584B 卡成 1 block/SM, 4/64 warp, 31% TC 利用率 |
| **P2** | VEC 的 GQA 去重(6:1 共享 KV); 可结合 TILE 分区打包 | decode **长上下文 +25%**, 短上下文 +6% | 858 GB/s 已打满, 2.62ms 里 5/6 是冗余 |
| P3 | 融合 433 次 `quantize_q8_1` + 305 次 `rms_norm`(grid 只有 1~48 block, 纯延迟) | decode +3~4% | 每内核 2.6-7.2 µs, 与 mmv 严格一对一 |
| — | ~~P0.5 lm_head 量化~~ | **已失效** | 现模型 `output.weight` 已是 Q6_K, 每 token 1.16 ms(3.0%) |

已否决/已证伪:
- **MMA_F16 量化直读(旧 P0, 2026-09-17 落地)**: 实测 -0.7% 负优化, 改动在分支 `v100/mma-q8-direct`。
  原因已由 nsys 查明:**FA 不吃带宽**(全程 ~14 GB/s), 瓶颈是占用率。消除 staging 只省显存。
  - 提交链: 7556cb465(内核装载) e8446f649(分派/显存接入) fcfb3833a(补 process_tile 漏掉的 type_K/type_V 模板参数 —— 该遗漏使 q8_0 实例 TU 从 7556cb465 起一直编译失败, 即该路径此前从未真正构建过) e76a66dca(修 K/V 切片装载指针前移与 elem0 双重计账; 当前 D=256 实例 elem0 恒 0 无症状, 分片形状(320/256, 512/512, 576/512)或 Volta 调参降 nbatch_K2/nbatch_V2 会静默读错)
  - 实例与分派: (256,256) x {(16,2),(32,2),(32,1),(64,1)}, fattn.cu `ggml_cuda_flash_attn_ext_mma_f16_q8_supported()`: D=256 + q8_0-q8_0 + 上述 4 组合, 否则逐调用静默回退 F16+staging; env GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 强制回退(做 A/B 正控制用)
  - 本模型实际进入情况: prefill ubatch(512 行) -> (32,2) 实例命中; Q 行数 <=8 的尾巴 ubatch -> (8,2) 未实例化逐调用回退(supported() 与 alloc_size 互为镜像, 无显存错配)
  - 验证方法论(重要): q8_0 与 baseline 的 PPL 逐位一致是设计预期(load_tile 反量化链与 convert.cu dequantize_block_q8_0_f16 是同一条 __hmul2), 因此 PPL 对拍既不能证明路径进入也不能证伪; 正控制 = 同一二进制设/不设 GGML_CUDA_FA_MMA_QUANT_FALLBACK 对比 compute buffer 大小(应差一个 staging)与 eval 时间。2026-09-17 实测 PPL 均值方差与 baseline 完全一致, 正控制待跑
- **prefill dequant 跨流重叠流水线(六轮, 2026-09-23 证伪, 分支 `v100/perfill-pipeline`)**:
  寄存器堆互斥 —— cutlass GEMM 236 regs × 128 线程自共驻 2 块/SM 吃掉 94% 寄存器堆,
  剩 4096 regs 放不进任何 256 线程 dequant 块, 任何 grid 上限下共驻都物理不可行。详见下节。

## prefill dequant 跨流重叠流水线 — 证伪复盘 (2026-09-23, 分支 v100/perfill-pipeline)

设想: cuBLAS 路径每次 prefill mul_mat 全量 dequant 权重 → F16 staging (~200 ms/ubatch, 23.5%),
用旁路 stream (s15) 预取下一个权重, 与主流 cutlass GEMM (~430 ms/ubatch) 重叠。
SKIP 正控制(staging 留脏不转换)天花板 1044 t/s vs baseline 727 → 可藏的 dequant 时间值 +43%。

六轮修复, bench 始终 ≈727 t/s:

| 轮 | 修复 | commit | 结果 |
|---|---|---|---|
| 1 | 流水线骨架(VMM slot + 轮换 event + prefetch stream) | 9332a8070 / 49ae4b069 | 重叠率 1.5% |
| 2 | ready/done event 按代轮换(等待未决时重 record, wait 静默跟最新 record → 隐式串行化) | 2eeea937e | 零增益 |
| 3 | 发射顺序: dequant(j) 在 GEMM(j-1) launch 后才入队 | 39b19d6f8 | 零增益 |
| 4 | nsys v2 诊断: first-free 槽复用把依赖压成距离 1(dequant(j) 等 done(j-1)) | — | 定位 |
| 5 | slot 轮换恢复距离 2(dequant(j) 只等 done(j-2)) | 07f84b058 | 仍串行 |
| 6 | 限 grid 持久化 dequant 内核(256 线程/块 grid-stride)+ CAP 扫描 | 本次提交 | **任何 CAP 零重叠** |

CAP 扫描(pp4096 @ d4096, graphs off, env `GGML_CUDA_DEQUANT_PERSISTENT_BLOCKS`):

| cap | 0(原生巨 grid 对照) | 160 | 320 | 480 | 640 |
|---|---|---|---|---|---|
| t/s | 728.7 | 676.0 | 701.3 | 717.3 | 733.3 |

cap=0 精确复现基线(正控制成立); cap 单调恢复 ⇒ 持久化内核确实进入、grid 上限确实生效;
差值全部由"dequant 单独跑变慢"解释(warp 数减半 + grid-stride 循环开销 ~14%), 无一档向 900+ 跳变。

**根因(决定性, 来自 CUPTI `registersPerThread` 实测而非推测): 寄存器堆互斥, 不是 block slot。**

- 主力 GEMM `cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8`(grid 32×17×2 = 1088 块,
  avg 1163 µs): **236 regs × 128 线程**, 按 warp 粒度 256 取整 = 30720 regs/块, smem 32KB
  → 自身最优共驻 **2 块/SM = 61440 / 65536 regs(94%)**, 每个 SM 只剩 **4096 regs**
- 任何 256 线程 dequant 块 ≥ 8192 regs ⇒ **物理上塞不进正在跑的 GEMM 波次**, 与 CAP 无关;
  GEMM 块退休腾出的资源立刻被它自己 6.8 波 grid 的下一波回收(发射顺序贪心)
- 4096 缝隙恰好放 4 个原生 32 线程 q4_K warp(27 regs → 1024/warp)。但 ① Volta 分发器是否
  跨 stream 回填小缝隙未验证; ② 即使回填, 4/32 warp ⇒ dequant 在 GEMM 窗口内只推进 ~27%,
  收益天花板 ≈ +8% (~785 t/s); ③ 共驻预算要求 ≤32 regs/线程, 与快速 P0 内核不相容
  (q4_K 一个超块 144B 的载入缓冲就要 ~9×uint4 ≈ 36 regs) ⇒ 不值得做
- 第 2-5 轮修的 event 拓扑/发射顺序/依赖距离都是必要非充分: 第二轮"巨 grid 独占 block slot"
  只是表象, 真正锁死共驻的是 GEMM 自身的寄存器占用

**结论: 重叠路线按 plan 预注册判据终止。dequant 唯一剩余方向 = P0 纯提效**(宽载入 + `__stcs`
流式写, 串行全占用下直接生效, 预期 +10~15%)。持久化内核机制保留在树上: env 默认 0=关,
cap=640 对原生巨 grid 实测中性(733 vs 729), 可直接当 P0 改造载体。

方法学沉淀:
- **PPL 对拍不能证明路径进入** —— 每个实验开关必须配正控制(本路线: SKIP、cap=0 vs >0)
- 依赖距离判别: 数"绑定的 done-record 与 wait 之间发生了多少次 GEMM launch"(0 = 距离-1 特征);
  "绑定的 record 是几代前"这个指标在距离 1 和距离 2 世界里都返回 0, 无判别力(第二轮误判教训)
- CUPTI KERNEL 表带 registersPerThread / static+dynamicSharedMemory / grid+block 维度:
  共驻可行性可以纯算术判定, 无需跑新 profile(本轮根因即由此得出, 未跑 nsys v3)
- "GEMM 不要去动它"新增一层含义: GEMM 的 2 块/SM 共驻偏好是它 82% 峰值效率的前提,
  换小 tile / 低寄存器内核来给 dequant 腾共驻缝隙, 损失会大于重叠收益

## flash-attention-v100/ 参考库(只读)

来自 V100 优化版 vLLM 的 FA 库, 放在仓库根目录仅供查阅。torch/ATen 依赖, paged KV, 不参与构建, 不要链接或移植代码。
- 它的 decode 正是"内核内读 fp8 反量化", prefill 用显式 fp8->F16 HBM bridge
- 可借鉴设计: smem bank conflict padding 步长(264/136), QK panel 双缓冲, 按固定 GQA 比值定制 WMMA M=8 tile(**6 头+零填充 —— 与本模型 gqa_ratio=6 直接对口, 见 P2**), sawtooth 分区路由
- 它高度特化(GROUP_SIZE=6/D=256/固定页数 784/1616/MTP5), 形状不匹配时 fallback 是无 GQA 打包的标量内核, 比 llama TILE 弱; 移植不划算
- fp8 软件转换参考: `kernel/fp8_kv_utils.cuh`(位操作, e4m3/e5m2)
