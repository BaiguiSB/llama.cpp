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

cmake --build build --target llama-bench llama-batched-bench llama-perplexity llama-cli -j

# decode 性能对比(fa 默认开; 对比 ctk/ctv f16 vs q8_0 vs q4_0)
./build/bin/llama-bench -m <model.gguf> -ngl 99 -p 512 -n 128 -fa 1 -ctk q8_0 -ctv q8_0

# 单并发 verify 形状 A/B(本 fork 新增 -ntgs: 每步单序列连续 ntgs 个 token, logits 全 true,
# 对齐 server MTP verify 的批次形状; 接受逗号列表如 -ntgs 1,2,3,4, 每 ntgs 值展开一行 tg 测试
# (标签 tg128 @ v4); ntgs>=2 时 FA 走 TILE, ntgs=1 走 VEC 作对照;
# 跨 ntgs 的 t/s 不可比——batch 化本身的收益会混入, 必须同 ntgs 对比两棵树)
./build/bin/llama-bench -m <model.gguf> -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 4096 -n 128 -ntgs 4 -r 3

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

## 方案 B: TILE 内核直读量化 KV (q8_0 已落地 2026-09-18, 本节保留原始设计)

[q8_0 已落地 2026-09-18 于分支 v100/tile-q8-direct, 实施细节与验证状态见下方"针对性优化优先级"P1 条目; 本节保留原始设计。]

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

## XQA-TC 内核: GQA 6:1 去重的 decode 内核 (2026-09-24 落地)

按 flash-attention-v100 xqa_tc 的设计实现(不移植代码, MMA 用标准 nvcuda::wmma):
一个 block 负责一个 (KV 头, KV 分片), 6 个 Q 头 x n_tokens 打包成 ceil(6n/8) 个
M=8 WMMA tile(尾部零填充), KV 只读一次。

- 文件: `fattn-xqa.cuh`(内核+launcher)/ `fattn-xqa.cu`(gate+入口); fattn.cu 5 处接入
  (enum 300, Volta 分支最前, alloc 不留 f16 staging, dispatch, include)
- gate(`ggml_cuda_fattn_xqa_supported`): D=256 + q8_0-q8_0 + gqa_ratio==6 + n_tokens∈[1,4] +
  Q/K/V ne[3]==1 + 无 ALiBi/softcap/sinks + mask F16 共享或 null + n_kv%128==0 + V 布局同 K;
  env `GGML_CUDA_FA_XQA_FALLBACK=1` 强制回退(同二进制 A/B 正控制)
- 结构: 256 线程; QK 与 PV 均走 wmma 8x32x16(A row_major ld=272 / B ld=144, 行首 32B 对齐)。
  QK = 4 consumer warp 各 32-token 切片; softmax = 8 warp 行轮转(r = warp + j*8) 标量在线
  max/sum, P 以 f16 写入 sP[8NT][144](wmma A 布局); PV(1a4252b9a 起) = warp 0-3 各认 V 面板
  32 维切片, fragment 每 tile 从零累加, store_matrix_sync 落 sO f32 后由行 warp 标量折叠
  (panel 0 顺带在线 rescale, 不依赖 fragment lane 映射); smem sP[8NT][144] half +
  sO[8NT][128] f32(S 先落此处, softmax 消费后复用为 PV chunk) + sQ[8NT][272] half +
  sKV[128][144] half (NT=1 共 47616B -> 2 block/SM; NT=2 58368B / NT=3 69120B -> 各 1 block/SM,
  launch_bounds 按 NT 放宽 (256, NT>=2?1:2)); 数值: S 保持 f32, P 过一次 f16,
  不再与 VEC 逐位一致 (PPL 极接近, 已验证)
- **KV 按 128 维 panel 分批进 smem**: QK 累加器跨 2 个 panel 存活, V 两面板两趟 PV。
  参考库的 136/144 步长是 panel(128 维)行步长而非整维行步长, 整块 256 维需 64KB+ 会爆 smem
- 分片: pb = min(2*nsm/H_kv, ntiles) = 40(V100), grid (1,40,4)=160 block 恰 1 波;
  pb>1 时走 `flash_attn_combine_results`(与 VEC/TILE 同一 dst_tmp/dst_meta ABI, 逐位镜像 VEC 写出公式)
- 路由效果: n=1(普通 decode 与 MTP draft)原走 VEC、n=2..4(MTP verify)原走 TILE q8 直读,
  现全部进 XQA; prefill(>4 token)仍 MMA_F16, 多序列/其他形状原路不动
- one-shot 日志 "FA XQA tensor-core path taken" 打印的是首次 eligible 调用
  (warmup 时 n_kv=256 -> pb=2); 稳态 n_kv>=5120 后 pb=40

### 实测 (llama-bench -p 0 -n 128 -fa 1 -ctk/ctv q8_0, Uncensored-Q4_K_M 15.65GiB) (文件现位于 llama_models/Qwen3.8-27B-Uncensored-GGUF/)

| 深度 | VEC(fallback) | XQA | 提升 |
|---|---:|---:|---|
| d10240 | 30.81 t/s (32.46 ms/step) | 31.53 (31.71) | +2.3% |
| d64000 | 22.67 t/s (44.11 ms/step) | 27.52 (36.34) | **+21.4%** |

64K 拆解: VEC FA = 15.65ms(6x 读 13.4GB) -> XQA FA 实测 7.9ms(1x 读 2.24GB,
**有效带宽 ~284GB/s, 距 ~850 还有 ~3x**)。该差距随 n_kv 线性放大(吞吐型而非固定开销),
d10240 的 +2.3% 偏小同因。P2 预估 +25% 已兑现大半, 剩余空间在下面。
(上表为初版代码 2026-09-24 记录; e02e1ead9 后 2026-09-25 复测 22.64/27.26 = +20.4%,
内核级 398.62μs/351GB/s, 见下方 v4 终审节; TC-PV 后同日 28.48 = +25.8%,
301.41μs/464GB/s, 见 v5 终审节)

### 三 commit 优化落地与裁决 (2026-09-24 晚, 详见 ~/report/xqa_v3_verdict.md)

commit: add96b7ea(修 NT>=2 PV/epilogue 行映射, 正确性) aad960a2e(split 按 token 均分)
d2e08e8ae(PV 期间寄存器预取下一 tile K p0 + PV unroll 8)。诊断基线 ~/report/xqa_analysis.md (v2)。

实测裁决 (nsys in-flight, 注意 v3 capture 是 --n-depth 10240, 基线 5120, 不可绝对直比):
- 主 launch (1,40,4): 91.23μs @n_kv 10496 vs 改前两点模型 (27.36μs+7.26ns/tok) 103.6μs
  -> **净 −11.9% @10K** (敏感区间 −10.0..−13.7%); 5376 等效无法折算只能给界 54-60μs
  (−9.5..−18.7%; --n-depth 工况 capture 内 n_kv 恒定, 一份 capture=直线单点, a/b 不可分);
  预测 28-30μs 未达成 (最好极端 54μs, 差 ~1.8x)
- 均衡目标完全兑现: SM active 55.6->97.7%, per-SM max/min 1.77x->±1.6%, lg_throttle 1.73->0.89
- **预取触发红线**: REG 128 顶死, pf 36 reg 中 10 个 spill (LDL/STL 23K/12.8K warp-inst,
  local L1 命中 0.02%, ~4.6MB DRAM 往返), long_sb 1.90->3.83/inst, 每 token DRAM 2.25->2.97KB,
  L2 hit 36.3->21.3%; **斜率恶化风险**: 10K 单点的 −11.9% 与"截距大降+斜率变差"兼容,
  64K 外推 −2.5%..+5% 含倒退; 分离 a/b 需同二进制 ≥2 深度 capture, 修 spill 前不跑 64K bench
- 纯开销直接测量: warmup 对照 (1,2,4) 同形状 27.5->34.3μs (+25%) = 逐列 (t>=tv) 谓词 +
  运行时边界 + 双路径 i-cache 的代价 (该形状下均衡/预取均不生效)
- e2e (d10240, 3 次): XQA 31.45/56.48/82.56 vs TILE-q8 回退 30.71/56.77/83.19 vs vanilla
  30.73/55.54/81.68 (v1/v2/v4); XQA/回退 = +2.41%/−0.51%/−0.76%。FA 族仅占 step 4.74%
  (1.55ms/32.7ms), d10240 对 XQA vs TILE 无鉴别力, 胜负在 >=32K 深度
- step 构成 (稳态 busy): mul_mat_vec_q (MoE+GEMV) 79.8%, FA 4.8%, rms 3.8%, GDN 全部 ~1%;
  FA 打磨到 DRAM 下限在 10K 深度天花板 ~+3% -> lm_head (P0.5) 与 MoE 路径优先级上升
- NT=1 PPL 已复验 (7937ffd2b); NT>=2 数值正确性仍欠 (server MTP 接受率或 logits 对比)

### v4 终审: e02e1ead9 后 64K 全链路验证 (2026-09-25, 详见 ~/report/xqa_full_v4/v4_verdict.md)

nsys+ncu 与三次 e2e bench 同深度 (64000) 同二进制同会话; 旧 report (5120/10240 深度) 作废。
- 内核 in-flight: 主 launch (1,40,4) 中位 **398.62μs** (p10-p90 ±0.4%, n_kv=64256),
  6.20ns/token, 有效带宽 **351GB/s** (峰值 39%), 距 DRAM 下限 ~160μs **2.5x**;
  combine 5.70μs; FA/step 6.47ms = step 的 17.2%
- ncu (锁频 1.23GHz): **LDL/STL 运行时全 0** (红线 cuobjdump 外第二确认), DRAM
  2.71KB/token (读冗余 1.24x, KV 单读兑现), 停顿 barrier 2.24 > long_sb 1.88 > wait 1.36
  (long_sb 从 spill 时代 3.83 回落), HMMA 2.4%, SM 均衡 ±1.8%, shared 冲突超额 wavefront
  +17.4% (折算 ~3-5%, 维持不单独修)
- e2e (d64000, -ntgs 1,2,4 三树): XQA 27.26/44.06/64.83, 回退 22.64(VEC)/43.07/64.81
  (TILE-q8), vanilla 22.64/39.66/60.00 -> XQA/回退 **+20.4%/+2.3%/+0.0%**,
  XQA/vanilla +20.4%/+11.1%/+8.1%; 会话内自洽链: XQA 内核比 VEC 快 2.2x @64K
- 裁决: ①64K 无倒退 (对初版软参照 +21.4%->+20.4%, 跨会话噪声内), v3 时代斜率风险解除;
  ②(1,2,4) 33.28μs 未回 27.5 = 小形状固定开销 (n_kv=256 每 split 恰 1 全满 tile, 截断
  机制不生效; 残余=运行时边界+双路径形态), 64K 每 block 12.55 tile 摊薄无 e2e 影响, 关闭;
  ③**v2/v4 对 TILE-q8 打平 -> TC-PV 立项**: 1x 读优势被 2.5x 于下限的实现吃掉
  (TILE-q8 近峰值流式), 到下限则 v1 +12% t/s (36.68->32.8ms/step)、v2/v4 翻盘;
  ④tile-q8 vs vanilla verify 收益随深度放大 (+8.6%/+8.0% @64K vs 17K 时 +1.3%/+2.4%) 兑现;
  ⑤MTP 实跑首证 (llama-server --spec-draft-n-max 3): verify 批 1-4 token, NT=1/2/3 全走
  XQA, 输出正常; JSON 56-61 t/s @ acceptance 0.798 / 自由文本 34-35 t/s @ 0.352
  (tok/step 2.94 vs 1.52, step ~48-53/~43-45ms 含 batch4 verify + 3x draft + 采样开销,
  与 bench 工况不同不可直比)

### TC-PV 落地与 v5 终审 (2026-09-25, commit 1a4252b9a+899db4b1c, 详见 ~/report/xqa_full_v5/v5_verdict.md)

PV 搬上 wmma: softmax 把 P 以 f16 写入 sP(wmma A 布局), V 面板直接作 B; PV fragment
每 tile 从零累加, 经 store_matrix_sync 落 sO f32, 行 warp 标量折叠(panel 0 顺带在线
rescale)——不依赖 fragment lane 映射, 累加保持 f32, epilogue/combine 未动。S 因 wmma
f32 累加器只有 float* 重载而保持 f32 进 sO(softmax 消费后 sO 复用为 PV chunk 缓冲),
数值差异仅 P 一次 f16 舍入; 边界组零填充成为 wmma 正确性必需(非死存储)。同工况与 v4
严格可比(对照组漂移 ≤0.15%):
- 内核: nsys 主 launch 398.62 -> **301.41μs (−24.4%)**, 4.69ns/token, 有效带宽 464GB/s
  (52% 峰值), 距 DRAM 下限 1.88x; **warp-inst 71.58M -> 35.95M (−50%)**; HMMA 2.4->6.5%,
  LSU 39.1->15.4%; 停顿谱换位为 long_sb 3.99 (28%) > barrier 2.70 > lg_throttle 2.00
  (内存延迟 42% 成第一瓶颈); (1,2,4) 33.28->25.60μs; combine 5.57
- e2e (d64000 三树): XQA **28.48/48.05/73.23** (v1/v2/v4), 较 TC-PV 前 +4.5/+9.1/+13.0%;
  对回退 **+25.8/+11.5/+12.8%**, 对 vanilla +25.7/+21.0/+22.0% —— v2/v4 从 TILE 平替
  翻成双位数领先; 涨幅随行数放大(标量 PV 代价 ∝ 行数, wmma 解耦)
- 门槛: cuobjdump NT=1/2/3 = REG 128/143/168, STACK/LDL/STL 全 0 + ncu 运行时 local 全 0;
  PPL 极接近 + server MTP 正常
- 剩余 1.88x 归因: 装载延迟(long_sb+lg_throttle 42%, 16 warp/SM) > int8->f16 转换
  (XU 26.8% 最忙计算管线) > bank conflict(4.53M, 非首项); FA 已降至 step 13.6%,
  到下限的 e2e 增量 v1 约 +4-7% -> FA branch 最大单项已收割, lm_head (P0.5) 优先级升回

### 后续优化点 (2026-09-25 修订, 旧版 bank-conflict 主线已证伪)

1. [已落地并验证 e02e1ead9] 修 spill + 削截断开销: pf 缩 pf[2][9] (18 reg, 91+18=109
   留 ptxas 余量) + softmax 按 32 列组截断 (整组有效走无谓词直路, 仅边界组保留逐列
   -INFINITY 覆写) + t>=tv 死存储消去 (PV 上界 tv, 下 tile store_matrix 只写有效 slice)
   -> 验证 2026-09-25: cuobjdump + ncu 运行时双确认 LDL/STL=0, 64K 无倒退 (v4 终审节);
   (1,2,4) 小形状 33.3μs 未回 27.5 属固定开销 (该形状全满 tile 截断不生效), 已关闭
2. [已落地 e02e1ead9] NT>=2 launch_bounds 放宽 (256,1): NT=2/3 smem 53.8/62.2KB 本来就
   1 block/SM, 2-block 承诺的 128 reg 上限纯损失 (add96b7ea 后 NT=3 spill 5 reg 的根因)
3. ~~STS bank conflict~~ 已证伪: ptxas 自动把 8xSTS.32 合并成 2xSTS.128, 残余冲突值 1-3%
4. [已落地并验证 1a4252b9a+899db4b1c, 2026-09-25] TC-PV: PV wmma 化, warp-inst −50% 运行时
   兑现, 内核 398.6->301.4μs, e2e v1/v2/v4 +4.5/+9.1/+13.0%, v2/v4 对 TILE 翻成双位数
   领先 (见 v5 终审节)
5. v5 后 FA 剩余项 (按 v5 停顿谱排序): PANEL=64 双缓冲/装载流水 (long_sb+lg_throttle 42%
   成第一瓶颈, 16 warp/SM 藏不住装载延迟) > int8->f16 magic 转换 (PRMT+I2F 2.5 条/元素
   -> ~1.2, XU 26.8% 最忙计算管线) > combine 并行化 (5.57μs, 24 块 latency-bound) >
   bank conflict (4.53M, f16 P 的 STS.U16 2-way, 非首项)。FA 已降至 step 13.6%, 打磨到
   DRAM 下限 e2e 增量 v1 约 +4-7%, 与 lm_head (P0.5) 量级相当
6. [已实施并证伪 2026-09-25, 已回退] QK 装载流水 (参考库 QK_SW_PIPELINE 结构): K 改 4x64 维
   panel 双缓冲 (sKQ0/1 [128][72] 别名 V 的 [128][144]), warp 4-7 生产者装载 p+1 与 warp 0-3
   消费 wmma 并行。寄存器两轮调通 (NT=1 第二轮 128/STACK 0, 靠 pf 打包 u16[9]x2 -> uint4+u16
   18->10 reg + 生产者 unroll 4->2), 但 64K 三树 e2e 全跌: **27.63/46.16/71.57 vs v5 基线
   28.48/48.05/73.23 = -3.0/-3.9/-2.3%**, 且 NT=2/3 在 REG 130/153 无 spill 下跌 ->
   结构性负优化, 与寄存器无关 (寄存器一刀切政策同批被用户指出不当, NT=2/3 预算 255 应放开用)。
   机理: 旧结构 8 warp 全员装载 (MLP 满) + pf 寄存器预取跨 tile 藏延迟已是好点; 流水线把装载
   收缩到 4 warp, q8_0 反量化每 16 维 ~10 条指令使生产者成关键路径 (估算 X~1.5x(C/2) ->
   QK 相 +7% -> e2e -2.8%, 与实测吻合)。参考库该路径默认启用的条件是 **fp16 KV**
   (use 处 k_cache.scalar_type()==at::kHalf, flash_decode_paged.cu:5526): 其生产者是纯拷贝
   (LDG.128+STS.128 零计算), 4 warp 够用 —— 与 q8_0 反量化生产者不可比。
   裁决: 该路线对 "q8 直读 + 16 warp/SM 延时受限" 内核**证伪关闭**。回退保留: P 尾列修复
   (t<ceil16(tv), 见下) / pf 打包 (18->10 reg, NT=1 拿回 ~8 reg 余量)。被否版本存档
   /tmp/xqa_qk_pipeline_rejected.patch。教训: (a) 移植参考库设计前核对其 env 默认的
   **使用点门控** (kHalf 条件藏在 5526 行, 不在 env 函数里); (b) 生产者带反量化的装载流水
   只在装载 << 计算时成立, 否则单边瓶颈; (c) FA 剩余项重排: backlog #1 关闭, 下一个候选
   = int8->f16 转换指令经济性 (XU 26.8%) 或 combine 并行化
7. [已实施并证伪 2026-09-25, 已回退, 该线彻底关闭] NT>=2 专用装载流水重试: NT=1 定住 v5 串行
   (if constexpr), NT>=2 重启 4x64 维双缓冲流水, 生产者 unroll 4 (cuobjdump 184/206 无栈,
   单变量对照被否版本的 unroll 2)。实测 64K: **v2 46.26 / v4 71.80 vs 基线 48.05/73.23 =
   -3.7/-2.0%; v1 哨兵 28.41 钉住 ✓**。关键信息: unroll 2->4 仅 +0.2/+0.3% (46.16/71.57 ->
   46.26/71.80) —— **生产者 MLP 不是瓶颈, 产能损失不在生产侧**。机理终版 (与全部 4 组实测
   自洽): 生产者每 panel 的不可约延迟链 = 全局 L(~500cyc) + 4 slice 反量化串行(~320) ≈ 820cyc
   , 而 64 维拆分把消费窗砍半 (NT=3 时 C/2≈450 < 820) —— 流水线变成把同一条延迟链付 3 次
   对照一个减半的窗口: 模型算 NT=3 QK 相 +4%/NT=1 +36%, 对应 e2e -2%/-3%, 与实测吻合;
   unroll 动不了 L, 所以零响应。唯一逃逸路径 = 生产者无反量化 (参考库 fp16 纯拷贝), q8_0
   结构性不可得。**裁决: 装载流水线对本内核所有 NT 形状证伪, v5 串行 + 8 warp 全员装载 +
   (NT=1) pf 寄存器预取即最优装载点, 不再重试**。被否版本存档
   /tmp/xqa_qk_pipeline_nt2plus_rejected.patch。nsys/ncu: 决策不需要 (e2e 二值判决), 机制
   模型已与四组实测吻合; 若要墓志铭级确认可跑一次 ntgs=4 的 serial vs pipeline 停顿谱对照
   (预期 barrier 停顿占比上升), 可选
   backlog 重排 (FA 线): 装载流水关闭 -> 剩余候选 = int8->f16 转换指令经济性 (PRMT+I2F
   2.5 条/元素 -> ~1.2, XU 26.8% 最忙管线, 参考库 E4M3_SHARED_LUT 的 q8_0 同构方案:
   256 项 int8->half LUT 进 smem, LDS+HMUL2 替换 PRMT+I2F, 用 LSU 换 XU 压力) > combine
   并行化 (5.57μs, 24 块 latency-bound, 参考库 split_reduce_dim_tile 的 D-tile 切法) >
   bank conflict (4.53M, 非首项)
   同批附带一处正确性修复 (独立改动可单独对拍): softmax P 写入门控 t<tv -> t<ceil16(tv)。
   e02e1ead9 的"死存储消去"按 t>=tv 删了边界组尾列的零存储, 但 PV 的 A 装载按 16 列整块读
   (kc 上界 16*kc<tv), tv%16!=0 时最后 chunk 读到 [tv,ceil16(tv)) 的 stale P×stale V 列;
   分片边界 floor(split*n_kv/40) 产生任意 tv, 每 split 尾 tile 必触发 (n_kv=64240 例: 尾
   tile tv=70, chunk[64,80) 含 10 个 stale 列)。原注释明说尾列"must hold exact zeros"与代码
   矛盾, 修复即恢复注释语义。PPL 判据: 修复前后应微小移动 (stale 实非零) 或不动 (若 stale
   恒零则原代码无害, 修复成无操作), 二者皆可接受

### 未完成的验证

- 稳态 pb=40 已确认 (nsys v3/v4: 主 grid (1,40,4), 5120 实例)
- e02e1ead9 验证链 (全部闭环 2026-09-25): 构建✓ -> cuobjdump✓ (NT=1/2/3 = REG 128/144/163,
  STACK/LDL/STL 全 0) -> nsys/ncu v4✓ (运行时 LDL/STL=0, long_sb 回落 1.88, 64K 内核
  398.62μs) -> 64K e2e✓ (无倒退, +20.4% 保住) -> PPL 复验✓ (用户实测通过, NT=1 逐位
  等价设计兑现)
- TC-PV 验证链 (全部闭环 2026-09-25): cuobjdump NT=1/2/3 = REG 128/143/168, STACK/LDL/STL
  全 0 + ncu 运行时 local 全 0 -> nsys/ncu v5✓ (主 launch 301.41μs, −24.4%) -> e2e 三树✓
  (对照组 ±0.15% 不动) -> PPL 极接近✓ + server MTP 正常✓ (用户实测)。
  **XQA 线数值与性能验证全部闭环, 剩余均为纯性能 backlog**
- [已闭环 2026-09-25, 运行级] NT>=2 数值正确性: llama-server 实跑 MTP (--spec-draft-n-max 3,
  verify 批 1-4 token -> NT=1/2/3 全走 XQA), 输出内容正常; JSON 56-61 t/s @ acceptance 0.798,
  自由文本 34-35 t/s @ 0.352 —— 接受率呈"结构化>>自由文本"健康签名 (行映射若错会是垃圾
  输出 + 接受率塌方, 不可能到 0.8)。严格 logits 逐位对拍不再必要
- 多深度 a/b 分离降级为可选 (64K e2e 已答斜率无倒退); 旧 report 作废,
  新基准 = ~/report/xqa_full_v4 (与 e2e 同深度 64000)
- [已闭环 2026-09-25] 回退验证: 性能恢复 (用户实测), 提交 9f5b6e237 (P 尾列修复 + pf 打包)
  + bbfd80876 (docs 裁决); P 尾列修复的 PPL 独立效应仍待跑 (预期微小移动或不动, 皆可)
- [已闭环 2026-09-25, 证伪] NT>=2 流水线重试: cuobjdump 184/206 无栈✓ -> 64K 三树 v1 哨兵
  28.41 钉住✓ / v2 46.26 v4 71.80 (-3.7/-2.0%)✗ -> 回退到 9f5b6e237 状态 (工作树已 checkout,
  裁决见"后续优化点"第 7 条); 下次重建即恢复基线性能
## 目标模型: Qwen3.8-27B (qwen35, 带视觉与 MTP)

(HF config 的 model_type 即 "qwen3_5"/Qwen3_5ForConditionalGeneration; 本地权重为 Qwen3.8-27B 系,
注意力画像与 Qwen3.5-27B 画像一致 —— 两线对同一架构的两种称呼, 2026-09-25 合并注)

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

另一主模型目录(2026-09-25 合并注: XQA/decode 线 bench 所用):
`/home/baigui/nvme/llama_models/Qwen3.8-27B-Uncensored-GGUF/`:

| 文件 | 大小 | 用途 |
|---|---:|---|
| `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | 16 GiB | 主模型(XQA/llama-bench 测量所用) |
| `mmproj-Qwen3.8-27B-Uncensored-f16.gguf` | 889 MiB | 视觉 |

MTP draft 用 LynnStyle 目录下的 `mtp-Qwen3.8-27B-Q4_0.gguf`。
(旧记录 `/home/baigui/nvme/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_M.gguf` 已随目录迁移失效。)

### 该模型在 V100 上的 FA 路由(已由 nsys 实测确认)

**(2026-09-24 起 n=1..4 的 decode/verify/draft 已被 XQA-TC 接管, 见上文专用节, 本节保留为历史画像)**

- 单序列 decode: 有效 batch 1x2=2 -> **VEC**(grid `(1,13,24)`), 量化直读, 无 staging
  (通用警告"gqa%4 落 TILE"对 6:1 不适用)。但见上文 GQA 冗余, 长上下文下它是第二大开销
- MTP verify(k>=2 个 draft)与多序列 decode: 有效 batch >=4 -> TILE -> staging 往返
  (多序列仅限 -kvu unified KV; 默认 split KV n_stream=n_seq_max, split_equal 按序列拆
  ubatch, 每个 FA 退化 1 行走 VEC, 2026-09-19 实测踩坑)
- prefill(ubatch 512): MMA_F16 -> staging, grid 192x1x1 / smem 67584B; staging 往返二次方
  增长, 64K 上下文时 staging 流量(~10.7GB/ubatch)与 tensor core 计算同级, full-attn prefill
  被拖慢 1.7-2x

### 优化优先级 (2026-09-20 nsys 重排)

| 优先级 | 动作 | 预期 | 依据 |
|---|---|---|---|
| **P0** | `dequantize_block_*` 提速(提高每线程工作量 / streaming store 绕 write-allocate) | prefill **+10~15%** | 52.1 GiB/ubatch 只跑出 330-578 GB/s; 32 线程/块 |
| **P1** | FA prefill 占用率(降 smem 或调 nbatch 让每 SM 驻留 >1 block) | prefill **+10%** | smem 67584B 卡成 1 block/SM, 4/64 warp, 31% TC 利用率 |
| **P2** | VEC 的 GQA 去重(6:1 共享 KV); 可结合 TILE 分区打包 | decode **长上下文 +25%**, 短上下文 +6% | 858 GB/s 已打满, 2.62ms 里 5/6 是冗余 |
| P3 | 融合 433 次 `quantize_q8_1` + 305 次 `rms_norm`(grid 只有 1~48 block, 纯延迟) | decode +3~4% | 每内核 2.6-7.2 µs, 与 mmv 严格一对一 |
| — | ~~P0.5 lm_head 量化~~ | **已失效** | 现模型 `output.weight` 已是 Q6_K, 每 token 1.16 ms(3.0%) |

合并注记 (2026-09-25, 两线合流时更新):
- 表 P2 (VEC 的 GQA 去重, 预测 decode 长上下文 +25%) -> **已由 XQA-TC 内核兑现** (2026-09-24/25
  落地, d64000 对回退 +25.8%, 见 XQA-TC 专用节)
- 表 P0 (dequantize_block_* 提速) -> `v100/perfill-pipeline` 线攻击中 (capped-grid 持久化
  dequant 内核; CAP 扫描证伪重叠路线=寄存器堆互斥, WIP)
- 表 P1 (FA prefill 占用率) / P3 (融合 quantize_q8_1 + rms_norm) -> 未动
- FA 内部剩余 backlog 见 XQA-TC 节"后续优化点 (2026-09-25 修订)"

已否决/已证伪:
- **MMA_F16 量化直读(旧 P0, 2026-09-17 落地)**: 实测 -0.7% 负优化, 改动在分支 `v100/mma-q8-direct`。
  原因已由 nsys 查明:**FA 不吃带宽**(全程 ~14 GB/s), 瓶颈是占用率。消除 staging 只省显存。
  - 提交链: 7556cb465(内核装载) e8446f649(分派/显存接入) fcfb3833a(补 process_tile 漏掉的 type_K/type_V 模板参数 —— 该遗漏使 q8_0 实例 TU 从 7556cb465 起一直编译失败, 即该路径此前从未真正构建过) e76a66dca(修 K/V 切片装载指针前移与 elem0 双重计账; 当前 D=256 实例 elem0 恒 0 无症状, 分片形状(320/256, 512/512, 576/512)或 Volta 调参降 nbatch_K2/nbatch_V2 会静默读错)
  - 实例与分派: (256,256) x {(16,2),(32,2),(32,1),(64,1)}, fattn.cu `ggml_cuda_flash_attn_ext_mma_f16_q8_supported()`: D=256 + q8_0-q8_0 + 上述 4 组合, 否则逐调用静默回退 F16+staging; env GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 强制回退(做 A/B 正控制用)
  - 本模型实际进入情况: prefill ubatch(512 行) -> (32,2) 实例命中; Q 行数 <=8 的尾巴 ubatch -> (8,2) 未实例化逐调用回退(supported() 与 alloc_size 互为镜像, 无显存错配)
  - 验证方法论(重要): q8_0 与 baseline 的 PPL 逐位一致是设计预期(load_tile 反量化链与 convert.cu dequantize_block_q8_0_f16 是同一条 __hmul2), 因此 PPL 对拍既不能证明路径进入也不能证伪; 正控制 = 同一二进制设/不设 GGML_CUDA_FA_MMA_QUANT_FALLBACK 对比 compute buffer 大小(应差一个 staging)与 eval 时间。2026-09-17 实测 PPL 均值方差与 baseline 完全一致, 正控制待跑

### 针对性优化优先级 (FA 线记录, 2026-09-25 合并自 xqa 线)

- P0.5: [已核查 2026-09-25, 杠杆已兑现, 关闭] lm_head 量化: 本模型 UD-Q4_K_M 的 output.weight
  实为 **Q6_K 1.04GB** (llama-gguf 直读 tensor[0]; 早前按 F16 2.5GB 估的 ~2.8ms/step=14% step
  不成立), 每 step 全读 ~1.2-1.5ms ≈ step 的 3.5-4% (nsys v5 佐证: mul_mat_vec_q<Q6_K> 长尾
  1.2-1.5ms 恰 320 实例 = 1/step, 且全 capture 无任何 F16 GEMV 内核)。再往下只有 q6_K->q4_K
  (~省 0.5ms/step, 输出层质量风险) 不值。decode 剩余大头 = mul_mat_vec_q 家族 ~72% busy
  (量化权重流), 下一个可选审计 = ncu 该家族达成带宽 (433 次/step 小 launch 多为 latency-bound,
  合并/加宽存在工程空间但回报递减)
- P1: [已落地 2026-09-18, 分支 v100/tile-q8-direct] 方案 B(TILE 量化直读)定位调整: 服务 MTP verify 与多序列 decode, 普通 decode 用不上
  - 改造点: fattn-tile.cuh `flash_attn_tile_load_tile` 加 type_KV/elem0(K 与 V 共用此函数), q8_0 分支复用 fattn-common.cuh `dequantize_V_q8_0<half,2*cpy_ne>` 寄存器反量化写 shared; iter_KQ/iter/kernel 透传 type_K/type_V, q8_0 时 stride 保持字节单位; q8_0 行(34B 块)无法 half2 指针前移定位切片, K 尾段由 elem0 定位(F16 走指针前移, 互斥不重复计账)
  - 提交链: 4cddb5514(内核装载) 0d9531b73(分派/显存接入 + 实例文件)
  - 实例与分派: (256,256) x ncols2∈{1,2} x ncols1∈{1,2,4,8,16} 共 9 组(ncols2=1 时 ncols1 恒 >=2), 实例文件 fattn-tile-instance-dkq256-dv256-q8_0.cu(CMake GLOB 自动收编, 新文件需重新 configure); ncols2=4/8(gqa%4 模型)与 ncols1=32 暂回退 staging, 扩容=加 DECL + 放宽 `ggml_cuda_fattn_tile_q8_supported` 里两处检查
  - 与 mma 分支的关键差异: supported()(fattn-tile.cu)是分派与 get_alloc_size 共用的唯一判定源, staging 恰好在被使用时才预留, 结构性规避 supported/alloc 镜像失配 bug 类; GGML_CUDA_FA_TILE_QUANT_FALLBACK env 已移除(2026-09-19, A/B 改用 /home/baigui/nvme/llama.cpp vanilla 树构建做跨二进制约); 路径进入确认 = ggml_cuda_flash_attn_ext_tile_case_q8 每实例(每 (ncols1,ncols2) 组合)首次被调度时往 stderr 打一行 fprintf
  - 验证状态: [运行级 A/B 已跑 2026-09-19] 载具 llama-batched-bench, 必须加 -kvu(默认 split KV 按序列拆 ubatch, FA 退化 1 行走 VEC, 踩坑实录)且 -fa 新版参数是 on/off/auto: `./build/bin/llama-batched-bench -m <模型> -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -kvu -c 16896 -npp 4096 -ntg 128 -npl 1,2,4`, 对照 = vanilla 树(/home/baigui/nvme/llama.cpp)同参数, 各 3 次: pl=1(VEC 对照) 27.96 vs 27.97 t/s 持平; pl=2 TILE(2,2) 49.53 vs 48.90 (+1.3%); pl=4 TILE(4,2) 68.81 vs 67.21 (+2.4%); 版本内重复性 <0.15%, 信号 10-30 倍于组内极差, 判定真实收益而非噪声; PP 两版持平(本分支未动 MMA)✓; 机制核对: pl=4/n_kv~16.9K 时消除的 staging 流量理论 ~2.2GB/step(~2.5ms@880GB/s), 实测省 1.39ms/step(~56%), 缺口即内核窄加载+反量化指令开销(上次指令经济性分析的预测, 归 P2); 收益 ∝ n_kv, 长上下文应继续放大(32K 变体: -npp 16000 -npl 1,2 -c 32768); 逐 token 数值一致性与 staging 显存回落待 server MTP 路径最终确认; pl=8(8,2) 实例未测(-c 16896 放不下, 需 33792); 单并发 verify 工况(llama-bench -ntgs, 见 Build/Bench 节)工具已就绪, 数据待跑
  (注: UD-Q4_K_M = 合并时的旧主模型记录, 现位于 llama_models/Qwen3.8-27B-Uncensored-GGUF/,
  见"本地模型文件"节)
- P2: 量化 decode 路由实验(VEC 只有 24 block, 占用率 30%; P1 后可试 TILE 分区+GQA 打包, 预期 3-5%)
  —— 已被 XQA-TC 内核实质兑现(见上文专用节), 关闭

## flash-attention-v100/ 参考库(只读)

来自 V100 优化版 vLLM 的 FA 库, 放在仓库根目录仅供查阅。torch/ATen 依赖, paged KV, 不参与构建, 不要链接或移植代码。
- 它的 decode 正是"内核内读 fp8 反量化", prefill 用显式 fp8->F16 HBM bridge
- 可借鉴设计: smem bank conflict padding 步长(264/136), QK panel 双缓冲, 按固定 GQA 比值定制 WMMA M=8 tile(**6 头+零填充 —— 与本模型 gqa_ratio=6 直接对口, 见 P2**), sawtooth 分区路由
- 它高度特化(GROUP_SIZE=6/D=256/固定页数 784/1616/MTP5), 形状不匹配时 fallback 是无 GQA 打包的标量内核, 比 llama TILE 弱; 移植不划算
- fp8 软件转换参考: `kernel/fp8_kv_utils.cuh`(位操作, e4m3/e5m2)

## 编辑历史
  **(2026-09-24 起 decode/verify/draft 已被 XQA-TC 接管, 见上文专用节, 本小节保留为历史画像)**
  **(2026-09-25 分支重组: base 线[nsys 剖面/PP/TG 画像, server 分阶段基准, checkpoint 审计, 模型迁移记录]与
  xqa 线[XQA-TC, tile q8, FA 线记录]在本文件合流; 优先级表交叉注记见"合并注记")**