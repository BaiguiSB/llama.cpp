IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

上游贡献规则见 [AGENTS.md](AGENTS.md)。若改动要提交上游, 必须先读它。本文件记录本 fork 的私有信息: 目标环境、已审计结论、当前改造方案。

## 本 fork 定位与环境(必读)

- 私有 fork, 只为作者一台机器服务, 不用考虑其他平台/架构:
  Ubuntu 24.04, 1x Tesla V100 32GB (sm_70, cc=700, Volta, 80 SM, L2 6MB, HBM ~900GB/s), CUDA 12.8
- V100 相关判定: `turing_mma_available()` 恒 false, `volta_mma_available()` 恒 true (common.cuh:360-366)
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

## 性能基准: 分阶段耗时 (2026-09-20 实测)

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

单个 decode step (稳态 ~58 ms):

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

单点: 14252 token = 20457 ms (697 t/s), 其中 prefill stage 17827 ms (87%)。prefill 期间
nvidia-smi 全程 99-100% util / ~240W → **GPU 受限**, 佐证 P0 的 FA 优化方向。

### 阶段覆盖不到的部分 (优化时注意)

- **缓存命中时 `prompt_ms` 的 75-85% 不是计算**: 14252 token 上下文时 `prompt_ms`
  359.5 ms 但 prefill stage 只有 66.6 ms, 差的 293 ms 是 checkpoint 恢复 / KV 回滚 /
  建 batch / sampler init。目前无 stage 覆盖, 长上下文 + cache_prompt 场景值得单独看。
- 每 step 约 1.2 ms (2%) 在 stage 外: `pre_decode` + `post_decode` + 队列 drain。

### 方法学 (踩过的坑)

- 微秒级阶段 (detok/sample) 单次采样不可信: 同一 prompt 读数在 0.0006 与 0.46 ms/token
  之间跳 (700x), 是 CPU 调度噪声。用累计值或中位数。
- 计时必须在 `yield_to_queue` 的 lambda 内: 该函数在 work() 返回后还会等队列 worker
  排空, 在外层计时会把队列等待算进来。
- 不能按 "是否 sync 过" 过滤 `llama_decode` 计时: 大 prompt batch 它是阻塞的, 过滤会让
  prefill 只剩 0.3% (修正后 87%)。详见 commit `efded36ad`。

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

## 方案 B: TILE 内核直读量化 KV (通用改造, 目标模型下的定位见"目标模型"节)

核心思想: decode 是显存带宽瓶颈, 消灭 staging 往返, 让 TILE 在装载时反量化直接进 shared memory, 下游计算路径不动。

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
- llama-bench 看 decode t/s; nvidia-smi 看显存回落; 长上下文(staging 大头)收益最大

非目标/边界:
- prefill 仍走 MMA_F16 + staging: 接受(参考库同款妥协, prefill 算力受限不敏感), 后续可做"MMA 内 shared 级 staging"
- VEC 不动; K/V cache 是独立张量, K 与 V 的反量化路径都要实现
- 已否决/搁置: 方案 A(放宽 VEC 路由给 GQA decode, 赌 L2 去重, 只作对照实验, 最坏比现状差 60%); fp8_e4m3 KV(ggml 无 F8 类型, 端到端新类型工程量是方案 B 的 2-4 倍, 只省 6% 显存; 若将来做, Hadamard rotation 需手动接线, fa-v100 的软件转换可抄)

## 目标模型: Qwen3.5-27B (qwen35, 带视觉与 MTP)

注意力画像(决定所有 FA 优化):
- 64 层 = 48 层线性注意力(GDN, 无 KV cache) + 16 层 full attention(标准 KV cache + FA)
- full attn: 24 Q 头 / 4 KV 头 / head_dim 256 -> gqa_ratio=6, gqa_ratio_eff=2; max_position 262K
- MTP 1 层(qwen35.cpp 有 graph_mtp); KV 每 token: F16 64KB / q8_0 33.5KB / q4_0 17.8KB(只有 16 层, 比常规模型小 4 倍)

本地实际模型(2026-09 确认): **Qwen3.8-27B**, 与上面同一 qwen3_5 架构(HF config 的 model_type 就是 "qwen3_5"/Qwen3_5ForConditionalGeneration), 注意力画像逐项吻合:
- 文件: /home/baigui/nvme/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_M.gguf(主模型) 与 mtp-Qwen3.8-27B-Q4_0.gguf(MTP draft); HF config 副本暂存仓库根 qwen3.8-27b.json(未提交)
- 补充 shape 事实: full_attention_interval=4; vocab 248320; hidden 5120; 无 ALiBi(max_bias=0, 分派 gqa_opt 成立); partial_rotary_factor 0.25 + mrope_interleaved section [11,11,10]; 线性注意力 16 k 头/48 v 头 x 128 维
- 对照基线树: /home/baigui/nvme/llama.cpp(上游 vanilla, 供 A/B 对拍构建)

该模型在 V100 上的 FA 路由(注意与通用结论的差异):
- 单序列 decode: 有效 batch 1x2=2 -> VEC, 量化直读, 无 staging(通用警告"gqa%4 落 TILE"对 6:1 不适用)
- MTP verify(k>=2 个 draft)与多序列 decode: 有效 batch >=4 -> TILE -> staging 往返
- prefill: MMA_F16 -> staging 往返; 二次方增长, 64K 上下文时 staging 流量(~10.7GB/ubatch)与 tensor core 计算同级, full-attn prefill 被拖慢 1.7-2x

针对性优化优先级:
- P0: [已落地 2026-09-17] MMA_F16 量化直读(内核内 dequant -> shared), 改造点 fattn-mma-f16.cuh 的 flash_attn_ext_f16_load_tile(K/V 本来就过 shared 带 swizzle; V100 无 cp.async, Volta 路径本来就是同步 LDG+STS, 无流水线损失); 只做 D=256 形状, 模板面窄。收益: 长文 prefill 1.5-2x, verify 同吃
  - 提交链: 7556cb465(内核装载) e8446f649(分派/显存接入) fcfb3833a(补 process_tile 漏掉的 type_K/type_V 模板参数 —— 该遗漏使 q8_0 实例 TU 从 7556cb465 起一直编译失败, 即该路径此前从未真正构建过) e76a66dca(修 K/V 切片装载指针前移与 elem0 双重计账; 当前 D=256 实例 elem0 恒 0 无症状, 分片形状(320/256, 512/512, 576/512)或 Volta 调参降 nbatch_K2/nbatch_V2 会静默读错)
  - 实例与分派: (256,256) x {(16,2),(32,2),(32,1),(64,1)}, fattn.cu `ggml_cuda_flash_attn_ext_mma_f16_q8_supported()`: D=256 + q8_0-q8_0 + 上述 4 组合, 否则逐调用静默回退 F16+staging; env GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 强制回退(做 A/B 正控制用)
  - 本模型实际进入情况: prefill ubatch(512 行) -> (32,2) 实例命中; Q 行数 <=8 的尾巴 ubatch -> (8,2) 未实例化逐调用回退(supported() 与 alloc_size 互为镜像, 无显存错配); decode 单 token -> VEC; MTP verify -> TILE(仍 staging, 归 P1)
  - 验证方法论(重要): q8_0 与 baseline 的 PPL 逐位一致是设计预期(load_tile 反量化链与 convert.cu dequantize_block_q8_0_f16 是同一条 __hmul2), 因此 PPL 对拍既不能证明路径进入也不能证伪; 正控制 = 同一二进制设/不设 GGML_CUDA_FA_MMA_QUANT_FALLBACK 对比 compute buffer 大小(应差一个 staging)与 eval 时间。2026-09-17 实测 PPL 均值方差与 baseline 完全一致, 正控制待跑
  - 实测落地后perfill速度下降0.7%，属于负优化，改动都在分支`v100/mma-q8-direct`
- P0.5: lm_head 量化检查(vocab 248K, F16 2.5GB, 每 step 全读 ~2.8ms=14% step; 转 GGUF 用 --output-tensor-type q6_K/q8_0, ~10% decode 提速)
- P1: 方案 B(TILE 量化直读)定位调整: 服务 MTP verify 与多序列 decode, 普通 decode 用不上
- P2: 量化 decode 路由实验(VEC 只有 24 block, 占用率 30%; P1 后可试 TILE 分区+GQA 打包, 预期 3-5%)

## flash-attention-v100/ 参考库(只读)

来自 V100 优化版 vLLM 的 FA 库, 放在仓库根目录仅供查阅。torch/ATen 依赖, paged KV, 不参与构建, 不要链接或移植代码。
- 它的 decode 正是"内核内读 fp8 反量化", prefill 用显式 fp8->F16 HBM bridge: 独立验证了方案 B 的方向判断
- 可借鉴设计: smem bank conflict padding 步长(264/136), QK panel 双缓冲, 按固定 GQA 比值定制 WMMA M=8 tile(6 头+零填充), sawtooth 分区路由
- 它高度特化(GROUP_SIZE=6/D=256/固定页数 784/1616/MTP5, 指纹指向 MiniMax 类模型), 形状不匹配时 fallback 是无 GQA 打包的标量内核, 比 llama TILE 弱; 移植不划算
- fp8 软件转换参考: `kernel/fp8_kv_utils.cuh`(位操作, e4m3/e5m2)
