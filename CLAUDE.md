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

## 方案 B: TILE 内核直读量化 KV (当前主线)

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

## flash-attention-v100/ 参考库(只读)

来自 V100 优化版 vLLM 的 FA 库, 放在仓库根目录仅供查阅。torch/ATen 依赖, paged KV, 不参与构建, 不要链接或移植代码。
- 它的 decode 正是"内核内读 fp8 反量化", prefill 用显式 fp8->F16 HBM bridge: 独立验证了方案 B 的方向判断
- 可借鉴设计: smem bank conflict padding 步长(264/136), QK panel 双缓冲, 按固定 GQA 比值定制 WMMA M=8 tile(6 头+零填充), sawtooth 分区路由
- 它高度特化(GROUP_SIZE=6/D=256/固定页数 784/1616/MTP5, 指纹指向 MiniMax 类模型), 形状不匹配时 fallback 是无 GQA 打包的标量内核, 比 llama TILE 弱; 移植不划算
- fp8 软件转换参考: `kernel/fp8_kv_utils.cuh`(位操作, e4m3/e5m2)
