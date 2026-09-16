# MMA FlashAttention q8_0 直读优化设计 (P0)

日期: 2026-09-16
状态: 已批准
适用: 本 fork (V100 专属), 目标模型 Qwen3.5-27B (qwen35)

## 背景与动机

量化 KV cache 在 CUDA FA 的 MMA_F16 路径上存在 HBM staging 往返: 每次 FA 算子执行
(每 step 每 layer)把当前全部 K/V 反量化为 F16 写入图 buffer 的 extra 区域, FA 内核
再从该 F16 副本读取。详见 CLAUDE.md "已审计结论" 一节。

对目标模型 (16 层 full attn, D=256, GQA 6:1, max_position 262K):
- 普通 decode 走 VEC (量化直读), 无此问题
- prefill 走 MMA_F16, staging 流量随 n_kv 二次方累积: 64K 上下文时约 10.7 GB/ubatch,
  与 tensor core 计算量同级, full-attn prefill 被拖慢约 1.7-2x
- MTP verify (batch>=9) 与多序列大批次同样走 MMA_F16

目标: MMA_F16 内核直读 q8_0 KV, 装载时在寄存器反量化进 shared memory, 消灭 staging
往返与 staging 显存。参照 flash-attention-v100 参考库的同类设计 (fp8 -> half smem ->
WMMA)。

## 范围

做:
- K/V 均为 GGML_TYPE_Q8_0 (用户保证对称组合)
- DKQ = DV = 256
- 非 sparse 路径 (V100 恒非 sparse)

不做 (自动回退现有 staging 路径, 行为不变):
- q4_0 (后续独立 commit, 复用同一模板)
- 其它 head dim
- 混合 K/V 类型, sparse, cp.async 路径 (Ampere+)

## 设计

### 内核侧 (ggml/src/ggml-cuda/fattn-mma-f16.cuh)

1. `flash_attn_ext_f16_load_tile` 增加模板参数 `type_KV` (默认 GGML_TYPE_F16)。
   量化分支 (q8_0):
   - 保留现有 线程 -> (KV 行 i, chunk k) 映射 与 shared 侧写入 (含 swizzle) 完全不变,
     即每线程仍产出一行中 8 个连续元素的 4 个 half2, 16B STS
   - 全局侧寻址改为字节制: 行基址 = KV + i_KV * nb11 (nb11 为 q8_0 原始字节 stride,
     D=256 时每行 272 B); 块号 blk = k >> 2; scale = U16 载入于 34*blk;
     payload = 4 次 U16 载入于 34*blk + 2 + 8*(k&3)
   - 反量化: scale(half) 与 int8 逐元素乘加得 half2, 表达式与现有
     dequantize q8_0 -> f16 转换内核逐位一致 (实现时逐行对齐, 见"数值等价")
   - oob/sparse 分支: 量化路径 static_assert(!use_cp_async && !use_sparse);
     oob 越界行照旧写 0
2. 内核 `flash_attn_ext_f16` 增加模板参数 `type_K/type_V` (默认 F16)。量化时 K/V
   指针按原始量化数据使用, stride 直接用 fattn_kernel_t 传入的 nb11/nb21 字节值,
   不做 half2 换算 (仅装载函数内部寻址变化, 下游全部读 shared, 不受影响)
3. case 函数 `ggml_cuda_flash_attn_ext_mma_f16_case` 增加两个默认模板参数
   `type_K/type_V`, DECL 宏透传 -> 现有 100+ 实例化零改动
4. 新增量化实例化集 (D=256, q8_0-q8_0): (ncols1,ncols2) = (16,2), (32,2), (32,1), (64,1)
   - 依据 fattn.cu Volta 分派可达组合; (16,1) 等 ncols<32 组合 Volta 内核本身排空,
     不实例化, 由回退逻辑兜底
5. 量化实例的 `launch_fattn` 调用传 need_f16_K = need_f16_V = false
   (launch_fattn 不做转换, 直传 K->data/V->data 与原始 stride, 该分支已存在)

### host 侧 (ggml/src/ggml-cuda/fattn.cu)

1. `ggml_cuda_flash_attn_ext_mma_f16` 的 case 256:
   K->type == V->type == GGML_TYPE_Q8_0 且非 sparse-shall-use 时, 进入镜像的
   ncols 选择逻辑调用量化 case; 选中组合未实例化 -> 回退现有 F16 case
2. `ggml_cuda_flash_attn_ext_get_alloc_size`: MMA_F16 且量化支持时
   need_f16_K = need_f16_V = false (staging 预留消失)
3. 环境变量 GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 强制走旧路径 (A/B 对拍)

### 带宽与指令预算 (Volta)

- q8_0 块 34 B 导致 payload 仅保证 2B 对齐 (34*blk+2+8*c mod 4 在 0/2 交替),
  故用 U16 装载: 每线程 5 条载入指令 (1 scale + 4 payload) vs F16 的 1 条 16B
- 合并访存: warp 32 线程 x 2 B 连续 = 64 B = 2 个 32B sector, 零浪费
- L1 吞吐需求: 900 GB/s / 80 SM / 1.4 GHz ~= 0.25 sector/cycle/SM, Volta L1 能力
  4 sector/cycle -> 16x 余量, 指令数 x5 不构成瓶颈
- V100 无 cp.async (Ampere+ 特性), Volta 路径本为同步 LDG+STS, 无流水线损失

### 数值等价

staging 路径的值 = to_fp16_cuda(q8_0) 内核的输出 = d(half) * q8 -> half。
新路径必须在同一边界内做同样的运算与舍入。实现时以 convert.cu 中 q8_0 -> f16
内核的逐元素表达式为准逐行对齐, 验证门槛是 perplexity 完全相等 (不是"接近")。

### 验证计划 (代码在 Windows 编写, 编译与运行在 V100 Ubuntu 机器, 手动同步)

1. 编译门: `cmake --build build --target ggml -j` (只编译库, 暴露模板错误)
2. 性能: llama-bench, pp512 / pp2048 / pp16384 三档, -ctk q8_0 -ctv q8_0,
   对比 GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 旧路径; 预期提升随上下文增大,
   成功线: pp16384 >= 40% (full-attn 部分 ~1.7x, 折算全局含线性层)
3. 正确性: llama-perplexity, q8_0 新路径 vs fallback 输出必须逐 token 一致
4. 显存: nvidia-smi 确认长上下文下 staging 消失 (F16 K+V 尺寸回落)

## Commit 划分 (中文 message, 以用户名义提交, 不加协作署名)

1. `cuda : FA MMA 内核装载支持 q8_0 直读 (D=256)` - 内核侧 + 实例化
2. `cuda : FA 分派与显存分配接入 q8_0 直读` - fattn.cu dispatch + alloc + fallback 开关
3. `docs : 记录 q8_0 MMA 直读优化` - CLAUDE.md 状态更新

## 风险与回退

- 实例集遗漏: 症状为落回 staging (性能不升但结果正确), 回退逻辑兜底
- 数值表达式不一致: 对拍阶段暴露, 实现时逐行对齐转换内核
- Windows 侧无法编译验证: 编译门放在同步后第一步, 模板错误集中暴露
- 总回退: 环境变量一键回旧行为
