# MMA FlashAttention q8_0 直读实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MMA_F16 FA 内核直读 q8_0 量化 KV（D=256, V100），装载时寄存器反量化进 shared memory，消灭 HBM staging 往返与 staging 显存。

**Architecture:** 改造 `flash_attn_ext_f16_load_tile` 增加 K/V 类型模板参数，量化分支用 U16 装载 + `__hmul2` 反量化，shared 布局与下游零改动；case 函数与 fattn.cu 分派/显存分配各加一个共享谓词决定的量化入口，未支持组合自动回退现有 F16 staging 路径。

**Tech Stack:** CUDA C++ (sm_70), ggml CUDA backend。规格见 `docs/superpowers/specs/2026-09-16-mma-q8-direct-design.md`。

**验证方式说明(TDD 适配):** 本机(Windows)无 nvcc/GPU, 无法本地编译或跑测试。每个任务的验证 = 代码自查清单; 编译门与数值/性能验证由用户在 V100 Ubuntu 机器执行(命令在 Task 5), 结果回贴。数值正确性门槛: perplexity 与 fallback 路径逐 token 一致(逐位等价设计)。

**代码风格(AGENTS.md):** 注释简洁 1-2 行, ASD-STE 简单英语, 不用 emdash/unicode 符号, 不硬折行。

**已核实的关键事实(实现时不要重新推导, 直接引用):**
- `GGML_CUDA_FATTN_MMA_CONFIG_CASE` 宏自带 `static_assert(nbatch_K2 % 4 == 0 && nbatch_V2 % 4 == 0)`
  (fattn-mma-f16.cuh:32-33), 且装载 chunk = 8 元素, 故量化分支的元素起点 e0 = elem0 + k*8 恒为 8 的倍数,
  8 元素 chunk 永不跨越 32 元素的 q8_0 块边界, 块内偏移 iq ∈ {0,8,16,24}
- V100 对 D=256 走 Ampere 配置表 (volta 表无 256 项, 125 行 fallthrough): ncols=32/64 时
  nbatch_K2 = nbatch_V2 = 128 = 整行, k0_start 与 i0_start 恒为 0; 但代码必须保持通用
- V100 无 cp.async: `nstages = 0`, load_tile 走 else 分支(纯 memcpy_1<16>), 量化分支无流水线损失
- sparse 仅 DKQ=512/576 可能 (fattn-mma-f16.cuh:1758-1762), D=256 恒非 sparse
- `V_is_K_view = (DKQ == 576)` 为编译期常量 (fattn-mma-f16.cuh:1986), D=256 恒 false
- staging 数值基准: convert.cu:44 `dequantize_block_q8_0_f16` 用
  `__hmul2(make_half2(qs.x,qs.y), __half2half2(d))` (half 乘法), 量化分支必须用完全相同表达式
- `block_q8_0` 结构体 (`half d` + 32 字节 qs, 共 34B) 由 vecdotq.cuh 提供
- `ggml_cuda_memcpy_1<nbytes, nbytes_min>` 双参数形态可做 2B 对齐保证的字节装载
  (fattn-common.cuh:418 有用例)
- `bytes_rc<stride_tile>(row, col_h2)` 的列参量单位是 half2 (fattn-swizzle.cuh:44),
  swizzle 与非 swizzle 写入的列偏移都是 `k*h2_per_chunk`(= 4k, 即每 16B chunk 4 个 half2)
- Volta 上 swizzle 恒关闭: `enabled()` 需要 TURING_MMA_AVAILABLE (fattn-swizzle.cuh:15-22),
  V100 只走非 swizzle 分支(tile_stride = nbatch+4 padding), swz 分支仅为其它架构编译
- Volta ncols 选择链 (fattn.cu:199-222 与 146-166): ncols2 = 8/4/2/1 (gqa_opt 且 gqa_ratio 整除),
  ncols1 = 16/32/64 / ncols2 (按 Q->ne[1]); 量化实例集 = {(16,2),(32,2),(32,1),(64,1)}
- 目标模型(Qwen3.5-27B, gqa=6)可达组合: prefill 大批 -> (32,2); Q->ne[1]=9..16 -> (16,2)

**文件结构:**
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (Task 1-3: load_tile / iter / kernel / case 函数 / 实例化)
- Modify: `ggml/src/ggml-cuda/fattn.cu` (Task 4: 谓词 + 分派 + alloc_size)
- Modify: `CLAUDE.md` (Task 5: 状态)

---

### Task 1: load_tile 增加 q8_0 装载分支

**Files:**
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:367-477` (`flash_attn_ext_f16_load_tile`)

- [x] **Step 1.1: 修改函数模板签名**

把 (fattn-mma-f16.cuh:367-370):

```cpp
template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
```

改为:

```cpp
template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse,
    ggml_type type_KV = GGML_TYPE_F16>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices, const int elem0 = 0) {

    static_assert(type_KV == GGML_TYPE_F16 || (!use_cp_async && !use_sparse),
                  "quantized KV loading requires the synchronous non-sparse path");
```

说明: `stride_KV` 对 F16 是 half2 单位行距, 对 q8_0 是字节行距(由内核侧按类型准备, 见 Task 2);
`elem0` 是本次装载切片在行内的起始元素号(K 调用传 k0_start*2, V 调用传 i0_start, 无偏移调用用默认 0)。

- [x] **Step 1.2: 在非 cp_async 分支内加量化路径**

现有 else 分支(fattn-mma-f16.cuh:430 起)的 `auto load = ...` lambda 之前插入类型分派, F16 路径体保持
逐字节不变。将 else 分支改为(完整代码, 含原有 F16 逻辑):

```cpp
    } else if constexpr (type_KV == GGML_TYPE_Q8_0) {
        auto load = [&] __device__ (const int n) {
            const int stride_k = 32 >> n;
            const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

                const int64_t i_KV = k_VKQ_0 + i;

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    // e0 is always a multiple of 8, one chunk never crosses a 32 element q8_0 block
                    const int e0 = elem0 + 8*k;
                    const int ib = e0 >> 5;
                    const int iq = e0 & 31;

                    const block_q8_0 * row = (const block_q8_0 *) ((const char *) KV + i_KV*stride_KV);

                    __align__(4) int8_t qs[8];
                    half2 val[4];

                    if (!oob_check || i < i_sup) {
                        ggml_cuda_memcpy_1<sizeof(qs), 2>(qs, row[ib].qs + iq);
                        const half2 d2 = __half2half2(row[ib].d);
#pragma unroll
                        for (int l0 = 0; l0 < 8; l0 += 2) {
                            // same expression as dequantize_block_q8_0_f16 in convert.cu
                            val[l0/2] = d2 * make_half2(qs[l0], qs[l0 + 1]);
                        }
                    } else {
                        const half2 zero = __float2half2_rn(0.0f);
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            val[l] = zero;
                        }
                    }

                    if constexpr (swz) {
                        // col unit is half2, same as the f16 path
                        ggml_cuda_memcpy_1<16>((char *) tile_KV + ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, k*h2_per_chunk), val);
                    } else {
                        ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*4, val);
                    }
                }
            }
        };
        // 1: max 32*8=256 bytes, 256 half -> 8 q8_0 elements per thread
        // 2: max 16*8=128 bytes, 128 half
        // 3: max  8*8= 64 bytes,  64 half
        // 4: max  4*8= 32 bytes,  32 half
        // 5: max  2*8= 16 bytes,  16 half
        // 6: max  1*8=  8 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        ... 原 F16 else 分支体保持不变 ...
    }
```

注意: 原 F16 分支开头的 `const half2 zero[4] = ...;` 与 lambda 原样保留, 只是外层从 `} else {` 变成
`} else if constexpr (type_KV == GGML_TYPE_Q8_0) { ... } else {`。

- [x] **Step 1.3: 更新文件内 6 个 load_tile 调用点传 elem0**

F16 与量化共用同一签名(elem0 默认 0, F16 分支不读它), 只需给两个带偏移的调用点补参:
- fattn-mma-f16.cuh:651 (K 装载): `(K_h2 + k0_start, tile_K, k0_diff, stride_K, k_VKQ_0, k_VKQ_sup, indices)`
  追加 `, k0_start*2`
- fattn-mma-f16.cuh:1004 (V 装载): `(V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_0, k_VKQ_sup, indices)`
  追加 `, i0_start`
其余调用点(630, 988, 1310 附近)切片起点为 0, 用默认参数, 不改。

- [x] **Step 1.4: 自查**

- [ ] `block_q8_0` 在本文件可见(经 fattn.cu 包含链 fattn-common.cuh -> vecdotq.cuh); 若不可见, 在
      fattn-mma-f16.cuh 头部 include 区加 `#include "vecdotq.cuh"`
- [ ] 量化分支没有引用 `indices`(sparse 专用), 对应 static_assert 已加
- [ ] swizzle 与非 swizzle 写入的列参量与 F16 路径完全一致(`k*h2_per_chunk` 与 `k*4`, 单位 half2);
      V100 上 swz 恒 false, 只走非 swizzle 分支, swz 分支供其它架构编译
- [ ] 零填充分支覆盖 `i >= i_sup` 的所有行

### Task 2: iter 与内核模板加 type_K/type_V, 字节寻址

**Files:**
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:566-596` (`flash_attn_ext_f16_iter` 模板与签名)
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1764-1766` (`flash_attn_ext_f16` 内核模板)
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:~1852-1855` (stride 计算)
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1323,1332,1343,1352` (4 个 iter 调用点)

- [x] **Step 2.1: iter 模板加参数并透传**

fattn-mma-f16.cuh:566-568 模板参数列表尾部追加两个参数:

```cpp
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16>
```

iter 内 3 处 load_tile 显式模板实参追加类型:
- 630 行: `flash_attn_ext_f16_load_tile<stride_tile_V, swz_V, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse, type_V>`
- 650 行: `flash_attn_ext_f16_load_tile<stride_tile_K, swz_K, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse, type_K>`
- 987/1003/1310 行同理(它们在不同分支, 凡显式写模板实参的都加 type_K 或 type_V)

- [x] **Step 2.2: 内核模板加参数**

fattn-mma-f16.cuh:1764:

```cpp
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool use_sparse,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16>
```

- [x] **Step 2.3: 内核 prologue 的 stride 按类型计算**

把 ~1852-1855 的:

```cpp
    const int stride_K    = nb11 / sizeof(half2);
    ...
    const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(half2);
```

改为:

```cpp
    // for q8_0 the stride is passed in bytes, the load_tile quantized branch indexes bytes
    const int stride_K = type_K == GGML_TYPE_Q8_0 ? nb11 : nb11 / sizeof(half2);
    ...
    const int stride_V = V_is_K_view ? stride_K : (type_V == GGML_TYPE_Q8_0 ? nb21 : nb21 / sizeof(half2));
```

(`K_h2`/`V_h2` 指针强转保持不变, 只是地址; 若两者类型不同且 V_is_K_view 为 true 属未实例化组合, 不处理。)

- [x] **Step 2.4: 4 个 iter 调用点透传类型**

1323/1332/1343/1352 的调用模板实参列表(在 `oob_check` 之后, T_* 之前)插入 `type_K, type_V`。
T_* 参数仍由实参推导, 不写。

- [x] **Step 2.5: 自查**

- [ ] 所有 load_tile 显式模板调用都补了类型参数(默认参数在显式实例化时不会自动匹配位置错误 -> 编译期暴露)
- [ ] stride 单位注释清楚(F16: half2, q8_0: bytes)

### Task 3: case 函数模板参数, 量化实例化, need_f16 透传

**Files:**
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1966-1967` (case 函数模板)
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:2019,2029,2041,2054` (内核选择)
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:2065-2066` (launch_fattn)
- Modify: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:2070-2100` (DECL 区)

- [x] **Step 3.1: case 函数模板与内核选择**

1966 行改为:

```cpp
template <int DKQ, int DV, int ncols1, int ncols2,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
```

函数内 4 处 `flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel>`
统一追加 `, type_K, type_V`。

- [x] **Step 3.2: need_f16 按类型传**

2065-2066 改为:

```cpp
    constexpr bool quant_kv = type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0;
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, !quant_kv, !quant_kv, true, use_sparse, warp_size_host);
```

(launch_fattn 的 need_f16_K/V 为 false 时直传原始量化数据与 stride, 该分支已存在,
fattn-common.cuh:1016-1019。)

- [x] **Step 3.3: 量化实例化**

现有 DECL 宏(2070-2072)不改动(默认模板参填 F16/F16, 现有 100+ 实例化行零改动)。
在实例化区(extern 声明区与定义区, 按 2075-2107 与 2109+ 两段的现有模式分别放置)加入:

```cpp
#define DECL_FATTN_MMA_F16_CASE_Q8_0(DKQ, DV, ncols1, ncols2)                                       \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                                             \
        <DKQ, DV, ncols1, ncols2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_MMA_F16_CASE_Q8_0(256, 256, 16, 2)
extern DECL_FATTN_MMA_F16_CASE_Q8_0(256, 256, 32, 2)
extern DECL_FATTN_MMA_F16_CASE_Q8_0(256, 256, 32, 1)
extern DECL_FATTN_MMA_F16_CASE_Q8_0(256, 256, 64, 1)
```

定义区(非 extern)同样四行。

- [x] **Step 3.4: 自查 + 编译门(V100 机器)**

- [ ] 确认 fattn.cu 里 switch 函数调用的 case 符号不变(默认参数, 4 参调用仍合法)
- [ ] 用户执行: `cmake --build build --target ggml -j` -> 编译通过(量化实例化会强制实例化内核模板,
      模板错误在此暴露)
- [ ] 此时行为无任何变化(没有分派入口), 属安全中间态

- [x] **Step 3.5: Commit 1**

```bash
git add ggml/src/ggml-cuda/fattn-mma-f16.cuh
git commit -m "cuda : FA MMA 内核装载支持 q8_0 直读 (D=256)"
```

### Task 4: fattn.cu 分派, 共享谓词, alloc_size, fallback 开关

**Files:**
- Modify: `ggml/src/ggml-cuda/fattn.cu:264-393` (`ggml_cuda_flash_attn_ext_mma_f16` case 256)
- Modify: `ggml/src/ggml-cuda/fattn.cu:690-724` (`ggml_cuda_flash_attn_ext_get_alloc_size`)
- 新增两个静态函数(放在 `ggml_cuda_flash_attn_ext_mma_f16` 之前)

- [x] **Step 4.1: 加共享谓词**

```cpp
// must stay in sync with ggml_cuda_flash_attn_ext_mma_f16_q8 below and with
// the Volta branch of ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2/_ncols1
static bool ggml_cuda_flash_attn_ext_mma_f16_q8_supported(const ggml_tensor * dst) {
    static const bool fallback = getenv("GGML_CUDA_FA_MMA_QUANT_FALLBACK") != nullptr;
    if (fallback) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0) {
        return false;
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    bool gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, mask}) {
        if (t == nullptr) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    int ncols2 = 1;
    if (gqa_opt && gqa_ratio % 8 == 0) {
        ncols2 = 8;
    } else if (gqa_opt && gqa_ratio % 4 == 0) {
        ncols2 = 4;
    } else if (gqa_opt && gqa_ratio % 2 == 0) {
        ncols2 = 2;
    }

    const int ncols1 = Q->ne[1] <= 16/ncols2 ? 16/ncols2 : (Q->ne[1] <= 32/ncols2 ? 32/ncols2 : 64/ncols2);

    return (ncols1 == 16 && ncols2 == 2) || (ncols1 == 32 && ncols2 == 2) ||
           (ncols1 == 32 && ncols2 == 1) || (ncols1 == 64 && ncols2 == 1);
}
```

说明: sparse 检查不需要(D=256 的 may_use_sparse 恒 false, fattn-mma-f16.cuh:1758-1762);
V_is_K_view 恒 false(D=256, fattn-mma-f16.cuh:1986)。
本函数在 alloc_size(图分配期)与运行时分派两处调用, 输入只依赖张量元数据(同一图内不变),
两处结果必然一致 -> 不存在"分配无 staging 但运行要 staging"的错配。

- [x] **Step 4.2: 加量化分派函数**

```cpp
template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(DKQ == 256 && DV == 256);

    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, mask}) {
        if (t == nullptr) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // mirror of the Volta branch in ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2
    int ncols2 = 1;
    if (gqa_opt && gqa_ratio % 8 == 0) {
        ncols2 = 8;
    } else if (gqa_opt && gqa_ratio % 4 == 0) {
        ncols2 = 4;
    } else if (gqa_opt && gqa_ratio % 2 == 0) {
        ncols2 = 2;
    }

    const int ncols1 = Q->ne[1] <= 16/ncols2 ? 16/ncols2 : (Q->ne[1] <= 32/ncols2 ? 32/ncols2 : 64/ncols2);

    if (ncols2 == 2 && ncols1 == 16) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
        return;
    }
    if (ncols2 == 2 && ncols1 == 32) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
        return;
    }
    if (ncols2 == 1 && ncols1 == 32) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
        return;
    }
    if (ncols2 == 1 && ncols1 == 64) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
        return;
    }

    // combination not instantiated, use the f16 path
    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<DKQ, DV>(ctx, dst);
}
```

说明: 这里的 gqa_opt/ncols 计算与 Step 4.1 谓词逐行相同(注释互指, 两处必须同步修改);
谓词通过后组合理论上必命中前四支, 末行 fallback 仅为纵深防御。

- [x] **Step 4.3: case 256 接入**

fattn.cu 的 `ggml_cuda_flash_attn_ext_mma_f16` 内(fattn.cu:289-292):

```cpp
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            if (ggml_cuda_flash_attn_ext_mma_f16_q8_supported(dst)) {
                ggml_cuda_flash_attn_ext_mma_f16_q8<256, 256>(ctx, dst);
                break;
            }
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
```

- [x] **Step 4.4: alloc_size 接入**

fattn.cu:705-710 改为:

```cpp
    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            if (ggml_cuda_flash_attn_ext_mma_f16_q8_supported(dst)) {
                break;
            }
            need_f16_K = true;
            need_f16_V = true;
            break;
```

(量化时 need_f16_K/V 保持 false, staging 预留消失。)

- [x] **Step 4.5: 自查 + 编译门(V100 机器)**

- [ ] 谓词与分派函数逻辑逐行对照一致(注释互指)
- [ ] `GGML_TYPE_Q8_0` 在 fattn.cu 可用(已包含 ggml 头, 现有代码在用)
- [ ] 用户执行: `cmake --build build --target ggml -j` -> 通过
- [ ] 用户执行 Task 5 的完整验证命令集

- [x] **Step 4.6: Commit 2**

```bash
git add ggml/src/ggml-cuda/fattn.cu
git commit -m "cuda : FA 分派与显存分配接入 q8_0 直读"
```

### Task 5: V100 机器验证与文档

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 5.1: 编译门(用户在 V100 机器)**

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --target ggml llama-bench llama-perplexity -j
```

预期: 无模板/链接错误。

- [ ] **Step 5.2: 数值对拍(用户, 门槛: 逐 token 一致)**

```sh
GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 ./build/bin/llama-perplexity -m <model.gguf> -ngl 99 -f <text.txt> -fa 1 -ctk q8_0 -ctv q8_0 -p 2048 2>&1 | grep -E "perplexity|llama_perf" > new_path.txt
./build/bin/llama-perplexity -m <model.gguf> -ngl 99 -f <text.txt> -fa 1 -ctk q8_0 -ctv q8_0 -p 2048 2>&1 | grep -E "perplexity|llama_perf" > fallback.txt
diff new_path.txt fallback.txt
```

预期: perplexity 数值完全相同; eval 时间新路径 <= fallback(长上下文时明显更快)。
注: 逐位等价要求 `-p 2048` 足够长以触发 prefill 的 MMA 路径(ubatch 全批进 MMA)。

- [ ] **Step 5.3: 性能基准(用户)**

```sh
for p in 512 2048 16384; do
  GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 ./build/bin/llama-bench -m <model.gguf> -ngl 99 -p $p -n 64 -fa 1 -ctk q8_0 -ctv q8_0
  ./build/bin/llama-bench -m <model.gguf> -ngl 99 -p $p -n 64 -fa 1 -ctk q8_0 -ctv q8_0
  ./build/bin/llama-bench -m <model.gguf> -ngl 99 -p $p -n 64 -fa 1 -ctk f16 -ctv f16
done
```

预期(成功线): pp16384(quant 新) 比 pp16384(fallback) 提升 >= 40%; pp512 提升较小(约 10-20%);
f16 基准作为上界参考。若提升为 0 -> 检查是否实际走了 fallback(看 CUDA_LAUNCH_BLOCKING=1 下无报错
+ 用 nsys 或添加临时日志确认分派命中)。

- [ ] **Step 5.4: 显存验证(用户)**

```sh
# 16K 上下文加载后观察显存, 对比 fallback: 应少约 2x16层x4头x256x16384x2B(K+V staging) ~= 0.4 GB
GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 ./build/bin/llama-cli -m <model.gguf> -ngl 99 -p 16000 -n 1 -ctk q8_0 -ctv q8_0 -no-cnv < /dev/null & sleep 60 && nvidia-smi
# 同上不带 fallback 再测一次
```

预期: 新路径显存低约 0.4 GB(16K 时, 按 16 层 x 8(K+V) x 16384 x 4 头 x 256 x 2 B 计算)。

- [ ] **Step 5.5: 更新 CLAUDE.md**

在"P0"条目后追加状态行(实际数字待用户回填):

```markdown
P0 实施状态: 已完成 (分支 v100/mma-q8-direct)。q8_0-q8_0 + D=256 走内核内直读,
GGML_CUDA_FA_MMA_QUANT_FALLBACK=1 回退旧行为。实测: pp16384 +XX%, 显存 -X.XGB。
```

- [ ] **Step 5.6: Commit 3**

```bash
git add CLAUDE.md
git commit -m "docs : 记录 q8_0 MMA 直读优化"
```

---

## 回退方案

- 运行时: `GGML_CUDA_FA_MMA_QUANT_FALLBACK=1` 一键回旧行为
- git: 分支 v100/mma-q8-direct 独立, 可整体丢弃

## 明确的非目标(不要顺手做)

- q4_0 变体(后续 commit, 模式: type 判定从 ==Q8_0 改为按类型分派即可)
- TILE 内核(P1), VEC, sparse, cp.async 路径, 其它 head dim
- 修改上游共享的 switch_ncols 函数结构
