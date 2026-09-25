#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#include <cstdio>
#include <mma.h>

// Specialized FlashAttention decode kernel for Volta, Qwen3.8-27B only:
// head_dim 256, GQA ratio 6 (24 Q heads on 4 KV heads), q8_0 K and V.
// One block serves one (KV head, KV split) and computes all 6*n_tokens Q rows
// on its split, so the KV of the split is read exactly once. This removes the
// 6:1 GQA read redundancy of the vector kernel and the 3:1 of the tile kernel.
// QK and P*V both run on wmma 8x32x16 (f16 inputs, f32 accumulator). QK uses
// 4 consumer warps, each covering a 32-token slice. Softmax runs scalar, one
// warp per Q row, reading the f32 scores and writing P as f16 into a wmma A
// layout buffer; the PV fragments start from zero every tile, land in shared
// memory as f32 and are folded into the row warps' scalar running output
// there, so the online softmax rescale stays on the scalars and no
// fragment-to-lane layout of the accumulator is ever assumed.
// The M = 8 wmma tile holds the 6*n_tokens Q rows, tail rows zero-padded.
// The KV smem buffer holds one 128-dim panel of the 128-token tile at a time,
// QK accumulates over the panels and PV visits one V panel per pass.
// Design adapted from the V100 XQA kernel in the flash-attention-v100
// reference directory, no code shared.

constexpr int fattn_xqa_nthreads = 256;

// Slices of the next tile's first K panel that each thread prefetches into
// registers during softmax+PV. 2 slices = 18 registers, keeping NT=1 at ~109
// of the 128 registers available under 2 blocks/SM so nothing spills to local
// memory (4 slices pinned the allocation at 128 and spilled 10 u16 through a
// local-memory round trip).
constexpr int fattn_xqa_pf_slices = 2;

// Register variant of dequantize_V_q8_0<half, 16>, bitwise identical: pf[0..7]
// hold the raw u16 of one 16-quant slice (little-endian int8 pairs, the same
// bytes ggml_cuda_memcpy_1<16, 2> loads), pf[8] the block scale as u16.
static __device__ __forceinline__ void dequantize_V_q8_0_regs(const unsigned short * pf, void * __restrict__ dst) {
    const half2 d = __half2half2(__ushort_as_half(pf[QK8_0/4]));
#pragma unroll
    for (int k = 0; k < QK8_0/4; ++k) {
        ((half2 *) dst)[k] = d * make_half2((int8_t) (pf[k] & 0xFF), (int8_t) (pf[k] >> 8));
    }
}

template<int NT> // NT == ceil(6*n_tokens/8), number of M = 8 Q tiles
// NT=1 fits 2 blocks/SM (2*45.3 KB smem <= 96 KB), NT>=2 needs 53.8/62.2 KB
// and is smem-limited to 1 block/SM anyway. Promising 2 blocks there only
// capped ptxas at 128 registers without any occupancy gain, which spilled
// registers once the NT>=2 row loops actually executed.
__launch_bounds__(fattn_xqa_nthreads, NT >= 2 ? 1 : 2)
static __global__ void flash_attn_ext_xqa(
        const float * __restrict__ Q_ptr,   // f32 [head_dim, n_tokens, n_head]
        const void  * __restrict__ K_ptr,   // q8_0 [head_dim, n_kv, n_head_kv]
        const void  * __restrict__ V_ptr,   // q8_0, same layout as K
        const half  * __restrict__ maskh,   // f16 [n_kv, n_tokens], additive, may be null
        float       * __restrict__ dst,
        float2      * __restrict__ dst_meta,
        const float  scale,
        const int    n_tokens,
        const int    rows,                  // 6*n_tokens
        const int    n_kv,                  // K->ne[1], multiple of the KV tile size
        const int32_t ne02,                 // Q->ne[2] == n_head
        const int32_t nb01, const int32_t nb02, // Q strides in f32 elements (token, head)
        const int32_t nb11, const int32_t nb12, // K strides in bytes (token, head)
        const int32_t nb21, const int32_t nb22, // V strides in bytes
        const int64_t s31) {                // mask row stride in halves
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D        = 256;
    constexpr int TILE     = 128; // KV tokens per iteration
    constexpr int PANEL    = 128; // K/V head dims held in smem per pass
    constexpr int QSTRIDE  = 272; // head_dim + 16, keeps every row start 32 B aligned for the wmma loads
    constexpr int KVSTRIDE = 144; // PANEL + 16, same alignment guarantee
    constexpr int SPSTRIDE = 144; // TILE + 16, same guarantee for the score/P fragments

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int kv_head = blockIdx.z;
    const int split   = blockIdx.y;

    const char * K_head = (const char *) K_ptr + (int64_t) kv_head*nb12;
    const char * V_head = (const char *) V_ptr + (int64_t) kv_head*nb22;

    extern __shared__ char smem[];
    // sP holds the softmax probabilities as f16 in the row-major layout the PV
    // wmma loads as its A operand. sO is double duty: the f32 QK scores land
    // here first, and once the softmax has consumed them into registers and
    // written P, each PV pass stores its f32 P*V chunk into the same buffer
    // for the row warps to fold into the running output.
    half  (* sP )[SPSTRIDE] = (half  (*)[SPSTRIDE]) smem;
    float (* sO )[TILE]     = (float (*)[TILE])     (smem + sizeof(*sP)*8*NT);
    half  (* sQ )[QSTRIDE]  = (half  (*)[QSTRIDE])  (smem + sizeof(*sP)*8*NT + sizeof(*sO)*8*NT);
    half  (* sKV)[KVSTRIDE] = (half  (*)[KVSTRIDE]) (smem + sizeof(*sP)*8*NT + sizeof(*sO)*8*NT + sizeof(*sQ)*8*NT);

    // Q rows are head-major, r = qh*n_tokens + tok, so the zero-padded rows are
    // the tail of the last M tile.
    const int q_head0 = kv_head*6;
    for (int i = tid; i < rows*D; i += fattn_xqa_nthreads) {
        const int r  = i / D;
        const int d  = i - r*D;
        const int qh = r / n_tokens;
        const int tk = r - qh*n_tokens;
        sQ[r][d] = __float2half_rn(scale * Q_ptr[nb02*(q_head0 + qh) + nb01*tk + d]);
    }
    for (int i = tid; i < (8*NT - rows)*QSTRIDE; i += fattn_xqa_nthreads) {
        ((half *) sQ)[rows*QSTRIDE + i] = __float2half_rn(0.0f);
    }

    float row_max[NT];
    float row_sum[NT];
    float acc[NT][D/WARP_SIZE];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        row_max[j] = -FLT_MAX/2.0f;
        row_sum[j] = 0.0f;
#pragma unroll
        for (int i = 0; i < D/WARP_SIZE; ++i) {
            acc[j][i] = 0.0f;
        }
    }

    __syncthreads();

    using namespace nvcuda::wmma;

    // Register prefetch storage for the next tile's first K panel (used with
    // NT == 1 only, covering the first 64 tokens of the tile; the remaining
    // slices take the global path). pf is left uninitialized on purpose:
    // reads are guarded by pf_valid, which stays false for NT >= 2, so both
    // are dead there and get eliminated.
    unsigned short pf[fattn_xqa_pf_slices][9];
    bool pf_valid = false;

    // Balanced token ranges: split s owns [lo, hi). The ranges differ by at most
    // one token, so every split runs the same number of tile iterations (the last
    // one truncated to tv valid tokens). The old tile-strided loop gave the low
    // splits one extra full tile whenever ntiles % gridDim.y != 0 (e.g. 42 tiles
    // on 40 splits), doubling the critical path of the blocks that got them.
    const int lo = (int) (((int64_t) split*n_kv) / gridDim.y);
    const int hi = (int) (((int64_t) (split + 1)*n_kv) / gridDim.y);

    for (int k0 = lo; k0 < hi; k0 += TILE) {
        const int tv = min(TILE, hi - k0); // valid tokens in this tile
        const int ngrp = (tv + WARP_SIZE - 1) / WARP_SIZE; // 32-column softmax groups with any valid column
        // QK over dim panels, sKV holds one panel of K at a time, the
        // accumulator fragments live across the panels.
        fragment<accumulator, 8, 32, 16, float> c[NT];
#pragma unroll
        for (int mt = 0; mt < NT; ++mt) {
            fill_fragment(c[mt], 0.0f);
        }

#pragma unroll
        for (int p = 0; p < D/PANEL; ++p) {
            // Load the K panel, dequantized to f16 in registers. 8 threads per token, one 16-dim slice each.
            // Panel 0 of every tile after the range's first one comes from the
            // register prefetch (use_pf, only the first fattn_xqa_pf_slices j
            // slots hold data, the rest take the global path); the j-loop form
            // keeps pf[j] indexed by an unrolled constant so the array stays
            // in registers.
            const int nslices = tv*(PANEL/16);
            const bool use_pf = (p == 0) && pf_valid;
#pragma unroll
            for (int j = 0; j < TILE*(PANEL/16)/fattn_xqa_nthreads; ++j) {
                const int i = tid + j*fattn_xqa_nthreads;
                if (i < nslices) {
                    const int t  = i / (PANEL/16);
                    const int sl = i - t*(PANEL/16);
                    if (use_pf && j < fattn_xqa_pf_slices) {
                        dequantize_V_q8_0_regs(pf[j], &sKV[t][sl*16]);
                    } else {
                        dequantize_V_q8_0<half, 16>(K_head + (int64_t) (k0 + t)*nb11, &sKV[t][sl*16], p*PANEL + sl*16);
                    }
                }
            }
            __syncthreads();

            // Tensor cores, one warp per 32-token slice. Slices entirely beyond
            // a truncated tail are skipped; their stale sO columns are masked in
            // the softmax below.
            if (warp*WARP_SIZE < tv) {
                const int slice = warp*32;
                fragment<matrix_b, 8, 32, 16, half, col_major> b;
#pragma unroll
                for (int k = 0; k < PANEL/16; ++k) {
                    load_matrix_sync(b, &sKV[slice][k*16], KVSTRIDE);
#pragma unroll
                    for (int mt = 0; mt < NT; ++mt) {
                        fragment<matrix_a, 8, 32, 16, half, row_major> a;
                        load_matrix_sync(a, &sQ[mt*8][p*PANEL + k*16], QSTRIDE);
                        mma_sync(c[mt], a, b, c[mt]);
                    }
                }
            }
            __syncthreads();
        }

        if (warp*WARP_SIZE < tv) {
            const int slice = warp*32;
#pragma unroll
            for (int mt = 0; mt < NT; ++mt) {
                store_matrix_sync(&sO[mt*8][slice], c[mt], TILE, mem_row_major);
            }
        }
        __syncthreads();

        // Load the first V panel; softmax and the tensor-core PV follow. The
        // second panel reuses the buffer, P stays in sP.
        for (int i = tid; i < tv*(PANEL/16); i += fattn_xqa_nthreads) {
            const int t  = i / (PANEL/16);
            const int sl = i - t*(PANEL/16);
            dequantize_V_q8_0<half, 16>(V_head + (int64_t) (k0 + t)*nb21, &sKV[t][sl*16], sl*16);
        }
        __syncthreads();

        // Issue the global loads for the next tile's first K panel here, so
        // they overlap with softmax + PV (the longest compute window). The
        // dequantize at the top of the next iteration then reads registers
        // instead of stalling on global latency (long_sb/lg_throttle were
        // ~30% of stall cycles). Same (t, sl) slice mapping as the load loop;
        // fattn_xqa_pf_slices slices per thread cover the first 64 tokens.
        if constexpr (NT == 1) {
            pf_valid = false;
            const int nk0 = k0 + TILE;
            if (nk0 < hi) {
                const int nslices = min(TILE, hi - nk0)*(PANEL/16);
#pragma unroll
                for (int j = 0; j < fattn_xqa_pf_slices; ++j) {
                    const int i = tid + j*fattn_xqa_nthreads;
                    if (i < nslices) {
                        const int t  = i / (PANEL/16);
                        const int sl = i - t*(PANEL/16);
                        // Slice sl covers dims sl*16..sl*16+15 = half of q8_0
                        // block sl/2 (QK8_0 == 32 == 2 slices per block).
                        const block_q8_0 * blk = (const block_q8_0 *) (K_head + (int64_t) (nk0 + t)*nb11) + sl/2;
                        const unsigned short * qs = (const unsigned short *) (blk->qs + (sl % 2)*16);
#pragma unroll
                        for (int k = 0; k < QK8_0/4; ++k) {
                            pf[j][k] = qs[k];
                        }
                        pf[j][QK8_0/4] = __half_as_ushort(blk->d);
                    }
                }
                pf_valid = true;
            }
        }

        // Online softmax, one warp per Q row. The f32 scores are read from sO
        // and P is written as f16 into sP, in the layout the PV wmma loads as
        // its A operand. The rescale of the running output is deferred to the
        // panel-0 accumulation below, so the wmma fragments never need a
        // (layout-dependent) per-row rescale.
        float ms_tile[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int r = warp + j*8; // 8 rows per M tile, one warp per row
            if (r >= rows) {
                continue;
            }
            const int tk = r % n_tokens;
            const half * mrow = maskh ? maskh + tk*s31 + k0 : nullptr;

            float mx = row_max[j];
            float vals[TILE/WARP_SIZE];
            // tv is warp-uniform, so every i iteration covers one contiguous
            // 32-column group: fully valid groups need no tail predicate (the
            // common full-tile path is predicate-free again), fully invalid
            // groups contribute nothing and are skipped, only the boundary
            // group keeps the per-column tail handling.
#pragma unroll
            for (int i = 0; i < TILE/WARP_SIZE; ++i) {
                if (i >= ngrp) {
                    vals[i] = -INFINITY; // never read: the exp loop guards the same way
                    continue;
                }
                const int t = lane + i*WARP_SIZE;
                // Boundary group only: tail columns (t >= tv) hold stale/garbage
                // scores from skipped MMA slices and the mask column can be out
                // of range. Overwrite with -inf (not add, so no NaN can leak
                // into mx); expf(-inf - mx) == 0 then zeroes their probability.
                float v;
                if ((i + 1)*WARP_SIZE <= tv) {
                    // Fully valid group.
                    v = (mrow ? sO[r][t] + __half2float(mrow[t]) : sO[r][t]);
                } else {
                    v = (t >= tv) ? -INFINITY : (mrow ? sO[r][t] + __half2float(mrow[t]) : sO[r][t]);
                }
                vals[i] = v;
                mx = fmaxf(mx, v + FATTN_KQ_MAX_OFFSET);
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, offset));
            }

            const float m_scale = expf(row_max[j] - mx);
            float msum = 0.0f;
#pragma unroll
            for (int i = 0; i < TILE/WARP_SIZE; ++i) {
                if (i >= ngrp) {
                    continue;
                }
                const int t = lane + i*WARP_SIZE;
                const float prob = expf(vals[i] - mx);
                // Every column of a group with at least one valid token gets a
                // P entry: the boundary group's k chunks overlap the valid range
                // in the PV wmma below, so its tail columns must hold exact
                // zeros there (expf of the -INFINITY tails is 0.0f). Groups past
                // ngrp are skipped here and their k chunks are skipped by the PV
                // loop bound, so stale data never enters a product.
                if ((i + 1)*WARP_SIZE <= tv || t < tv) {
                    sP[r][t] = __float2half_rn(prob);
                }
                msum += prob;
            }
            msum = warp_reduce_sum(msum);

            row_sum[j] = row_sum[j]*m_scale + msum;
            row_max[j] = mx;
            ms_tile[j] = m_scale;
        }
        __syncthreads(); // P complete: the PV below reads the rows of all warps

        // P*V on tensor cores, warps 0-3 each own a 32-dim slice of the V panel
        // (the same consumer warps that produced the scores). The f32 result
        // goes through sO to the row warps; the fragment starts from zero every
        // tile and is never rescaled, so its lane layout is never assumed.
        auto pv_mma = [&]() {
#pragma unroll
            for (int mt = 0; mt < NT; ++mt) {
                fragment<accumulator, 8, 32, 16, float> c;
                fill_fragment(c, 0.0f);
                for (int kc = 0; 16*kc < tv; ++kc) {
                    fragment<matrix_a, 8, 32, 16, half, row_major> a;
                    load_matrix_sync(a, &sP[mt*8][16*kc], SPSTRIDE);
                    fragment<matrix_b, 8, 32, 16, half, row_major> b;
                    load_matrix_sync(b, &sKV[16*kc][warp*32], KVSTRIDE);
                    mma_sync(c, a, b, c);
                }
                store_matrix_sync(&sO[mt*8][warp*32], c, TILE, mem_row_major);
            }
        };
        if (warp < 4) {
            pv_mma();
        }
        __syncthreads(); // sO ready; sKV free for the second V panel

        // Fold the panel chunk into the running output, one warp per Q row.
        // Panel 0 applies the softmax rescale deferred from the softmax above.
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int r = warp + j*8; // 8 rows per M tile, one warp per row
            if (r >= rows) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < PANEL/WARP_SIZE; ++i) {
                acc[j][i] = acc[j][i]*ms_tile[j] + sO[r][i*WARP_SIZE + lane];
            }
        }

        // Second V panel, PV only. P and the rescaled output are in place.
        for (int i = tid; i < tv*(PANEL/16); i += fattn_xqa_nthreads) {
            const int t  = i / (PANEL/16);
            const int sl = i - t*(PANEL/16);
            dequantize_V_q8_0<half, 16>(V_head + (int64_t) (k0 + t)*nb21, &sKV[t][sl*16], PANEL + sl*16);
        }
        __syncthreads();

        if (warp < 4) {
            pv_mma();
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int r = warp + j*8; // 8 rows per M tile, one warp per row
            if (r >= rows) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < PANEL/WARP_SIZE; ++i) {
                acc[j][PANEL/WARP_SIZE + i] += sO[r][i*WARP_SIZE + lane];
            }
        }
        __syncthreads();
    }

    // Epilogue, the writeout mirrors flash_attn_ext_vec.
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int r = warp + j*8; // 8 rows per M tile, one warp per row
        if (r >= rows) {
            continue;
        }
        const int qh   = r / n_tokens;
        const int tk   = r - qh*n_tokens;
        const int head = kv_head*6 + qh;
        const int64_t row = tk*ne02 + head;

        if (gridDim.y == 1) {
#pragma unroll
            for (int i = 0; i < D/WARP_SIZE; ++i) {
                dst[row*D + lane + i*WARP_SIZE] = acc[j][i] / row_sum[j];
            }
        } else {
#pragma unroll
            for (int i = 0; i < D/WARP_SIZE; ++i) {
                dst[(row*gridDim.y + split)*D + lane + i*WARP_SIZE] = acc[j][i];
            }
            if (lane == 0) {
                dst_meta[row*gridDim.y + split] = make_float2(row_max[j], row_sum[j]);
            }
        }
    }
#else // FLASH_ATTN_AVAILABLE
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, maskh, dst, dst_meta, scale, n_tokens, rows, n_kv,
        ne02, nb01, nb02, nb11, nb12, nb21, nb22, s31);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int NT>
static void fattn_xqa_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int D = 256;
    constexpr int TILE = 128;
    constexpr int QSTRIDE  = 272;
    constexpr int KVSTRIDE = 144;
    constexpr int SPSTRIDE = 144;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const int n_tokens = Q->ne[1];
    const int rows     = 6*n_tokens;

    float scale = 1.0f;
    memcpy(&scale, (const float *) KQV->op_params + 0, sizeof(float));

    constexpr size_t nbytes_shared = sizeof(half)*8*NT*SPSTRIDE + sizeof(float)*8*NT*TILE
                                   + sizeof(half)*(8*NT*QSTRIDE + TILE*KVSTRIDE);

    // Enough KV splits to fill the GPU, each split reads its tiles exactly once.
    // The kernel is memory bound, more splits than two waves only add combine work.
    const int ntiles = K->ne[1] / TILE;
    const int parallel_blocks = std::min(std::max(2*nsm / (int) K->ne[2], 1), ntiles);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "%s: FA XQA tensor-core path taken, NT = %d, rows = %d, n_tokens = %d, parallel_blocks = %d\n",
                __func__, NT, rows, n_tokens, parallel_blocks);
    }

    ggml_cuda_pool_alloc<float> dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    CUDA_SET_SHARED_MEMORY_LIMIT(flash_attn_ext_xqa<NT>, nbytes_shared);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    {
        // Prefer the full shared memory carveout so 2 blocks can be resident per SM where the smem fits.
        static bool carveout_set[GGML_CUDA_MAX_DEVICES] = { false };
        if (!carveout_set[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(flash_attn_ext_xqa<NT>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            carveout_set[id] = true;
        }
    }
#endif

    const dim3 block_dim(fattn_xqa_nthreads, 1, 1);
    const dim3 blocks_num(1, parallel_blocks, K->ne[2]);

    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, nbytes_shared, stream);
    ggml_cuda_kernel_launch(flash_attn_ext_xqa<NT>, launch_params,
        (const float *) Q->data,
        K->data,
        V->data,
        mask ? (const half *) mask->data : nullptr,
        parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, n_tokens, rows, K->ne[1],
        Q->ne[2], Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float),
        K->nb[1], K->nb[2],
        V->nb[1], V->nb[2],
        mask ? mask->nb[1]/sizeof(half) : 0);
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params_combine(blocks_num_combine, block_dim_combine, nbytes_shared_combine, stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<D>, launch_params_combine,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_fattn_xqa_supported(const int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_xqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
