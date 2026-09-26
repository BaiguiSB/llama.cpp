#include "common.cuh"
#include "fattn-xqa8.cuh"

#include <cstdio>

using namespace fattn_xqa8;

// Single source of truth for the GQA-8 XQA tensor-core path, consulted by both the
// routing in ggml_cuda_get_best_fattn_kernel and ggml_cuda_flash_attn_ext_get_alloc_size.
// Only the Qwen3.6-35B-A3B decode shapes are supported; the GQA ratio check makes this
// gate mutually exclusive with the GQA-6 gate in fattn-xqa.cu (consulted first),
// everything else falls back to the regular kernel selection.
bool ggml_cuda_fattn_xqa8_supported(const int device, const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    static const bool fallback = []() {
        const char * env = getenv("GGML_CUDA_FA_XQA_FALLBACK");
        return env != nullptr && atoi(env) != 0;
    }();
    if (fallback) {
        return false;
    }

    if (!volta_mma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }

    // Only the shapes instantiated in this file are supported.
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0) {
        return false;
    }

    GGML_ASSERT(K->ne[2] > 0);
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] != 8) {
        return false; // GROUP_SIZE 8 is built into the kernel
    }

    if (Q->ne[1] < 1 || Q->ne[1] > 4) {
        return false; // rows = 8*n_tokens <= 32 = 4 M tiles
    }

    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false; // single sequence, gridDim.z holds the KV heads
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return false; // no ALiBi
    }

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false;
    }

    if (sinks != nullptr) {
        return false;
    }

    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) {
        return false;
    }

    if (K->ne[1] % 128 != 0) {
        return false; // full 128-token tiles, the KV cache padding gives multiples of 256
    }

    if (V->ne[1] != K->ne[1] || V->nb[1] != K->nb[1] || V->nb[2] != K->nb[2]) {
        return false; // V must share the layout of K
    }

    return true;
}

void ggml_cuda_flash_attn_ext_xqa8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    const int nt = (8*Q->ne[1] + 7) / 8; // == Q->ne[1], the M = 8 tiles are exactly full

    switch (nt) {
        case 1: fattn_xqa_launch<1>(ctx, dst); return;
        case 2: fattn_xqa_launch<2>(ctx, dst); return;
        case 3: fattn_xqa_launch<3>(ctx, dst); return;
        case 4: fattn_xqa_launch<4>(ctx, dst); return;
        default:
            GGML_ABORT("fatal error");
    }
}
