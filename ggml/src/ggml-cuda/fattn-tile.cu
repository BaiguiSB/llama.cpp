#include "common.cuh"
#include "fattn-tile.cuh"

// Single source of truth for the tile q8_0 direct-loading path, consulted by both
// ggml_cuda_flash_attn_ext_tile (dispatch) and ggml_cuda_flash_attn_ext_get_alloc_size
// (f16 staging reservation). Must mirror launch_fattn_tile_switch_ncols2/_ncols1.
bool ggml_cuda_fattn_tile_q8_supported(const ggml_tensor * dst, int * ncols1_out, int * ncols2_out) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // only (DKQ, DV) == (256, 256) has q8_0 instances
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0) {
        return false;
    }

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false; // no use_logit_softcap instances for q8_0
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // Mirror of launch_fattn_tile_switch_ncols2 (NVIDIA, DV == 256):
    const bool nvidia      = GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
    const int  gqa_limit   = nvidia && gqa_ratio <= 4 && 256 <= 256 ? 16 : INT_MAX;
    const bool use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;

    int ncols2 = 1;
    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ncols2 = 8;
    } else if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ncols2 = 4;
    } else if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ncols2 = 2;
    }

    if (ncols2 > 2) {
        return false; // no q8_0 instances for ncols2 == 4/8 (gqa_ratio divisible by 4)
    }

    // Mirror of launch_fattn_tile_switch_ncols1 (NVIDIA branch, DKQ <= 256), ncols2 <= 2 here:
    int ncols1;
    if (Q->ne[1] > 16/ncols2) {
        ncols1 = 32/ncols2;
    } else if (Q->ne[1] > 8/ncols2) {
        ncols1 = 16/ncols2;
    } else if (Q->ne[1] > 4/ncols2) {
        ncols1 = 8/ncols2;
    } else if (Q->ne[1] > 2/ncols2) {
        ncols1 = 4/ncols2;
    } else {
        ncols1 = 2/ncols2;
    }

    if (ncols1 > 16) {
        return false; // ncols1 == 32 has no q8_0 instance (unreachable via the Volta routing anyway)
    }

    if (ncols1_out != nullptr) {
        *ncols1_out = ncols1;
    }
    if (ncols2_out != nullptr) {
        *ncols2_out = ncols2;
    }
    return true;
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_fattn_tile_q8_supported(dst)) {
        ggml_cuda_flash_attn_ext_tile_q8<256, 256>(ctx, dst);
        return;
    }

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}
