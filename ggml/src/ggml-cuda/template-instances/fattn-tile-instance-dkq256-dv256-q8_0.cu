// Direct q8_0 K/V loading instances for the tile kernel, see ggml_cuda_fattn_tile_q8_supported.
// The quantized load path requires FAST_FP16_AVAILABLE, this fork only builds sm_70 (V100).

#include "../fattn-tile.cuh"

#define DECL_FATTN_TILE_CASE_Q8(DKQ, DV, ncols1, ncols2)               \
    template void ggml_cuda_flash_attn_ext_tile_case_q8                \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

DECL_FATTN_TILE_CASE_Q8(256, 256,  1, 2);
DECL_FATTN_TILE_CASE_Q8(256, 256,  2, 1);
DECL_FATTN_TILE_CASE_Q8(256, 256,  2, 2);
DECL_FATTN_TILE_CASE_Q8(256, 256,  4, 1);
DECL_FATTN_TILE_CASE_Q8(256, 256,  4, 2);
DECL_FATTN_TILE_CASE_Q8(256, 256,  8, 1);
DECL_FATTN_TILE_CASE_Q8(256, 256,  8, 2);
DECL_FATTN_TILE_CASE_Q8(256, 256, 16, 1);
DECL_FATTN_TILE_CASE_Q8(256, 256, 16, 2);
