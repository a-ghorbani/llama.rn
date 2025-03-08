#import "ggml-metal.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal-impl.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max memory buffers that can be mapped to the device
#define LM_GGML_METAL_MAX_BUFFERS 64

// max number of MTLCommandBuffer used to submit a graph for processing
#define LM_GGML_METAL_MAX_COMMAND_BUFFERS 8

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define LM_GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// globals

// overload of MTLGPUFamilyMetal3 (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;

// initialized in lm_ggml_backend_metal_reg
static struct lm_ggml_backend_reg    g_lm_ggml_backend_metal_reg;
static struct lm_ggml_backend_device g_lm_ggml_backend_metal_device;

// information about a Metal device
// note: assumes single GPU device - the default one
// TODO: support multiple GPU devices
static struct lm_ggml_backend_metal_device_context {
    id<MTLDevice> mtl_device;
    int           mtl_device_ref_count;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_residency_sets;
    bool has_bfloat;
    bool use_bfloat;

    char name[128];
} g_lm_ggml_ctx_dev_main = {
    /*.mtl_device              =*/ nil,
    /*.mtl_device_ref_count    =*/ 0,
    /*.has_simdgroup_reduction =*/ false,
    /*.has_simdgroup_mm        =*/ false,
    /*.has_residency_sets      =*/ false,
    /*.has_bfloat              =*/ false,
    /*.use_bfloat              =*/ false,
    /*.name                    =*/ "",
};

// acquire
static id<MTLDevice> lm_ggml_backend_metal_device_acq(struct lm_ggml_backend_metal_device_context * ctx) {
    assert(ctx != NULL);

    if (ctx->mtl_device == nil) {
        ctx->mtl_device = MTLCreateSystemDefaultDevice();
    }

    if (ctx->mtl_device) {
        ctx->has_simdgroup_reduction  = [ctx->mtl_device supportsFamily:MTLGPUFamilyApple7];
        ctx->has_simdgroup_reduction |= [ctx->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

        ctx->has_simdgroup_mm = [ctx->mtl_device supportsFamily:MTLGPUFamilyApple7];

#if defined(LM_GGML_METAL_HAS_RESIDENCY_SETS)
        ctx->has_residency_sets = getenv("LM_GGML_METAL_NO_RESIDENCY") == NULL;
#endif

        ctx->has_bfloat  = [ctx->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
        ctx->has_bfloat |= [ctx->mtl_device supportsFamily:MTLGPUFamilyApple6];

#if defined(LM_GGML_METAL_USE_BF16)
        ctx->use_bfloat = ctx->has_bfloat;
#else
        ctx->use_bfloat = false;
#endif

        strncpy(ctx->name, [[ctx->mtl_device name] UTF8String], sizeof(ctx->name) - 1);
    }

    ctx->mtl_device_ref_count++;

    return ctx->mtl_device;
}

// release
static void lm_ggml_backend_metal_device_rel(struct lm_ggml_backend_metal_device_context * ctx) {
    assert(ctx != NULL);
    assert(ctx->mtl_device_ref_count > 0);

    ctx->mtl_device_ref_count--;

    if (ctx->mtl_device_ref_count == 0) {
        if (ctx->mtl_device) {
            [ctx->mtl_device release];
            ctx->mtl_device = nil;
        }
    }
}

// kernels

struct lm_ggml_metal_kernel {
    id<MTLComputePipelineState> pipeline;
};

enum lm_ggml_metal_kernel_type {
    LM_GGML_METAL_KERNEL_TYPE_ADD,
    LM_GGML_METAL_KERNEL_TYPE_ADD_ROW,
    LM_GGML_METAL_KERNEL_TYPE_SUB,
    LM_GGML_METAL_KERNEL_TYPE_SUB_ROW,
    LM_GGML_METAL_KERNEL_TYPE_MUL,
    LM_GGML_METAL_KERNEL_TYPE_MUL_ROW,
    LM_GGML_METAL_KERNEL_TYPE_DIV,
    LM_GGML_METAL_KERNEL_TYPE_DIV_ROW,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_F32,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_F16,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_I32,
    LM_GGML_METAL_KERNEL_TYPE_REPEAT_I16,
    LM_GGML_METAL_KERNEL_TYPE_SCALE,
    LM_GGML_METAL_KERNEL_TYPE_SCALE_4,
    LM_GGML_METAL_KERNEL_TYPE_CLAMP,
    LM_GGML_METAL_KERNEL_TYPE_TANH,
    LM_GGML_METAL_KERNEL_TYPE_RELU,
    LM_GGML_METAL_KERNEL_TYPE_SIGMOID,
    LM_GGML_METAL_KERNEL_TYPE_GELU,
    LM_GGML_METAL_KERNEL_TYPE_GELU_4,
    LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK,
    LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK_4,
    LM_GGML_METAL_KERNEL_TYPE_SILU,
    LM_GGML_METAL_KERNEL_TYPE_SILU_4,
    LM_GGML_METAL_KERNEL_TYPE_ELU,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16_4,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32,
    LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32_4,
    LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF,
    LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F32,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F16,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_BF16,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS,
    LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_I32,
    LM_GGML_METAL_KERNEL_TYPE_RMS_NORM,
    LM_GGML_METAL_KERNEL_TYPE_GROUP_NORM,
    LM_GGML_METAL_KERNEL_TYPE_NORM,
    LM_GGML_METAL_KERNEL_TYPE_SSM_CONV_F32,
    LM_GGML_METAL_KERNEL_TYPE_SSM_SCAN_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_1ROW,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_L4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_BF16,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_2,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_3,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_4,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_5,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32,
  //LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_1ROW,
  //LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_L4,
  //LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F16,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F32,
    LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F32,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F32,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F16,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F32,
    LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F16,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_F16,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_F32,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F16,
    LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F32,
    LM_GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_UPSCALE_F32,
    LM_GGML_METAL_KERNEL_TYPE_PAD_F32,
    LM_GGML_METAL_KERNEL_TYPE_PAD_REFLECT_1D_F32,
    LM_GGML_METAL_KERNEL_TYPE_ARANGE_F32,
    LM_GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32,
    LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC,
    LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC,
    LM_GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H64,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H80,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H96,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H112,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H128,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H256,
    LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H256,
    LM_GGML_METAL_KERNEL_TYPE_SET_I32,
    LM_GGML_METAL_KERNEL_TYPE_SET_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_BF16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_BF16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1,
    LM_GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F16,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F32,
    LM_GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F16,
    LM_GGML_METAL_KERNEL_TYPE_CONCAT,
    LM_GGML_METAL_KERNEL_TYPE_SQR,
    LM_GGML_METAL_KERNEL_TYPE_SQRT,
    LM_GGML_METAL_KERNEL_TYPE_SIN,
    LM_GGML_METAL_KERNEL_TYPE_COS,
    LM_GGML_METAL_KERNEL_TYPE_SUM_ROWS,
    LM_GGML_METAL_KERNEL_TYPE_POOL_2D_AVG_F32,
    LM_GGML_METAL_KERNEL_TYPE_POOL_2D_MAX_F32,
    LM_GGML_METAL_KERNEL_TYPE_ARGMAX,

    LM_GGML_METAL_KERNEL_TYPE_COUNT
};

struct lm_ggml_backend_metal_context {
    id<MTLCommandQueue> queue;

    dispatch_queue_t d_queue;

    struct lm_ggml_metal_kernel kernels[LM_GGML_METAL_KERNEL_TYPE_COUNT];

    // capture state
    bool capture_next_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct lm_ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    id<MTLCommandBuffer> command_buffers[LM_GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // abort lm_ggml_metal_graph_compute if callback returns true
    lm_ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
// static NSString * const msl_library_source = @"see metal.metal";

#if !LM_GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface LMGGMLMetalClass : NSObject
@end
@implementation LMGGMLMetalClass
@end
#endif

static void * lm_ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        LM_GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        LM_GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

static struct lm_ggml_backend_metal_context * lm_ggml_metal_init(lm_ggml_backend_dev_t dev) {
    LM_GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !LM_GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        LM_GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    struct lm_ggml_backend_metal_context * ctx = calloc(1, sizeof(struct lm_ggml_backend_metal_context));
    struct lm_ggml_backend_metal_device_context * ctx_dev = dev->context;

    id<MTLDevice> device = lm_ggml_backend_metal_device_acq(ctx_dev);
    LM_GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    ctx->queue  = [device newCommandQueue];
    if (ctx->queue == nil) {
        LM_GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    id<MTLLibrary> metal_library = nil;

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    {
        NSError * error = nil;
        NSString * src = nil;

#if LM_GGML_METAL_EMBED_LIBRARY
        LM_GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

        extern const char lm_ggml_metallib_start[];
        extern const char lm_ggml_metallib_end[];

        src = [[NSString alloc] initWithBytes:lm_ggml_metallib_start length:(lm_ggml_metallib_end-lm_ggml_metallib_start) encoding:NSUTF8StringEncoding];

#else

#ifdef SWIFT_PACKAGE
        NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
        NSBundle * bundle = [NSBundle bundleForClass:[LMGGMLMetalClass class]];
#endif

#if TARGET_OS_SIMULATOR
        NSString * path_lib = [bundle pathForResource:@"ggml-llama-sim" ofType:@"metallib"];
#else
        NSString * path_lib = [bundle pathForResource:@"ggml-llama" ofType:@"metallib"];
#endif
        if (path_lib == nil) {
            // Try to find the resource in the directory where the current binary located.
            NSString * current_binary = [[NSProcessInfo processInfo] arguments][0];
            NSString * bin_dir = [current_binary stringByDeletingLastPathComponent];
            NSString * default_metallib_path = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
            if ([[NSFileManager defaultManager] isReadableFileAtPath:default_metallib_path]) {
                LM_GGML_LOG_INFO("%s: found '%s'\n", __func__, [default_metallib_path UTF8String]);
                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:default_metallib_path error:&error];
                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                    // Optionally, if this is a symlink, try to resolve it.
                    default_metallib_path = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:default_metallib_path error:&error];
                    if (default_metallib_path && [default_metallib_path length] > 0 && ![[default_metallib_path substringToIndex:1] isEqualToString:@"/"]) {
                        // It is a relative path, adding the binary directory as directory prefix.
                        default_metallib_path = [NSString pathWithComponents:@[bin_dir, default_metallib_path]];
                    }
                    if (!default_metallib_path || ![[NSFileManager defaultManager] isReadableFileAtPath:default_metallib_path]) {
                        // Link to the resource could not be resolved.
                        default_metallib_path = nil;
                    } else {
                        LM_GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [default_metallib_path UTF8String]);
                    }
                }
            } else {
                // The resource couldn't be found in the binary's directory.
                default_metallib_path = nil;
            }
            path_lib = default_metallib_path;
        }

        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            LM_GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            metal_library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                LM_GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
        } else {
            LM_GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"LM_GGML_METAL_PATH_RESOURCES"];

            LM_GGML_LOG_INFO("%s: LM_GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                LM_GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            LM_GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                LM_GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
        }
#endif

        if (!metal_library) {
            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ctx_dev->use_bfloat) {
                    [prep setObject:@"1" forKey:@"LM_GGML_METAL_USE_BF16"];
                }

#if LM_GGML_METAL_EMBED_LIBRARY
                [prep setObject:@"1" forKey:@"LM_GGML_METAL_EMBED_LIBRARY"];
#endif

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                //[options setFastMathEnabled:false];

                metal_library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    LM_GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return NULL;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
        }

#if LM_GGML_METAL_EMBED_LIBRARY
        [src release];
#endif // LM_GGML_METAL_EMBED_LIBRARY
    }

    // print MTL GPU family:
    LM_GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, [[device name] UTF8String]);

    // determine max supported GPU family
    // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
    // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    {
        for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
            if ([device supportsFamily:i]) {
                LM_GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                break;
            }
        }

        for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
            if ([device supportsFamily:i]) {
                LM_GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                break;
            }
        }

        for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
            if ([device supportsFamily:i]) {
                LM_GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                break;
            }
        }
    }

    LM_GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, ctx_dev->has_simdgroup_reduction     ? "true" : "false");
    LM_GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, ctx_dev->has_simdgroup_mm            ? "true" : "false");
    LM_GGML_LOG_INFO("%s: has residency sets    = %s\n", __func__, ctx_dev->has_residency_sets          ? "true" : "false");
    LM_GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, ctx_dev->has_bfloat                  ? "true" : "false");
    LM_GGML_LOG_INFO("%s: use bfloat            = %s\n", __func__, ctx_dev->use_bfloat                  ? "true" : "false");
    LM_GGML_LOG_INFO("%s: hasUnifiedMemory      = %s\n", __func__, ctx_dev->mtl_device.hasUnifiedMemory ? "true" : "false");

    ctx->capture_next_compute = false;
    ctx->capture_started = false;
    ctx->capture_scope = nil;

    ctx->gf = nil;
    ctx->encode_async = nil;
    for (int i = 0; i < LM_GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        ctx->command_buffers[i] = nil;
    }

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        LM_GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, device.recommendedMaxWorkingSetSize / 1e6);
    }
#endif

    // load kernels
    {
        NSError * error = nil;

        for (int i = 0; i < LM_GGML_METAL_KERNEL_TYPE_COUNT; ++i) {
            ctx->kernels[i].pipeline = nil;
        }

#define LM_GGML_METAL_ADD_KERNEL(e, name, supported) \
        if (supported) { \
            struct lm_ggml_metal_kernel * kernel = &ctx->kernels[e]; \
            id<MTLFunction> metal_function = [metal_library newFunctionWithName:@"kernel_"#name]; \
            kernel->pipeline = [device newComputePipelineStateWithFunction:metal_function error:&error]; \
            LM_GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, "kernel_"#name, (void *) kernel->pipeline, \
                    (int) kernel->pipeline.maxTotalThreadsPerThreadgroup, \
                    (int) kernel->pipeline.threadExecutionWidth); \
            [metal_function release]; \
            if (error) { \
                LM_GGML_LOG_ERROR("%s: error: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
                [metal_library release]; \
                return NULL; \
            } \
        } else { \
            LM_GGML_LOG_WARN("%s: skipping %-40s (not supported)\n", __func__, "kernel_"#name); \
        }

        const bool has_simdgroup_mm        = ctx_dev->has_simdgroup_mm;
        const bool has_simdgroup_reduction = ctx_dev->has_simdgroup_reduction;
        const bool use_bfloat              = ctx_dev->use_bfloat;

        // simd_sum and simd_max requires MTLGPUFamilyApple7

        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ADD,                           add,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ADD_ROW,                       add_row,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SUB,                           sub,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SUB_ROW,                       sub_row,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL,                           mul,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_ROW,                       mul_row,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_DIV,                           div,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_DIV_ROW,                       div_row,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_REPEAT_F32,                    repeat_f32,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_REPEAT_F16,                    repeat_f16,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_REPEAT_I32,                    repeat_i32,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_REPEAT_I16,                    repeat_i16,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SCALE,                         scale,                          true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SCALE_4,                       scale_4,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CLAMP,                         clamp,                          true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_TANH,                          tanh,                           true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_RELU,                          relu,                           true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SIGMOID,                       sigmoid,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GELU,                          gelu,                           true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GELU_4,                        gelu_4,                         true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK,                    gelu_quick,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK_4,                  gelu_quick_4,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SILU,                          silu,                           true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SILU_4,                        silu_4,                         true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ELU,                           elu,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16,                  soft_max_f16,                   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16_4,                soft_max_f16_4,                 has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32,                  soft_max_f32,                   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32_4,                soft_max_f32_4,                 has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF,                 diag_mask_inf,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8,               diag_mask_inf_8,                true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F32,                  get_rows_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F16,                  get_rows_f16,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_BF16,                 get_rows_bf16,                  use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0,                 get_rows_q4_0,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1,                 get_rows_q4_1,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0,                 get_rows_q5_0,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1,                 get_rows_q5_1,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0,                 get_rows_q8_0,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K,                 get_rows_q2_K,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K,                 get_rows_q3_K,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K,                 get_rows_q4_K,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K,                 get_rows_q5_K,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K,                 get_rows_q6_K,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS,              get_rows_iq2_xxs,               true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS,               get_rows_iq2_xs,                true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS,              get_rows_iq3_xxs,               true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S,                get_rows_iq3_s,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S,                get_rows_iq2_s,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S,                get_rows_iq1_s,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M,                get_rows_iq1_m,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL,               get_rows_iq4_nl,                true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS,               get_rows_iq4_xs,                true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_I32,                  get_rows_i32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_RMS_NORM,                      rms_norm,                       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_GROUP_NORM,                    group_norm,                     has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_NORM,                          norm,                           true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SSM_CONV_F32,                  ssm_conv_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SSM_SCAN_F32,                  ssm_scan_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32,                mul_mv_f32_f32,                 has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32,               mul_mv_bf16_f32,                has_simdgroup_reduction && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_1ROW,          mul_mv_bf16_f32_1row,           has_simdgroup_reduction && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_L4,            mul_mv_bf16_f32_l4,             has_simdgroup_reduction && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_BF16,              mul_mv_bf16_bf16,               has_simdgroup_reduction && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32,                mul_mv_f16_f32,                 has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW,           mul_mv_f16_f32_1row,            has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4,             mul_mv_f16_f32_l4,              has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16,                mul_mv_f16_f16,                 has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32,               mul_mv_q4_0_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32,               mul_mv_q4_1_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32,               mul_mv_q5_0_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32,               mul_mv_q5_1_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32,               mul_mv_q8_0_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_2,       mul_mv_ext_f16_f32_r1_2,        has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_3,       mul_mv_ext_f16_f32_r1_3,        has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_4,       mul_mv_ext_f16_f32_r1_4,        has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_5,       mul_mv_ext_f16_f32_r1_5,        has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_2,      mul_mv_ext_q4_0_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_3,      mul_mv_ext_q4_0_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_4,      mul_mv_ext_q4_0_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_5,      mul_mv_ext_q4_0_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_2,      mul_mv_ext_q4_1_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_3,      mul_mv_ext_q4_1_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_4,      mul_mv_ext_q4_1_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_5,      mul_mv_ext_q4_1_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_2,      mul_mv_ext_q5_0_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_3,      mul_mv_ext_q5_0_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_4,      mul_mv_ext_q5_0_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_5,      mul_mv_ext_q5_0_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_2,      mul_mv_ext_q5_1_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_3,      mul_mv_ext_q5_1_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_4,      mul_mv_ext_q5_1_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_5,      mul_mv_ext_q5_1_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_2,      mul_mv_ext_q8_0_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_3,      mul_mv_ext_q8_0_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_4,      mul_mv_ext_q8_0_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_5,      mul_mv_ext_q8_0_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_2,      mul_mv_ext_q4_K_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_3,      mul_mv_ext_q4_K_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_4,      mul_mv_ext_q4_K_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_5,      mul_mv_ext_q4_K_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_2,      mul_mv_ext_q5_K_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_3,      mul_mv_ext_q5_K_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_4,      mul_mv_ext_q5_K_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_5,      mul_mv_ext_q5_K_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_2,      mul_mv_ext_q6_K_f32_r1_2,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_3,      mul_mv_ext_q6_K_f32_r1_3,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_4,      mul_mv_ext_q6_K_f32_r1_4,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_5,      mul_mv_ext_q6_K_f32_r1_5,       has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_2,    mul_mv_ext_iq4_nl_f32_r1_2,     has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_3,    mul_mv_ext_iq4_nl_f32_r1_3,     has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_4,    mul_mv_ext_iq4_nl_f32_r1_4,     has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_5,    mul_mv_ext_iq4_nl_f32_r1_5,     has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32,               mul_mv_q2_K_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32,               mul_mv_q3_K_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32,               mul_mv_q4_K_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32,               mul_mv_q5_K_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32,               mul_mv_q6_K_f32,                has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32,            mul_mv_iq2_xxs_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32,             mul_mv_iq2_xs_f32,              has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32,            mul_mv_iq3_xxs_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32,              mul_mv_iq3_s_f32,               has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32,              mul_mv_iq2_s_f32,               has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32,              mul_mv_iq1_s_f32,               has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32,              mul_mv_iq1_m_f32,               has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32,             mul_mv_iq4_nl_f32,              has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32,             mul_mv_iq4_xs_f32,              has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32,             mul_mv_id_f32_f32,              has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32,             mul_mv_id_f16_f32,              has_simdgroup_reduction);
      //LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_1ROW,        mul_mv_id_f16_f32_1row,         has_simdgroup_reduction);
      //LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_L4,          mul_mv_id_f16_f32_l4,           has_simdgroup_reduction);
      //LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F16,             mul_mv_id_f16_f16,              has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_BF16_F32,            mul_mv_id_bf16_f32,             has_simdgroup_reduction && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32,            mul_mv_id_q4_0_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32,            mul_mv_id_q4_1_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32,            mul_mv_id_q5_0_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32,            mul_mv_id_q5_1_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32,            mul_mv_id_q8_0_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32,            mul_mv_id_q2_K_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32,            mul_mv_id_q3_K_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32,            mul_mv_id_q4_K_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32,            mul_mv_id_q5_K_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32,            mul_mv_id_q6_K_f32,             has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32,         mul_mv_id_iq2_xxs_f32,          has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32,          mul_mv_id_iq2_xs_f32,           has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32,         mul_mv_id_iq3_xxs_f32,          has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32,           mul_mv_id_iq3_s_f32,            has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32,           mul_mv_id_iq2_s_f32,            has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32,           mul_mv_id_iq1_s_f32,            has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32,           mul_mv_id_iq1_m_f32,            has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32,          mul_mv_id_iq4_nl_f32,           has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32,          mul_mv_id_iq4_xs_f32,           has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32,                mul_mm_f32_f32,                 has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32,                mul_mm_f16_f32,                 has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_BF16_F32,               mul_mm_bf16_f32,                has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32,               mul_mm_q4_0_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32,               mul_mm_q4_1_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32,               mul_mm_q5_0_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32,               mul_mm_q5_1_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32,               mul_mm_q8_0_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32,               mul_mm_q2_K_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32,               mul_mm_q3_K_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32,               mul_mm_q4_K_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32,               mul_mm_q5_K_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32,               mul_mm_q6_K_f32,                has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32,            mul_mm_iq2_xxs_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32,             mul_mm_iq2_xs_f32,              has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32,            mul_mm_iq3_xxs_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32,              mul_mm_iq3_s_f32,               has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32,              mul_mm_iq2_s_f32,               has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32,              mul_mm_iq1_s_f32,               has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32,              mul_mm_iq1_m_f32,               has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32,             mul_mm_iq4_nl_f32,              has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32,             mul_mm_iq4_xs_f32,              has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F32,             mul_mm_id_f32_f32,              has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F32,             mul_mm_id_f16_f32,              has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_BF16_F32,            mul_mm_id_bf16_f32,             has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F32,            mul_mm_id_q4_0_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F32,            mul_mm_id_q4_1_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F32,            mul_mm_id_q5_0_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F32,            mul_mm_id_q5_1_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F32,            mul_mm_id_q8_0_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F32,            mul_mm_id_q2_K_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F32,            mul_mm_id_q3_K_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F32,            mul_mm_id_q4_K_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F32,            mul_mm_id_q5_K_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F32,            mul_mm_id_q6_K_f32,             has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F32,         mul_mm_id_iq2_xxs_f32,          has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F32,          mul_mm_id_iq2_xs_f32,           has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F32,         mul_mm_id_iq3_xxs_f32,          has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F32,           mul_mm_id_iq3_s_f32,            has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F32,           mul_mm_id_iq2_s_f32,            has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F32,           mul_mm_id_iq1_s_f32,            has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F32,           mul_mm_id_iq1_m_f32,            has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F32,          mul_mm_id_iq4_nl_f32,           has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F32,          mul_mm_id_iq4_xs_f32,           has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F32,                 rope_norm_f32,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F16,                 rope_norm_f16,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F32,                 rope_neox_f32,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F16,                 rope_neox_f16,                  true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_IM2COL_F16,                    im2col_f16,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_IM2COL_F32,                    im2col_f32,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F16,                im2col_ext_f16,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F32,                im2col_ext_f32,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F32_F32,     conv_transpose_1d_f32_f32,      true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F16_F32,     conv_transpose_1d_f16_f32,      true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_UPSCALE_F32,                   upscale_f32,                    true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_PAD_F32,                       pad_f32,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_PAD_REFLECT_1D_F32,            pad_reflect_1d_f32,             true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32,        timestep_embedding_f32,         true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ARANGE_F32,                    arange_f32,                     true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC,           argsort_f32_i32_asc,            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC,          argsort_f32_i32_desc,           true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32,                leaky_relu_f32,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H64,        flash_attn_ext_f16_h64,         has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H80,        flash_attn_ext_f16_h80,         has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H96,        flash_attn_ext_f16_h96,         has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H112,       flash_attn_ext_f16_h112,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H128,       flash_attn_ext_f16_h128,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H256,       flash_attn_ext_f16_h256,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H64,       flash_attn_ext_bf16_h64,        has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H80,       flash_attn_ext_bf16_h80,        has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H96,       flash_attn_ext_bf16_h96,        has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H112,      flash_attn_ext_bf16_h112,       has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H128,      flash_attn_ext_bf16_h128,       has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H256,      flash_attn_ext_bf16_h256,       has_simdgroup_mm && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H64,       flash_attn_ext_q4_0_h64,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H80,       flash_attn_ext_q4_0_h80,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H96,       flash_attn_ext_q4_0_h96,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H112,      flash_attn_ext_q4_0_h112,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H128,      flash_attn_ext_q4_0_h128,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H256,      flash_attn_ext_q4_0_h256,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H64,       flash_attn_ext_q4_1_h64,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H80,       flash_attn_ext_q4_1_h80,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H96,       flash_attn_ext_q4_1_h96,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H112,      flash_attn_ext_q4_1_h112,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H128,      flash_attn_ext_q4_1_h128,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H256,      flash_attn_ext_q4_1_h256,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H64,       flash_attn_ext_q5_0_h64,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H80,       flash_attn_ext_q5_0_h80,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H96,       flash_attn_ext_q5_0_h96,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H112,      flash_attn_ext_q5_0_h112,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H128,      flash_attn_ext_q5_0_h128,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H256,      flash_attn_ext_q5_0_h256,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H64,       flash_attn_ext_q5_1_h64,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H80,       flash_attn_ext_q5_1_h80,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H96,       flash_attn_ext_q5_1_h96,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H112,      flash_attn_ext_q5_1_h112,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H128,      flash_attn_ext_q5_1_h128,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H256,      flash_attn_ext_q5_1_h256,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H64,       flash_attn_ext_q8_0_h64,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H80,       flash_attn_ext_q8_0_h80,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H96,       flash_attn_ext_q8_0_h96,        has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H112,      flash_attn_ext_q8_0_h112,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H128,      flash_attn_ext_q8_0_h128,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H256,      flash_attn_ext_q8_0_h256,       has_simdgroup_mm);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H128,   flash_attn_ext_vec_f16_h128,    has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H128,  flash_attn_ext_vec_bf16_h128,   has_simdgroup_reduction && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H128,  flash_attn_ext_vec_q4_0_h128,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H128,  flash_attn_ext_vec_q4_1_h128,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H128,  flash_attn_ext_vec_q5_0_h128,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H128,  flash_attn_ext_vec_q5_1_h128,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H128,  flash_attn_ext_vec_q8_0_h128,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H256,   flash_attn_ext_vec_f16_h256,    has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H256,  flash_attn_ext_vec_bf16_h256,   has_simdgroup_reduction && use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H256,  flash_attn_ext_vec_q4_0_h256,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H256,  flash_attn_ext_vec_q4_1_h256,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H256,  flash_attn_ext_vec_q5_0_h256,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H256,  flash_attn_ext_vec_q5_1_h256,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H256,  flash_attn_ext_vec_q8_0_h256,   has_simdgroup_reduction);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SET_F32,                       set_f32,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SET_I32,                       set_i32,                        true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F32,                   cpy_f32_f32,                    true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F16,                   cpy_f32_f16,                    true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_BF16,                  cpy_f32_bf16,                   use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F32,                   cpy_f16_f32,                    true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F16,                   cpy_f16_f16,                    true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_F32,                  cpy_bf16_f32,                   use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_BF16,                 cpy_bf16_bf16,                  use_bfloat);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0,                  cpy_f32_q8_0,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0,                  cpy_f32_q4_0,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1,                  cpy_f32_q4_1,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0,                  cpy_f32_q5_0,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1,                  cpy_f32_q5_1,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL,                cpy_f32_iq4_nl,                 true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F32,                  cpy_q4_0_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F16,                  cpy_q4_0_f16,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F32,                  cpy_q4_1_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F16,                  cpy_q4_1_f16,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F32,                  cpy_q5_0_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F16,                  cpy_q5_0_f16,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F32,                  cpy_q5_1_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F16,                  cpy_q5_1_f16,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F32,                  cpy_q8_0_f32,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F16,                  cpy_q8_0_f16,                   true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_CONCAT,                        concat,                         true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SQR,                           sqr,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SQRT,                          sqrt,                           true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SIN,                           sin,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_COS,                           cos,                            true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_SUM_ROWS,                      sum_rows,                       true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_ARGMAX,                        argmax,                         true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_POOL_2D_AVG_F32,               pool_2d_avg_f32,                true);
        LM_GGML_METAL_ADD_KERNEL(LM_GGML_METAL_KERNEL_TYPE_POOL_2D_MAX_F32,               pool_2d_max_f32,                true);
    }

    [metal_library release];

    return ctx;
}

static void lm_ggml_metal_free(struct lm_ggml_backend_metal_context * ctx) {
    LM_GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < LM_GGML_METAL_KERNEL_TYPE_COUNT; ++i) {
        [ctx->kernels[i].pipeline release];
    }

    Block_release(ctx->encode_async);

    [ctx->queue release];

    dispatch_release(ctx->d_queue);

    free(ctx);
}

// temporarily defined here for compatibility between ggml-backend and the old API

struct lm_ggml_backend_metal_buffer {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct lm_ggml_backend_metal_buffer_context {
    void * all_data;
    size_t all_size;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct lm_ggml_backend_metal_buffer buffers[LM_GGML_METAL_MAX_BUFFERS];

    // optional MTLResidencySet
    id rset;
};

// rset init
static bool lm_ggml_backend_metal_buffer_rset_init(
        struct lm_ggml_backend_metal_buffer_context * ctx,
        struct lm_ggml_backend_metal_device_context * ctx_dev,
        id<MTLDevice> device) {
    ctx->rset = nil;

    if (!ctx_dev->has_residency_sets) {
        return true;
    }

#if defined(LM_GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"lm_ggml_backend_metal";
        desc.initialCapacity = ctx->n_buffers;

        NSError * error;
        ctx->rset = [device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            LM_GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < ctx->n_buffers; i++) {
            [ctx->rset addAllocation:ctx->buffers[i].metal];
        }

        [ctx->rset commit];
        [ctx->rset requestResidency];

        return true;
    }
#else
    LM_GGML_UNUSED(ctx_dev);
    LM_GGML_UNUSED(device);
#endif

    return true;
}

// rset free
static void lm_ggml_backend_metal_buffer_rset_free(struct lm_ggml_backend_metal_buffer_context * ctx) {
#if defined(LM_GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (ctx->rset) {
            [ctx->rset endResidency];
            [ctx->rset removeAllAllocations];
            [ctx->rset release];
        }
    }
#else
    LM_GGML_UNUSED(ctx);
#endif
}

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
static id<MTLBuffer> lm_ggml_metal_get_buffer(struct lm_ggml_tensor * t, size_t * offs) {
    //LM_GGML_LOG_INFO("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = lm_ggml_nbytes(t);

    lm_ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    struct lm_ggml_backend_metal_buffer_context * buf_ctx = (struct lm_ggml_backend_metal_buffer_context *) buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf_ctx->buffers[i].data;

        //LM_GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf_ctx->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf_ctx->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf_ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //LM_GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return buf_ctx->buffers[i].metal;
        }
    }

    LM_GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return nil;
}

static bool lm_ggml_metal_supports_op(const struct lm_ggml_backend_metal_device_context * ctx_dev, const struct lm_ggml_tensor * op) {
    const bool has_simdgroup_mm        = ctx_dev->has_simdgroup_mm;
    const bool has_simdgroup_reduction = ctx_dev->has_simdgroup_reduction;
    const bool use_bfloat              = ctx_dev->use_bfloat;

    if (!use_bfloat) {
        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == LM_GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case LM_GGML_OP_UNARY:
            switch (lm_ggml_get_unary_op(op)) {
                case LM_GGML_UNARY_OP_TANH:
                case LM_GGML_UNARY_OP_RELU:
                case LM_GGML_UNARY_OP_SIGMOID:
                case LM_GGML_UNARY_OP_GELU:
                case LM_GGML_UNARY_OP_GELU_QUICK:
                case LM_GGML_UNARY_OP_SILU:
                case LM_GGML_UNARY_OP_ELU:
                    return lm_ggml_is_contiguous(op->src[0]) && op->src[0]->type == LM_GGML_TYPE_F32;
                default:
                    return false;
            }
        case LM_GGML_OP_NONE:
        case LM_GGML_OP_RESHAPE:
        case LM_GGML_OP_VIEW:
        case LM_GGML_OP_TRANSPOSE:
        case LM_GGML_OP_PERMUTE:
        case LM_GGML_OP_CONCAT:
            return true;
        case LM_GGML_OP_ADD:
        case LM_GGML_OP_SUB:
        case LM_GGML_OP_MUL:
        case LM_GGML_OP_DIV:
            return op->src[0]->type == LM_GGML_TYPE_F32;
        case LM_GGML_OP_ACC:
        case LM_GGML_OP_REPEAT:
        case LM_GGML_OP_SCALE:
        case LM_GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case LM_GGML_OP_CLAMP:
            return op->src[0]->type == LM_GGML_TYPE_F32;
        case LM_GGML_OP_SQR:
        case LM_GGML_OP_SQRT:
        case LM_GGML_OP_SIN:
        case LM_GGML_OP_COS:
            return lm_ggml_is_contiguous(op->src[0]) && op->src[0]->type == LM_GGML_TYPE_F32;
        case LM_GGML_OP_LOG:
            return false; // TODO: implement
        case LM_GGML_OP_SUM_ROWS:
        case LM_GGML_OP_SOFT_MAX:
        case LM_GGML_OP_GROUP_NORM:
            return has_simdgroup_reduction && lm_ggml_is_contiguous(op->src[0]);
        case LM_GGML_OP_RMS_NORM:
            return has_simdgroup_reduction && (op->ne[0] % 4 == 0 && lm_ggml_is_contiguous_1(op->src[0]));
        case LM_GGML_OP_ARGMAX:
            return true;
        case LM_GGML_OP_NORM:
            return has_simdgroup_reduction && (op->ne[0] % 4 == 0 && lm_ggml_is_contiguous_1(op->src[0]));
        case LM_GGML_OP_ROPE:
            {
                const int mode = ((const int32_t *) op->op_params)[2];
                if (mode & LM_GGML_ROPE_TYPE_MROPE) {
                    return false;
                }
                if (mode & LM_GGML_ROPE_TYPE_VISION) {
                    return false;
                }
                return true;
            }
        case LM_GGML_OP_IM2COL:
            return op->src[0]->type == LM_GGML_TYPE_F16;
        case LM_GGML_OP_POOL_1D:
            return false;
        case LM_GGML_OP_POOL_2D:
        case LM_GGML_OP_UPSCALE:
        case LM_GGML_OP_PAD:
        case LM_GGML_OP_PAD_REFLECT_1D:
        case LM_GGML_OP_TIMESTEP_EMBEDDING:
        case LM_GGML_OP_ARGSORT:
        case LM_GGML_OP_LEAKY_RELU:
            return op->src[0]->type == LM_GGML_TYPE_F32;
        case LM_GGML_OP_ARANGE:
            return true;
        case LM_GGML_OP_FLASH_ATTN_EXT:
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case LM_GGML_OP_SSM_CONV:
        case LM_GGML_OP_SSM_SCAN:
            return true;
        case LM_GGML_OP_MUL_MAT:
        case LM_GGML_OP_MUL_MAT_ID:
            return has_simdgroup_reduction &&
                (op->src[0]->type != LM_GGML_TYPE_F32 || op->src[1]->type == LM_GGML_TYPE_F32);
        case LM_GGML_OP_CPY:
        case LM_GGML_OP_DUP:
        case LM_GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case LM_GGML_TYPE_F32:
                        switch (op->type) {
                           case LM_GGML_TYPE_F32:
                           case LM_GGML_TYPE_F16:
                           case LM_GGML_TYPE_BF16:
                           case LM_GGML_TYPE_Q8_0:
                           case LM_GGML_TYPE_Q4_0:
                           case LM_GGML_TYPE_Q4_1:
                           case LM_GGML_TYPE_Q5_0:
                           case LM_GGML_TYPE_Q5_1:
                           case LM_GGML_TYPE_IQ4_NL:
                                return true;
                           default:
                                return false;
                        }
                    case LM_GGML_TYPE_F16:
                        switch (op->type) {
                            case LM_GGML_TYPE_F32:
                            case LM_GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case LM_GGML_TYPE_BF16:
                        switch (op->type) {
                            case LM_GGML_TYPE_F32:
                            case LM_GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case LM_GGML_TYPE_Q4_0:
                    case LM_GGML_TYPE_Q4_1:
                    case LM_GGML_TYPE_Q5_0:
                    case LM_GGML_TYPE_Q5_1:
                    case LM_GGML_TYPE_Q8_0:
                        switch (op->type) {
                            case LM_GGML_TYPE_F32:
                            case LM_GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    default:
                        return false;
                };
            }
        case LM_GGML_OP_SET:
            {
                switch (op->src[0]->type) {
                    case LM_GGML_TYPE_F32:
                    case LM_GGML_TYPE_I32:
                        return true;
                    default:
                        return false;
                };
            }
        case LM_GGML_OP_DIAG_MASK_INF:
        case LM_GGML_OP_GET_ROWS:
            {
                return op->ne[3] == 1;
            }
        default:
            return false;
    }
}

static void lm_ggml_metal_encode_node(
                        lm_ggml_backend_t   backend,
                                   int   idx,
          id<MTLComputeCommandEncoder>   encoder) {
    struct lm_ggml_backend_metal_context        * ctx     = backend->context;
    struct lm_ggml_backend_metal_device_context * ctx_dev = backend->device->context;

    struct lm_ggml_cgraph * gf = ctx->gf;

    struct lm_ggml_tensor * node = lm_ggml_graph_node(gf, idx);

    //LM_GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, lm_ggml_op_name(node->op));

    struct lm_ggml_tensor * src0 = node->src[0];
    struct lm_ggml_tensor * src1 = node->src[1];
    struct lm_ggml_tensor * src2 = node->src[2];
    struct lm_ggml_tensor * dst  = node;

    if (lm_ggml_is_empty(dst)) {
        return;
    }

    switch (dst->op) {
        case LM_GGML_OP_NONE:
        case LM_GGML_OP_RESHAPE:
        case LM_GGML_OP_VIEW:
        case LM_GGML_OP_TRANSPOSE:
        case LM_GGML_OP_PERMUTE:
            {
                // noop -> next node
            } return;
        default:
            {
            } break;
    }

    if (!lm_ggml_metal_supports_op(ctx_dev, dst)) {
        LM_GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, lm_ggml_op_desc(dst));
        LM_GGML_ABORT("unsupported op");
    }

    const int64_t  ne00 = src0 ? src0->ne[0] : 0;
    const int64_t  ne01 = src0 ? src0->ne[1] : 0;
    const int64_t  ne02 = src0 ? src0->ne[2] : 0;
    const int64_t  ne03 = src0 ? src0->ne[3] : 0;

    const uint64_t nb00 = src0 ? src0->nb[0] : 0;
    const uint64_t nb01 = src0 ? src0->nb[1] : 0;
    const uint64_t nb02 = src0 ? src0->nb[2] : 0;
    const uint64_t nb03 = src0 ? src0->nb[3] : 0;

    const int64_t  ne10 = src1 ? src1->ne[0] : 0;
    const int64_t  ne11 = src1 ? src1->ne[1] : 0;
    const int64_t  ne12 = src1 ? src1->ne[2] : 0;
    const int64_t  ne13 = src1 ? src1->ne[3] : 0;

    const uint64_t nb10 = src1 ? src1->nb[0] : 0;
    const uint64_t nb11 = src1 ? src1->nb[1] : 0;
    const uint64_t nb12 = src1 ? src1->nb[2] : 0;
    const uint64_t nb13 = src1 ? src1->nb[3] : 0;

    const int64_t  ne20 = src2 ? src2->ne[0] : 0;
    const int64_t  ne21 = src2 ? src2->ne[1] : 0;
    const int64_t  ne22 = src2 ? src2->ne[2] : 0; LM_GGML_UNUSED(ne22);
    const int64_t  ne23 = src2 ? src2->ne[3] : 0; LM_GGML_UNUSED(ne23);

    const uint64_t nb20 = src2 ? src2->nb[0] : 0; LM_GGML_UNUSED(nb20);
    const uint64_t nb21 = src2 ? src2->nb[1] : 0;
    const uint64_t nb22 = src2 ? src2->nb[2] : 0;
    const uint64_t nb23 = src2 ? src2->nb[3] : 0; LM_GGML_UNUSED(nb23);

    const int64_t  ne0  =  dst ?  dst->ne[0] : 0;
    const int64_t  ne1  =  dst ?  dst->ne[1] : 0;
    const int64_t  ne2  =  dst ?  dst->ne[2] : 0;
    const int64_t  ne3  =  dst ?  dst->ne[3] : 0;

    const uint64_t nb0  =  dst ?  dst->nb[0] : 0;
    const uint64_t nb1  =  dst ?  dst->nb[1] : 0;
    const uint64_t nb2  =  dst ?  dst->nb[2] : 0;
    const uint64_t nb3  =  dst ?  dst->nb[3] : 0;

    const enum lm_ggml_type src0t = src0 ? src0->type : LM_GGML_TYPE_COUNT;
    const enum lm_ggml_type src1t = src1 ? src1->type : LM_GGML_TYPE_COUNT;
    const enum lm_ggml_type dstt  = dst  ? dst->type  : LM_GGML_TYPE_COUNT;

    size_t offs_src0 = 0;
    size_t offs_src1 = 0;
    size_t offs_src2 = 0;
    size_t offs_dst  = 0;

    id<MTLBuffer> id_src0 = src0 ? lm_ggml_metal_get_buffer(src0, &offs_src0) : nil;
    id<MTLBuffer> id_src1 = src1 ? lm_ggml_metal_get_buffer(src1, &offs_src1) : nil;
    id<MTLBuffer> id_src2 = src2 ? lm_ggml_metal_get_buffer(src2, &offs_src2) : nil;
    id<MTLBuffer> id_dst  = dst  ? lm_ggml_metal_get_buffer(dst,  &offs_dst)  : nil;

#if 0
    LM_GGML_LOG_INFO("%s: op - %s\n", __func__, lm_ggml_op_name(dst->op));
    if (src0) {
        LM_GGML_LOG_INFO("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, lm_ggml_type_name(src0t), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                lm_ggml_is_contiguous(src0), src0->name);
    }
    if (src1) {
        LM_GGML_LOG_INFO("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, lm_ggml_type_name(src1t), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                lm_ggml_is_contiguous(src1), src1->name);
    }
    if (dst) {
        LM_GGML_LOG_INFO("%s: dst  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, lm_ggml_type_name(dstt), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                dst->name);
    }
#endif

    id<MTLDevice> device = ctx_dev->mtl_device;

    switch (dst->op) {
        case LM_GGML_OP_CONCAT:
            {
                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CONCAT].pipeline;

                const int32_t dim = ((const int32_t *) dst->op_params)[0];

                lm_ggml_metal_kargs_concat args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.ne13 =*/ ne13,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                    /*.dim  =*/ dim,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_ADD:
        case LM_GGML_OP_SUB:
        case LM_GGML_OP_MUL:
        case LM_GGML_OP_DIV:
            {
                LM_GGML_ASSERT(src0t == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);

                const size_t offs = 0;

                bool bcast_row = false;

                id<MTLComputePipelineState> pipeline = nil;

                if (lm_ggml_nelements(src1) == ne10 && lm_ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
                    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                    // src1 is a row
                    LM_GGML_ASSERT(ne11 == 1);

                    switch (dst->op) {
                        case LM_GGML_OP_ADD: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ADD_ROW].pipeline; break;
                        case LM_GGML_OP_SUB: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SUB_ROW].pipeline; break;
                        case LM_GGML_OP_MUL: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_ROW].pipeline; break;
                        case LM_GGML_OP_DIV: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_DIV_ROW].pipeline; break;
                        default: LM_GGML_ABORT("fatal error");
                    }

                    bcast_row = true;
                } else {
                    switch (dst->op) {
                        case LM_GGML_OP_ADD: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ADD].pipeline; break;
                        case LM_GGML_OP_SUB: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SUB].pipeline; break;
                        case LM_GGML_OP_MUL: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL].pipeline; break;
                        case LM_GGML_OP_DIV: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_DIV].pipeline; break;
                        default: LM_GGML_ABORT("fatal error");
                    }
                }

                lm_ggml_metal_kargs_bin args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.ne13 =*/ ne13,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                    /*.offs =*/ offs,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                if (bcast_row) {
                    const int64_t n = lm_ggml_nelements(dst)/4;

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } else {
                    const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                }
            } break;
        case LM_GGML_OP_REPEAT:
            {
                id<MTLComputePipelineState> pipeline;

                switch (src0t) {
                    case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_REPEAT_F32].pipeline; break;
                    case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_REPEAT_F16].pipeline; break;
                    case LM_GGML_TYPE_I32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_REPEAT_I32].pipeline; break;
                    case LM_GGML_TYPE_I16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_REPEAT_I16].pipeline; break;
                    default: LM_GGML_ABORT("fatal error");
                }

                lm_ggml_metal_kargs_repeat args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_ACC:
            {
                LM_GGML_ASSERT(src0t == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(dstt  == LM_GGML_TYPE_F32);

                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));

                const size_t pnb1 = ((const int32_t *) dst->op_params)[0];
                const size_t pnb2 = ((const int32_t *) dst->op_params)[1];
                const size_t pnb3 = ((const int32_t *) dst->op_params)[2];
                const size_t offs = ((const int32_t *) dst->op_params)[3];

                const bool inplace = (bool) ((const int32_t *) dst->op_params)[4];

                if (!inplace) {
                    // run a separete kernel to cpy src->dst
                    // not sure how to avoid this
                    // TODO: make a simpler cpy_bytes kernel

                    const id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F32].pipeline;

                    lm_ggml_metal_kargs_cpy args = {
                        /*.ne00 =*/ ne00,
                        /*.ne01 =*/ ne01,
                        /*.ne02 =*/ ne02,
                        /*.ne03 =*/ ne03,
                        /*.nb00 =*/ nb00,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.nb03 =*/ nb03,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.ne2  =*/ ne2,
                        /*.ne3  =*/ ne3,
                        /*.nb0  =*/ nb0,
                        /*.nb1  =*/ nb1,
                        /*.nb2  =*/ nb2,
                        /*.nb3  =*/ nb3,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                    const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00);

                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                }

                const id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ADD].pipeline;

                lm_ggml_metal_kargs_bin args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ pnb1,
                    /*.nb02 =*/ pnb2,
                    /*.nb03 =*/ pnb3,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.ne13 =*/ ne13,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ pnb1,
                    /*.nb2  =*/ pnb2,
                    /*.nb3  =*/ pnb3,
                    /*.offs =*/ offs,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00);

                [encoder dispatchThreadgroups:MTLSizeMake(ne11, ne12, ne13) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_SCALE:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                float scale;
                memcpy(&scale, dst->op_params, sizeof(scale));

                int64_t n = lm_ggml_nelements(dst);

                id<MTLComputePipelineState> pipeline = nil;

                if (n % 4 == 0) {
                    n /= 4;
                    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SCALE_4].pipeline;
                } else {
                    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SCALE].pipeline;
                }

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0   offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst    offset:offs_dst  atIndex:1];
                [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_CLAMP:
            {
                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CLAMP].pipeline;

                float min;
                float max;
                memcpy(&min, ((const int32_t *) dst->op_params) + 0, sizeof(float));
                memcpy(&max, ((const int32_t *) dst->op_params) + 1, sizeof(float));

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&min   length:sizeof(min) atIndex:2];
                [encoder setBytes:&max   length:sizeof(max) atIndex:3];

                const int64_t n = lm_ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_UNARY:
            switch (lm_ggml_get_unary_op(node)) {
                // we are not taking into account the strides, so for now require contiguous tensors
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                case LM_GGML_UNARY_OP_TANH:
                {
                    id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_TANH].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = lm_ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case LM_GGML_UNARY_OP_RELU:
                {
                    id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_RELU].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = lm_ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case LM_GGML_UNARY_OP_SIGMOID:
                {
                    id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SIGMOID].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = lm_ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case LM_GGML_UNARY_OP_GELU:
                {
                    int64_t n = lm_ggml_nelements(dst);

                    id<MTLComputePipelineState> pipeline = nil;

                    if (n % 4 == 0) {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GELU_4].pipeline;
                        n /= 4;
                    } else {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GELU].pipeline;
                    }

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case LM_GGML_UNARY_OP_GELU_QUICK:
                {
                    int64_t n = lm_ggml_nelements(dst);

                    id<MTLComputePipelineState> pipeline = nil;

                    if (n % 4 == 0) {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK_4].pipeline;
                        n /= 4;
                    } else {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GELU_QUICK].pipeline;
                    }

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case LM_GGML_UNARY_OP_SILU:
                {
                    int64_t n = lm_ggml_nelements(dst);

                    id<MTLComputePipelineState> pipeline = nil;

                    if (n % 4 == 0) {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SILU_4].pipeline;
                        n /= 4;
                    } else {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SILU].pipeline;
                    }

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case LM_GGML_UNARY_OP_ELU:
                {
                    id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ELU].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = lm_ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                default:
                {
                    LM_GGML_LOG_WARN("%s: node %3d, op = %8s not implemented\n", __func__, idx, lm_ggml_op_name(dst->op));
                    LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_OP_SQR:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SQR].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = lm_ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_SQRT:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SQRT].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = lm_ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_SIN:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SIN].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = lm_ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_COS:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_COS].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = lm_ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_SUM_ROWS:
            {
                LM_GGML_ASSERT(src0->nb[0] == lm_ggml_type_size(src0->type));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SUM_ROWS].pipeline;


                lm_ggml_metal_kargs_sum_rows args = {
                   /*.ne00 =*/ ne00,
                   /*.ne01 =*/ ne01,
                   /*.ne02 =*/ ne02,
                   /*.ne03 =*/ ne03,
                   /*.nb00 =*/ nb00,
                   /*.nb01 =*/ nb01,
                   /*.nb02 =*/ nb02,
                   /*.nb03 =*/ nb03,
                   /*.ne10 =*/ ne10,
                   /*.ne11 =*/ ne11,
                   /*.ne12 =*/ ne12,
                   /*.ne13 =*/ ne13,
                   /*.nb10 =*/ nb10,
                   /*.nb11 =*/ nb11,
                   /*.nb12 =*/ nb12,
                   /*.nb13 =*/ nb13,
                   /*.ne0  =*/ ne0,
                   /*.ne1  =*/ ne1,
                   /*.ne2  =*/ ne2,
                   /*.ne3  =*/ ne3,
                   /*.nb0  =*/ nb0,
                   /*.nb1  =*/ nb1,
                   /*.nb2  =*/ nb2,
                   /*.nb3  =*/ nb3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_SOFT_MAX:
            {
                LM_GGML_ASSERT(!src1 || src1->type == LM_GGML_TYPE_F16 || src1->type == LM_GGML_TYPE_F32);

                int nth = 32; // SIMD width

                id<MTLComputePipelineState> pipeline = nil;

                const bool use_f16 = (src1 && src1->type == LM_GGML_TYPE_F16);

                if (ne00%4 == 0) {
                    while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
                        nth *= 2;
                    }
                    if (use_f16) {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16_4].pipeline;
                    } else {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32_4].pipeline;
                    }
                } else {
                    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
                        nth *= 2;
                    }
                    if (use_f16) {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16].pipeline;
                    } else {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32].pipeline;
                    }
                }

                float scale;
                float max_bias;

                memcpy(&scale,    ((const int32_t *) dst->op_params) + 0, sizeof(scale));
                memcpy(&max_bias, ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));

                const int64_t nrows_x = lm_ggml_nrows(src0);
                const int64_t nrows_y = src0->ne[1];

                const uint32_t n_head      = nrows_x/nrows_y;
                const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

                const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
                const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

                lm_ggml_metal_kargs_soft_max args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.scale =*/ scale,
                    /*.max_bias =*/ max_bias,
                    /*.m0 =*/ m0,
                    /*.m1 =*/ m1,
                    /*.n_head_log2 =*/ n_head_log2,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0   atIndex:0];
                if (id_src1) {
                    [encoder setBuffer:id_src1 offset:offs_src1   atIndex:1];
                } else {
                    [encoder setBuffer:id_src0 offset:offs_src0   atIndex:1];
                }
                [encoder setBuffer:id_dst      offset:offs_dst            atIndex:2];
                [encoder setBytes:&args        length:sizeof(args)        atIndex:3];

                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01*ne02*ne03, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_DIAG_MASK_INF:
            {
                const int n_past = ((const int32_t *)(dst->op_params))[0];

                id<MTLComputePipelineState> pipeline = nil;

                if (ne00%8 == 0) {
                    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8].pipeline;
                } else {
                    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF].pipeline;
                }

                lm_ggml_metal_kargs_diag_mask_inf args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.n_past =*/ n_past,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args  length:sizeof(args) atIndex:2];

                if (ne00%8 == 0) {
                    [encoder dispatchThreadgroups:MTLSizeMake(ne00*ne01*ne02/8, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                }
                else {
                    [encoder dispatchThreadgroups:MTLSizeMake(ne00, ne01, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                }
            } break;
        case LM_GGML_OP_SSM_CONV:
            {
                LM_GGML_ASSERT(src0t == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);

                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SSM_CONV_F32].pipeline;

                lm_ggml_metal_kargs_ssm_conv args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0    atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1    atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst     atIndex:2];
                [encoder setBytes:&args    length:sizeof(args) atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne1, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_SSM_SCAN:
            {
                struct lm_ggml_tensor * src3 = node->src[3];
                struct lm_ggml_tensor * src4 = node->src[4];
                struct lm_ggml_tensor * src5 = node->src[5];

                LM_GGML_ASSERT(src3);
                LM_GGML_ASSERT(src4);
                LM_GGML_ASSERT(src5);

                size_t offs_src3 = 0;
                size_t offs_src4 = 0;
                size_t offs_src5 = 0;

                id<MTLBuffer> id_src3 = src3 ? lm_ggml_metal_get_buffer(src3, &offs_src3) : nil;
                id<MTLBuffer> id_src4 = src4 ? lm_ggml_metal_get_buffer(src4, &offs_src4) : nil;
                id<MTLBuffer> id_src5 = src5 ? lm_ggml_metal_get_buffer(src5, &offs_src5) : nil;

                const int64_t  ne30 = src3->ne[0]; LM_GGML_UNUSED(ne30);
                const int64_t  ne31 = src3->ne[1]; LM_GGML_UNUSED(ne31);

                const uint64_t nb30 = src3->nb[0];
                const uint64_t nb31 = src3->nb[1];

                const int64_t  ne40 = src4->ne[0]; LM_GGML_UNUSED(ne40);
                const int64_t  ne41 = src4->ne[1]; LM_GGML_UNUSED(ne41);
                const int64_t  ne42 = src4->ne[2]; LM_GGML_UNUSED(ne42);

                const uint64_t nb40 = src4->nb[0];
                const uint64_t nb41 = src4->nb[1];
                const uint64_t nb42 = src4->nb[2];

                const int64_t  ne50 = src5->ne[0]; LM_GGML_UNUSED(ne50);
                const int64_t  ne51 = src5->ne[1]; LM_GGML_UNUSED(ne51);
                const int64_t  ne52 = src5->ne[2]; LM_GGML_UNUSED(ne52);

                const uint64_t nb50 = src5->nb[0];
                const uint64_t nb51 = src5->nb[1];
                const uint64_t nb52 = src5->nb[2];

                const int64_t d_state      = ne00;
                const int64_t d_inner      = ne01;
                const int64_t n_seq_tokens = ne11;
                const int64_t n_seqs       = ne02;

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SSM_SCAN_F32].pipeline;

                lm_ggml_metal_kargs_ssm_scan args = {
                    /*.d_state =*/ d_state,
                    /*.d_inner =*/ d_inner,
                    /*.n_seq_tokens =*/ n_seq_tokens,
                    /*.n_seqs =*/ n_seqs,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.nb20 =*/ nb20,
                    /*.nb21 =*/ nb21,
                    /*.nb22 =*/ nb22,
                    /*.nb30 =*/ nb30,
                    /*.nb31 =*/ nb31,
                    /*.nb40 =*/ nb40,
                    /*.nb41 =*/ nb41,
                    /*.nb42 =*/ nb42,
                    /*.nb50 =*/ nb50,
                    /*.nb51 =*/ nb51,
                    /*.nb52 =*/ nb52,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                [encoder setBuffer:id_src2 offset:offs_src2 atIndex:2];
                [encoder setBuffer:id_src3 offset:offs_src3 atIndex:3];
                [encoder setBuffer:id_src4 offset:offs_src4 atIndex:4];
                [encoder setBuffer:id_src5 offset:offs_src5 atIndex:5];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:6];
                [encoder setBytes:&args    length:sizeof(args) atIndex:7];

                [encoder dispatchThreadgroups:MTLSizeMake(d_inner, n_seqs, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_MUL_MAT:
            {
                LM_GGML_ASSERT(ne00 == ne10);

                LM_GGML_ASSERT(ne12 % ne02 == 0);
                LM_GGML_ASSERT(ne13 % ne03 == 0);

                const uint32_t r2 = ne12/ne02;
                const uint32_t r3 = ne13/ne03;

                // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                // to the matrix-vector kernel
                const int ne11_mm_min = 4;

                // first try to use small-batch mat-mv kernels
                // these should be efficient for BS [2, ~8]
                if (src1t == LM_GGML_TYPE_F32 && (ne00%256 == 0) &&
                    (
                     (
                      (
                       src0t == LM_GGML_TYPE_F16  || // TODO: helper function
                       src0t == LM_GGML_TYPE_Q4_0 ||
                       src0t == LM_GGML_TYPE_Q4_1 ||
                       src0t == LM_GGML_TYPE_Q5_0 ||
                       src0t == LM_GGML_TYPE_Q5_1 ||
                       src0t == LM_GGML_TYPE_Q8_0 ||
                       src0t == LM_GGML_TYPE_IQ4_NL ||
                       false) && (ne11 >= 2 && ne11 <= 8)
                     ) ||
                     (
                      (
                       src0t == LM_GGML_TYPE_Q4_K ||
                       src0t == LM_GGML_TYPE_Q5_K ||
                       src0t == LM_GGML_TYPE_Q6_K ||
                       false) && (ne11 >= 4 && ne11 <= 8)
                     )
                    )
                   ) {
                    // TODO: determine the optimal parameters based on grid utilization
                    //       I still don't know why we should not always use the maximum available threads:
                    //
                    //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
                    //
                    //       my current hypothesis is that the work grid is not evenly divisible for different nsg
                    //       values and there can be some tail effects when nsg is high. need to confirm this
                    //
                    const int nsg    = 2;                 // num simdgroups per threadgroup
                    const int nxpsg  = ne11 < 3 ? 16 : 8; // num threads along row per simdgroup
                    const int nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
                    const int r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
                          int r1ptg  = 4;                 // num src1 rows per threadgroup

                    // note: not sure how optimal are those across all different hardware. there might be someting cleverer
                    switch (ne11) {
                        case 2:
                            r1ptg = 2; break;
                        case 3:
                        case 6:
                            r1ptg = 3; break;
                        case 4:
                        case 7:
                        case 8:
                            r1ptg = 4; break;
                        case 5:
                            r1ptg = 5; break;
                    };

                    id<MTLComputePipelineState> pipeline = nil;

                    switch (src0->type) {
                        case LM_GGML_TYPE_F16:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q4_0:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q4_1:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q5_0:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q5_1:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q8_0:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q4_K:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q5_K:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_Q6_K:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        case LM_GGML_TYPE_IQ4_NL:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_5].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            } break;
                        default: LM_GGML_ABORT("not implemented");
                    }

                    lm_ggml_metal_kargs_mul_mv_ext args = {
                        /*.ne00  =*/ ne00,
                        /*.ne01  =*/ ne01,
                        /*.ne02  =*/ ne02,
                        /*.nb00  =*/ nb00,
                        /*.nb01  =*/ nb01,
                        /*.nb02  =*/ nb02,
                        /*.nb03  =*/ nb03,
                        /*.ne10  =*/ ne10,
                        /*.ne11  =*/ ne11,
                        /*.ne12  =*/ ne12,
                        /*.nb10  =*/ nb10,
                        /*.nb11  =*/ nb11,
                        /*.nb12  =*/ nb12,
                        /*.nb13  =*/ nb13,
                        /*.ne0   =*/ ne0,
                        /*.ne1   =*/ ne1,
                        /*.r2    =*/ r2,
                        /*.r3    =*/ r3,
                        /*.nsg   =*/ nsg,
                        /*.nxpsg =*/ nxpsg,
                        /*.r1ptg =*/ r1ptg,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                    //printf("ne01 = %lld nr0ptg = %d\n", ne01, nr0ptg);
                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + r0ptg - 1)/r0ptg, (ne11 + r1ptg - 1)/r1ptg, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                } else
                // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                if ([device supportsFamily:MTLGPUFamilyApple7] &&
                        !lm_ggml_is_transposed(src0) &&
                        !lm_ggml_is_transposed(src1) &&
                        src1t == LM_GGML_TYPE_F32 &&
                        ne00 % 32 == 0 && ne00 >= 64 &&
                        (ne11 > ne11_mm_min || (lm_ggml_is_quantized(src0t) && ne12 > 1))) {
                    //printf("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

                    // some Metal matrix data types require aligned pointers
                    // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
                    switch (src0->type) {
                        case LM_GGML_TYPE_F32:  LM_GGML_ASSERT(nb01 % 16 == 0); break;
                        case LM_GGML_TYPE_F16:  LM_GGML_ASSERT(nb01 % 8  == 0); break;
                        case LM_GGML_TYPE_BF16: LM_GGML_ASSERT(nb01 % 8  == 0); break;
                        default: break;
                    }

                    id<MTLComputePipelineState> pipeline = nil;

                    switch (src0->type) {
                        case LM_GGML_TYPE_F32:     pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32    ].pipeline; break;
                        case LM_GGML_TYPE_F16:     pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32    ].pipeline; break;
                        case LM_GGML_TYPE_BF16:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_BF16_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q4_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q4_1:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q5_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q5_1:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q8_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q2_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q3_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q4_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q5_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q6_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32].pipeline; break;
                        case LM_GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32 ].pipeline; break;
                        case LM_GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32].pipeline; break;
                        case LM_GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ1_M:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32 ].pipeline; break;
                        case LM_GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32 ].pipeline; break;
                        default: LM_GGML_ABORT("MUL MAT-MAT not implemented");
                    }

                    lm_ggml_metal_kargs_mul_mm args = {
                        /*.ne00 =*/ ne00,
                        /*.ne02 =*/ ne02,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.nb03 =*/ nb03,
                        /*.ne12 =*/ ne12,
                        /*.nb10 =*/ nb10,
                        /*.nb11 =*/ nb11,
                        /*.nb12 =*/ nb12,
                        /*.nb13 =*/ nb13,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.r2   =*/ r2,
                        /*.r3   =*/ r3,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0    atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1    atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst     atIndex:3];

                    [encoder setThreadgroupMemoryLength:8192 atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake( (ne11 + 31)/32, (ne01 + 63)/64, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    int nth0 = 32;
                    int nth1 = 1;
                    int nrows = 1;
                    //printf("vector: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

                    id<MTLComputePipelineState> pipeline = nil;

                    // use custom matrix x vector kernel
                    switch (src0t) {
                        case LM_GGML_TYPE_F32:
                            {
                                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32].pipeline;
                                nrows = 4;
                            } break;
                        case LM_GGML_TYPE_F16:
                            {
                                nth0 = 32;
                                nth1 = 1;
                                if (src1t == LM_GGML_TYPE_F32) {
                                    if (ne11 * ne12 < 4) {
                                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW].pipeline;
                                    } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4].pipeline;
                                        nrows = ne11;
                                    } else {
                                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32].pipeline;
                                        nrows = 4;
                                    }
                                } else {
                                    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16].pipeline;
                                    nrows = 4;
                                }
                            } break;
                        case LM_GGML_TYPE_BF16:
                            {
                                nth0 = 32;
                                nth1 = 1;
                                if (src1t == LM_GGML_TYPE_F32) {
                                    if (ne11 * ne12 < 4) {
                                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_1ROW].pipeline;
                                    } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_L4].pipeline;
                                        nrows = ne11;
                                    } else {
                                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32].pipeline;
                                        nrows = 4;
                                    }
                                } else {
                                    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_BF16].pipeline;
                                    nrows = 4;
                                }
                            } break;
                        case LM_GGML_TYPE_Q4_0:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q4_1:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q5_0:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q5_1:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q8_0:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q2_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q3_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q4_K:
                            {
                                nth0 = 4; //1;
                                nth1 = 8; //32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q5_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q6_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ2_XXS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ2_XS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ3_XXS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ3_S:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ2_S:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ1_S:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ1_M:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ4_NL:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ4_XS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32].pipeline;
                            } break;
                        default:
                            {
                                LM_GGML_LOG_ERROR("Asserting on type %d\n", (int)src0t);
                                LM_GGML_ABORT("not implemented");
                            }
                    };

                    lm_ggml_metal_kargs_mul_mv args = {
                        /*.ne00 =*/ ne00,
                        /*.ne01 =*/ ne01,
                        /*.ne02 =*/ ne02,
                        /*.nb00 =*/ nb00,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.nb03 =*/ nb03,
                        /*.ne10 =*/ ne10,
                        /*.ne11 =*/ ne11,
                        /*.ne12 =*/ ne12,
                        /*.nb10 =*/ nb10,
                        /*.nb11 =*/ nb11,
                        /*.nb12 =*/ nb12,
                        /*.nb13 =*/ nb13,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.r2   =*/ r2,
                        /*.r3   =*/ r3,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                    if (src0t == LM_GGML_TYPE_Q4_0  || src0t == LM_GGML_TYPE_Q4_1  || src0t == LM_GGML_TYPE_Q5_0 ||
                        src0t == LM_GGML_TYPE_Q5_1  || src0t == LM_GGML_TYPE_Q8_0  || src0t == LM_GGML_TYPE_Q2_K ||
                        src0t == LM_GGML_TYPE_IQ1_S || src0t == LM_GGML_TYPE_IQ1_M || src0t == LM_GGML_TYPE_IQ2_S) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_IQ2_XXS || src0t == LM_GGML_TYPE_IQ2_XS) {
                        const int mem_size = src0t == LM_GGML_TYPE_IQ2_XXS ? 256*8+128 : 512*8+128;
                        [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_IQ3_XXS || src0t == LM_GGML_TYPE_IQ3_S) {
                        const int mem_size = src0t == LM_GGML_TYPE_IQ3_XXS ? 256*4+128 : 512*4;
                        [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_IQ4_NL || src0t == LM_GGML_TYPE_IQ4_XS) {
                        const int mem_size = 32*sizeof(float);
                        [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q4_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q3_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q5_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q6_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    } else {
                        const int64_t ny = (ne11 + nrows - 1)/nrows;
                        [encoder dispatchThreadgroups:MTLSizeMake(ne01, ny, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                }
            } break;
        case LM_GGML_OP_MUL_MAT_ID:
            {
                const int n_as = src0->ne[2];

                // src2 = ids
                const enum lm_ggml_type src2t = src2->type; LM_GGML_UNUSED(src2t);

                LM_GGML_ASSERT(src2t == LM_GGML_TYPE_I32);

                LM_GGML_ASSERT(!lm_ggml_is_transposed(src0));
                LM_GGML_ASSERT(!lm_ggml_is_transposed(src1));

                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);

                LM_GGML_ASSERT(ne03 == 1);
                LM_GGML_ASSERT(ne13 == 1);

                // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                // to the matrix-vector kernel
                // ne20 = n_used_experts
                // ne21 = n_rows
                const int dst_rows = ne20*ne21;
                const int dst_rows_min = n_as;
                const int dst_rows_max = (device.maxThreadgroupMemoryLength - 32 - 8192)/4;

                // max size of the rowids array in the kernel shared buffer
                LM_GGML_ASSERT(dst_rows <= dst_rows_max);

                // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                // !!!
                // TODO: for now, always use mat-vec kernels until we figure out how to improve the
                //       indirect matrix multiplication
                // !!!
                if ([device supportsFamily:MTLGPUFamilyApple7] &&
                        ne00 % 32 == 0 && ne00 >= 64 &&
                        dst_rows > dst_rows_min) {
                    // some Metal matrix data types require aligned pointers
                    // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
                    switch (src0->type) {
                        case LM_GGML_TYPE_F32:  LM_GGML_ASSERT(nb01 % 16 == 0); break;
                        case LM_GGML_TYPE_F16:  LM_GGML_ASSERT(nb01 % 8  == 0); break;
                        case LM_GGML_TYPE_BF16: LM_GGML_ASSERT(nb01 % 8  == 0); break;
                        default: break;
                    }

                    id<MTLComputePipelineState> pipeline = nil;

                    switch (src0->type) {
                        case LM_GGML_TYPE_F32:     pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F32    ].pipeline; break;
                        case LM_GGML_TYPE_F16:     pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F32    ].pipeline; break;
                        case LM_GGML_TYPE_BF16:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_BF16_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q4_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q4_1:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q5_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q5_1:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q8_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q2_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q3_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q4_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q5_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_Q6_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F32   ].pipeline; break;
                        case LM_GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F32].pipeline; break;
                        case LM_GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F32 ].pipeline; break;
                        case LM_GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F32].pipeline; break;
                        case LM_GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ1_M:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F32  ].pipeline; break;
                        case LM_GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F32 ].pipeline; break;
                        case LM_GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F32 ].pipeline; break;
                        default: LM_GGML_ABORT("MUL_MAT_ID not implemented");
                    }

                    lm_ggml_metal_kargs_mul_mm_id args = {
                        /*.nei0 =*/ ne20,
                        /*.nei1 =*/ ne21,
                        /*.nbi1 =*/ nb21,
                        /*.ne00 =*/ ne00,
                        /*.ne02 =*/ ne02,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.ne11 =*/ ne11,
                        /*.ne12 =*/ ne12,
                        /*.ne13 =*/ ne13,
                        /*.nb10 =*/ nb10,
                        /*.nb11 =*/ nb11,
                        /*.nb12 =*/ nb12,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0    atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1    atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst     atIndex:3];
                    [encoder setBuffer:id_src2 offset:offs_src2    atIndex:4];

                    [encoder setThreadgroupMemoryLength:LM_GGML_PAD(8192 + dst_rows*4/*sizeof(ushort2)*/, 16) atIndex:0];

                    [encoder dispatchThreadgroups:MTLSizeMake((ne21 + 31)/32, (ne01 + 63)/64, n_as) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    int nth0 = 32;
                    int nth1 = 1;
                    int nrows = 1;
                    //printf("vector: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

                    id<MTLComputePipelineState> pipeline = nil;

                    // use custom matrix x vector kernel
                    switch (src0t) {
                        case LM_GGML_TYPE_F32:
                            {
                                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_F16:
                            {
                                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);
                                nth0 = 32;
                                nth1 = 1;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_BF16:
                            {
                                LM_GGML_ASSERT(src1t == LM_GGML_TYPE_F32);
                                nth0 = 32;
                                nth1 = 1;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_BF16_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q4_0:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q4_1:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q5_0:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q5_1:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q8_0:
                            {
                                nth0 = 8;
                                nth1 = 8;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q2_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q3_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q4_K:
                            {
                                nth0 = 4; //1;
                                nth1 = 8; //32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q5_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_Q6_K:
                            {
                                nth0 = 2;
                                nth1 = 32;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ2_XXS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ2_XS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ3_XXS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ3_S:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ2_S:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ1_S:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ1_M:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ4_NL:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32].pipeline;
                            } break;
                        case LM_GGML_TYPE_IQ4_XS:
                            {
                                nth0 = 4;
                                nth1 = 16;
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32].pipeline;
                            } break;
                        default:
                            {
                                LM_GGML_LOG_ERROR("Asserting on type %d\n", (int)src2t);
                                LM_GGML_ABORT("not implemented");
                            }
                    };

                    if (lm_ggml_is_quantized(src0t)) {
                        LM_GGML_ASSERT(ne00 >= nth0*nth1);
                    }

                    lm_ggml_metal_kargs_mul_mv_id args = {
                        /*.nei0 =*/ ne20,
                        /*.nei1 =*/ ne21,
                        /*.nbi1 =*/ nb21,
                        /*.ne00 =*/ ne00,
                        /*.ne01 =*/ ne01,
                        /*.ne02 =*/ ne02,
                        /*.nb00 =*/ nb00,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.ne10 =*/ ne10,
                        /*.ne11 =*/ ne11,
                        /*.ne12 =*/ ne12,
                        /*.ne13 =*/ ne13,
                        /*.nb10 =*/ nb10,
                        /*.nb11 =*/ nb11,
                        /*.nb12 =*/ nb12,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.nb1  =*/ nb1,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];
                    [encoder setBuffer:id_src2 offset:offs_src2 atIndex:4];

                    const int64_t _ne1 = 1;
                    const int tgz = dst_rows;

                    if (src0t == LM_GGML_TYPE_Q4_0  || src0t == LM_GGML_TYPE_Q4_1  || src0t == LM_GGML_TYPE_Q5_0 ||
                            src0t == LM_GGML_TYPE_Q5_1  || src0t == LM_GGML_TYPE_Q8_0  || src0t == LM_GGML_TYPE_Q2_K ||
                            src0t == LM_GGML_TYPE_IQ1_S || src0t == LM_GGML_TYPE_IQ1_M || src0t == LM_GGML_TYPE_IQ2_S) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_IQ2_XXS || src0t == LM_GGML_TYPE_IQ2_XS) {
                        const int mem_size = src0t == LM_GGML_TYPE_IQ2_XXS ? 256*8+128 : 512*8+128;
                        [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_IQ3_XXS || src0t == LM_GGML_TYPE_IQ3_S) {
                        const int mem_size = src0t == LM_GGML_TYPE_IQ3_XXS ? 256*4+128 : 512*4;
                        [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_IQ4_NL || src0t == LM_GGML_TYPE_IQ4_XS) {
                        const int mem_size = 32*sizeof(float);
                        [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q4_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q3_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q5_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                    else if (src0t == LM_GGML_TYPE_Q6_K) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    } else {
                        const int64_t ny = (_ne1 + nrows - 1)/nrows; // = _ne1
                        [encoder dispatchThreadgroups:MTLSizeMake(ne01, ny, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                    }
                }
            } break;
        case LM_GGML_OP_GET_ROWS:
            {
                id<MTLComputePipelineState> pipeline = nil;

                switch (src0->type) {
                    case LM_GGML_TYPE_F32:     pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F32    ].pipeline; break;
                    case LM_GGML_TYPE_F16:     pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_F16    ].pipeline; break;
                    case LM_GGML_TYPE_BF16:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_BF16   ].pipeline; break;
                    case LM_GGML_TYPE_Q4_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0   ].pipeline; break;
                    case LM_GGML_TYPE_Q4_1:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1   ].pipeline; break;
                    case LM_GGML_TYPE_Q5_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0   ].pipeline; break;
                    case LM_GGML_TYPE_Q5_1:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1   ].pipeline; break;
                    case LM_GGML_TYPE_Q8_0:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0   ].pipeline; break;
                    case LM_GGML_TYPE_Q2_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K   ].pipeline; break;
                    case LM_GGML_TYPE_Q3_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K   ].pipeline; break;
                    case LM_GGML_TYPE_Q4_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K   ].pipeline; break;
                    case LM_GGML_TYPE_Q5_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K   ].pipeline; break;
                    case LM_GGML_TYPE_Q6_K:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K   ].pipeline; break;
                    case LM_GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS].pipeline; break;
                    case LM_GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS ].pipeline; break;
                    case LM_GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS].pipeline; break;
                    case LM_GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S  ].pipeline; break;
                    case LM_GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S  ].pipeline; break;
                    case LM_GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S  ].pipeline; break;
                    case LM_GGML_TYPE_IQ1_M:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M  ].pipeline; break;
                    case LM_GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL ].pipeline; break;
                    case LM_GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS ].pipeline; break;
                    case LM_GGML_TYPE_I32:     pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GET_ROWS_I32    ].pipeline; break;
                    default: LM_GGML_ABORT("not implemented");
                }

                lm_ggml_metal_kargs_get_rows args = {
                    /*.ne00 =*/ ne00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.ne10 =*/ ne10,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0     offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_src1     offset:offs_src1 atIndex:1];
                [encoder setBuffer:id_dst      offset:offs_dst  atIndex:2];
                [encoder setBytes:&args length:sizeof(args) atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(ne10, ne11, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } break;
        case LM_GGML_OP_RMS_NORM:
            {
                LM_GGML_ASSERT(ne00 % 4 == 0);
                LM_GGML_ASSERT(lm_ggml_is_contiguous_1(src0));

                float eps;
                memcpy(&eps, dst->op_params, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_RMS_NORM].pipeline;

                int nth = 32; // SIMD width

                while (nth < ne00/4 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                nth = MIN(nth, ne00/4);

                lm_ggml_metal_kargs_rms_norm args = {
                    /*.ne00   =*/ ne00,
                    /*.ne00_4 =*/ ne00/4,
                    /*.nb01   =*/ nb01,
                    /*.eps    =*/ eps,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                const int64_t nrows = lm_ggml_nrows(src0);

                [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_GROUP_NORM:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

                float eps;
                memcpy(&eps, dst->op_params + 1, sizeof(float));

                const int32_t n_groups = ((const int32_t *) dst->op_params)[0];

                int nth = 32; // SIMD width

                //while (nth < ne00/4 && nth < 1024) {
                //    nth *= 2;
                //}

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_GROUP_NORM].pipeline;

                lm_ggml_metal_kargs_group_norm args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.n_groups =*/ n_groups,
                    /*.eps =*/ eps,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0  offset:offs_src0        atIndex:0];
                [encoder setBuffer:id_dst   offset:offs_dst         atIndex:1];
                [encoder setBytes:&args     length:sizeof(args)     atIndex:2];
                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_NORM:
            {
                LM_GGML_ASSERT(ne00 % 4 == 0);
                LM_GGML_ASSERT(lm_ggml_is_contiguous_1(src0));

                float eps;
                memcpy(&eps, dst->op_params, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_NORM].pipeline;

                int nth = 32; // SIMD width

                while (nth < ne00/4 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                nth = MIN(nth, ne00/4);

                lm_ggml_metal_kargs_norm args = {
                    /*.ne00   =*/ ne00,
                    /*.ne00_4 =*/ ne00/4,
                    /*.nb01   =*/ nb01,
                    /*.eps    =*/ eps,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                const int64_t nrows = lm_ggml_nrows(src0);

                [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_ROPE:
            {
                // make sure we have one or more position id(ne10) per token(ne02)
                LM_GGML_ASSERT(ne10 % ne02 == 0);
                LM_GGML_ASSERT(ne10 >= ne02);

                const int nth = MIN(1024, ne00);

                const int n_past     = ((const int32_t *) dst->op_params)[0];
                const int n_dims     = ((const int32_t *) dst->op_params)[1];
                const int mode       = ((const int32_t *) dst->op_params)[2];
                // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
                const int n_ctx_orig = ((const int32_t *) dst->op_params)[4];

                float freq_base;
                float freq_scale;
                float ext_factor;
                float attn_factor;
                float beta_fast;
                float beta_slow;

                memcpy(&freq_base,   (const int32_t *) dst->op_params +  5, sizeof(float));
                memcpy(&freq_scale,  (const int32_t *) dst->op_params +  6, sizeof(float));
                memcpy(&ext_factor,  (const int32_t *) dst->op_params +  7, sizeof(float));
                memcpy(&attn_factor, (const int32_t *) dst->op_params +  8, sizeof(float));
                memcpy(&beta_fast,   (const int32_t *) dst->op_params +  9, sizeof(float));
                memcpy(&beta_slow,   (const int32_t *) dst->op_params + 10, sizeof(float));

                const bool is_neox = mode & LM_GGML_ROPE_TYPE_NEOX;

                id<MTLComputePipelineState> pipeline = nil;

                if (!is_neox) {
                    switch (src0->type) {
                        case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F32].pipeline; break;
                        case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ROPE_NORM_F16].pipeline; break;
                        default: LM_GGML_ABORT("fatal error");
                    };
                } else {
                    switch (src0->type) {
                        case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F32].pipeline; break;
                        case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F16].pipeline; break;
                        default: LM_GGML_ABORT("fatal error");
                    };
                }

                lm_ggml_metal_kargs_rope args = {
                    /*.ne00        =*/ ne00,
                    /*.ne01        =*/ ne01,
                    /*.ne02        =*/ ne02,
                    /*.ne03        =*/ ne03,
                    /*.nb00        =*/ nb00,
                    /*.nb01        =*/ nb01,
                    /*.nb02        =*/ nb02,
                    /*.nb03        =*/ nb03,
                    /*.ne0         =*/ ne0,
                    /*.ne1         =*/ ne1,
                    /*.ne2         =*/ ne2,
                    /*.ne3         =*/ ne3,
                    /*.nb0         =*/ nb0,
                    /*.nb1         =*/ nb1,
                    /*.nb2         =*/ nb2,
                    /*.nb3         =*/ nb3,
                    /*.n_past      =*/ n_past,
                    /*.n_dims      =*/ n_dims,
                    /*.n_ctx_orig  =*/ n_ctx_orig,
                    /*.freq_base   =*/ freq_base,
                    /*.freq_scale  =*/ freq_scale,
                    /*.ext_factor  =*/ ext_factor,
                    /*.attn_factor =*/ attn_factor,
                    /*.beta_fast   =*/ beta_fast,
                    /*.beta_slow   =*/ beta_slow,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args)     atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0     atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1     atIndex:2];
                if (id_src2 != nil) {
                    [encoder setBuffer:id_src2 offset:offs_src2 atIndex:3];
                } else {
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:3];
                }
                [encoder setBuffer:id_dst  offset:offs_dst      atIndex:4];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_IM2COL:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
                LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F16 || dst->type == LM_GGML_TYPE_F32);

                const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
                const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
                const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
                const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
                const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
                const int32_t d1 = ((const int32_t *)(dst->op_params))[5];

                const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

                const int32_t N  = src1->ne[is_2D ? 3 : 2];
                const int32_t IC = src1->ne[is_2D ? 2 : 1];
                const int32_t IH = is_2D ? src1->ne[1] : 1;
                const int32_t IW =         src1->ne[0];

                const int32_t KH = is_2D ? src0->ne[1] : 1;
                const int32_t KW =         src0->ne[0];

                const int32_t OH = is_2D ? dst->ne[2] : 1;
                const int32_t OW =         dst->ne[1];

                const int32_t CHW = IC * KH * KW;

                const uint64_t ofs0 = src1->nb[is_2D ? 3 : 2] / 4;
                const uint64_t ofs1 = src1->nb[is_2D ? 2 : 1] / 4;

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_IM2COL_F32].pipeline;

                const bool is_gt_mttpt = ((size_t)(N * KH * KW)) > pipeline.maxTotalThreadsPerThreadgroup;

                switch (dst->type) {
                    case LM_GGML_TYPE_F32: {
                        pipeline = (is_gt_mttpt ?
                                    ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F32].pipeline
                                    :
                                    ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_IM2COL_F32].pipeline);
                    } break;
                    case LM_GGML_TYPE_F16: {
                        pipeline = (is_gt_mttpt ?
                                    ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F16].pipeline
                                    :
                                    ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_IM2COL_F16].pipeline);
                    } break;
                    default: LM_GGML_ABORT("fatal error");
                };

                lm_ggml_metal_kargs_im2col args = {
                    /*.ofs0 =*/ ofs0,
                    /*.ofs1 =*/ ofs1,
                    /*.IW   =*/ IW,
                    /*.IH   =*/ IH,
                    /*.CHW  =*/ CHW,
                    /*.s0   =*/ s0,
                    /*.s1   =*/ s1,
                    /*.p0   =*/ p0,
                    /*.p1   =*/ p1,
                    /*.d0   =*/ d0,
                    /*.d1   =*/ d1,
                    /*.N    =*/ N,
                    /*.KH   =*/ KH,
                    /*.KW   =*/ KW,
                    /*.KHW  =*/ KH * KW,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src1 offset:offs_src1       atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst        atIndex:1];
                [encoder setBytes:&args length:sizeof(args)       atIndex:2];

                if (is_gt_mttpt) {
                    const uint64_t n_threads = MIN(pipeline.maxTotalThreadsPerThreadgroup, (uint64_t)N);

                    const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

                    [encoder dispatchThreadgroups:MTLSizeMake(quotient * CHW, OH, OW) threadsPerThreadgroup:MTLSizeMake(n_threads, 1, 1)];
                } else {
                    [encoder dispatchThreadgroups:MTLSizeMake(IC, OH, OW) threadsPerThreadgroup:MTLSizeMake(N, KH, KW)];
                }
            } break;
        case LM_GGML_OP_CONV_TRANSPOSE_1D:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16 || src0->type == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);

                const int32_t s0 = ((const int32_t *)(dst->op_params))[0];

                const int32_t IC = src1->ne[1];
                const int32_t IL = src1->ne[0];

                const int32_t K  = src0->ne[0];

                const int32_t OL = dst->ne[0];
                const int32_t OC = dst->ne[1];

                id<MTLComputePipelineState> pipeline;

                switch (src0->type) {
                    case LM_GGML_TYPE_F32: {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F32_F32].pipeline;
                    } break;
                    case LM_GGML_TYPE_F16: {
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F16_F32].pipeline;
                    } break;
                    default: LM_GGML_ABORT("fatal error");
                };

                lm_ggml_metal_kargs_conv_transpose_1d args = {
                    /*.IC =*/ IC,
                    /*.IL =*/ IL,
                    /*.K  =*/ K,
                    /*.s0 =*/ s0,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0         atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1         atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst          atIndex:2];
                [encoder setBytes:&args    length:sizeof(args)       atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(OL, OC, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_UPSCALE:
            {
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);

                const float sf0 = (float)ne0/src0->ne[0];
                const float sf1 = (float)ne1/src0->ne[1];
                const float sf2 = (float)ne2/src0->ne[2];
                const float sf3 = (float)ne3/src0->ne[3];

                const id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_UPSCALE_F32].pipeline;

                lm_ggml_metal_kargs_upscale args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0 =*/ ne0,
                    /*.ne1 =*/ ne1,
                    /*.ne2 =*/ ne2,
                    /*.ne3 =*/ ne3,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                    /*.nb3 =*/ nb3,
                    /*.sf0 =*/ sf0,
                    /*.sf1 =*/ sf1,
                    /*.sf2 =*/ sf2,
                    /*.sf3 =*/ sf3
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_PAD:
            {
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_PAD_F32].pipeline;

                lm_ggml_metal_kargs_pad args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0 =*/ ne0,
                    /*.ne1 =*/ ne1,
                    /*.ne2 =*/ ne2,
                    /*.ne3 =*/ ne3,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                    /*.nb3 =*/ nb3
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_PAD_REFLECT_1D:
            {
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);

                const int32_t p0 = ((const int32_t *)(dst->op_params))[0];
                const int32_t p1 = ((const int32_t *)(dst->op_params))[1];

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_PAD_REFLECT_1D_F32].pipeline;

                lm_ggml_metal_kargs_pad_reflect_1d args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0 =*/ ne0,
                    /*.ne1 =*/ ne1,
                    /*.ne2 =*/ ne2,
                    /*.ne3 =*/ ne3,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                    /*.nb3 =*/ nb3,
                    /*.p0 =*/ p0,
                    /*.p1 =*/ p1
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_ARANGE:
            {
                LM_GGML_ASSERT(dst->type == LM_GGML_TYPE_F32);

                float start;
                float step;

                memcpy(&start, ((const int32_t *) dst->op_params) + 0, sizeof(float));
                memcpy(&step,  ((const int32_t *) dst->op_params) + 2, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ARANGE_F32].pipeline;

                lm_ggml_metal_kargs_arange args = {
                    /*.ne0 =*/ ne0,
                    /*.start =*/ start,
                    /*.step =*/ step
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:0];
                [encoder setBytes:&args length:sizeof(args) atIndex:1];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_TIMESTEP_EMBEDDING:
            {
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);

                const int dim        = dst->op_params[0];
                const int max_period = dst->op_params[1];

                const int half = dim / 2;

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32].pipeline;

                lm_ggml_metal_kargs_timestep_embedding args = {
                    /*.nb1 =*/ nb1,
                    /*.dim =*/ dim,
                    /*.max_period =*/ max_period
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN(1024, half);

                [encoder dispatchThreadgroups:MTLSizeMake(ne00, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_ARGSORT:
            {
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_I32);

                const int nrows = lm_ggml_nrows(src0);

                enum lm_ggml_sort_order order = (enum lm_ggml_sort_order) dst->op_params[0];

                // bitonic sort requires the number of elements to be power of 2
                int64_t ne00_padded = 1;
                while (ne00_padded < ne00) {
                    ne00_padded *= 2;
                }

                // Metal kernels require the buffer size to be multiple of 16 bytes
                // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
                const int mem_size = LM_GGML_PAD(ne00_padded*sizeof(int32_t), 16);

                id<MTLComputePipelineState> pipeline = nil;

                switch (order) {
                    case LM_GGML_SORT_ORDER_ASC:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC].pipeline;  break;
                    case LM_GGML_SORT_ORDER_DESC: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC].pipeline; break;
                    default: LM_GGML_ABORT("fatal error");
                };

                lm_ggml_metal_kargs_argsort args = {
                    /*.ncols =*/ ne00,
                    /*.ncols_pad =*/ ne00_padded
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];
                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(1, nrows, 1) threadsPerThreadgroup:MTLSizeMake(ne00_padded, 1, 1)];
            } break;
        case LM_GGML_OP_LEAKY_RELU:
            {
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);

                float slope;
                memcpy(&slope, dst->op_params, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32].pipeline;

                lm_ggml_metal_kargs_leaky_relu args = {
                    /*.slope =*/ slope
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0   atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst    atIndex:1];
                [encoder setBytes:&args length:sizeof(args)   atIndex:2];

                const int64_t n = lm_ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case LM_GGML_OP_FLASH_ATTN_EXT:
            {
                LM_GGML_ASSERT(ne00 % 4  == 0);
                LM_GGML_ASSERT(ne11 % 32 == 0);

                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(src1->type == src2->type);

                LM_GGML_ASSERT(lm_ggml_are_same_shape (src1, src2));

                struct lm_ggml_tensor * src3 = node->src[3];

                size_t offs_src3 = 0;

                id<MTLBuffer> id_src3 = src3 ? lm_ggml_metal_get_buffer(src3, &offs_src3) : nil;

                LM_GGML_ASSERT(!src3 || src3->type == LM_GGML_TYPE_F16);
                LM_GGML_ASSERT(!src3 || src3->ne[1] >= LM_GGML_PAD(src0->ne[1], 8) &&
                        "the Flash-Attention Metal kernel requires the mask to be padded to 8 and at least n_queries big");

                const int64_t  ne30 = src3 ? src3->ne[0] : 0; LM_GGML_UNUSED(ne30);
                //const int64_t  ne31 = src3 ? src3->ne[1] : 0;
                const int64_t  ne32 = src3 ? src3->ne[2] : 0; LM_GGML_UNUSED(ne32);
                const int64_t  ne33 = src3 ? src3->ne[3] : 0; LM_GGML_UNUSED(ne33);

                const uint64_t nb30 = src3 ? src3->nb[0] : 0; LM_GGML_UNUSED(nb30);
                const uint64_t nb31 = src3 ? src3->nb[1] : 0;
                const uint64_t nb32 = src3 ? src3->nb[2] : 0; LM_GGML_UNUSED(nb32);
                const uint64_t nb33 = src3 ? src3->nb[3] : 0; LM_GGML_UNUSED(nb33);

                const enum lm_ggml_type src2t = src2 ? src2->type : LM_GGML_TYPE_COUNT; LM_GGML_UNUSED(src2t);

                float scale;
                float max_bias;
                float logit_softcap;
                memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
                memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
                memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

                if (logit_softcap != 0.0f) {
                    scale /= logit_softcap;
                }

                const uint32_t n_head      = src0->ne[2];
                const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

                const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
                const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

                id<MTLComputePipelineState> pipeline = nil;

                bool use_vec_kernel = false;

                // TODO: add vec kernels for (ne00%64 == 0) and maybe also for (ne00%32 == 0)
                //       for now avoiding mainly to keep the number of templates/kernels a bit lower
                if (ne01 >= 4 || (ne00%128 != 0)) {
                    switch (src1->type) {
                        case LM_GGML_TYPE_F16:
                            {
                                switch (ne00) {
                                    case 64:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H64 ].pipeline; break;
                                    case 80:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H80 ].pipeline; break;
                                    case 96:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H96 ].pipeline; break;
                                    case 112: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H112].pipeline; break;
                                    case 128: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H128].pipeline; break;
                                    case 256: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H256].pipeline; break;
                                    default:
                                              {
                                                  LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                                  LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                                  LM_GGML_ABORT("add template specialization for this size");
                                              }
                                }
                            } break;
                        case LM_GGML_TYPE_BF16:
                            {
                                switch (ne00) {
                                    case 64:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H64 ].pipeline; break;
                                    case 80:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H80 ].pipeline; break;
                                    case 96:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H96 ].pipeline; break;
                                    case 112: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H112].pipeline; break;
                                    case 128: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H128].pipeline; break;
                                    case 256: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H256].pipeline; break;
                                    default:
                                              {
                                                  LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                                  LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                                  LM_GGML_ABORT("add template specialization for this size");
                                              }
                                }
                            } break;
                        case LM_GGML_TYPE_Q4_0:
                            {
                                switch (ne00) {
                                    case 64:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H64 ].pipeline; break;
                                    case 80:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H80 ].pipeline; break;
                                    case 96:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H96 ].pipeline; break;
                                    case 112: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H112].pipeline; break;
                                    case 128: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H128].pipeline; break;
                                    case 256: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H256].pipeline; break;
                                    default:
                                              {
                                                  LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                                  LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                                  LM_GGML_ABORT("add template specialization for this size");
                                              }
                                }
                            } break;
                        case LM_GGML_TYPE_Q4_1:
                            {
                                switch (ne00) {
                                    case 64:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H64 ].pipeline; break;
                                    case 80:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H80 ].pipeline; break;
                                    case 96:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H96 ].pipeline; break;
                                    case 112: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H112].pipeline; break;
                                    case 128: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H128].pipeline; break;
                                    case 256: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H256].pipeline; break;
                                    default:
                                              {
                                                  LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                                  LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                                  LM_GGML_ABORT("add template specialization for this size");
                                              }
                                }
                            } break;
                        case LM_GGML_TYPE_Q5_0:
                            {
                                switch (ne00) {
                                    case 64:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H64 ].pipeline; break;
                                    case 80:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H80 ].pipeline; break;
                                    case 96:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H96 ].pipeline; break;
                                    case 112: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H112].pipeline; break;
                                    case 128: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H128].pipeline; break;
                                    case 256: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H256].pipeline; break;
                                    default:
                                              {
                                                  LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                                  LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                                  LM_GGML_ABORT("add template specialization for this size");
                                              }
                                }
                            } break;
                        case LM_GGML_TYPE_Q5_1:
                            {
                                switch (ne00) {
                                    case 64:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H64 ].pipeline; break;
                                    case 80:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H80 ].pipeline; break;
                                    case 96:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H96 ].pipeline; break;
                                    case 112: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H112].pipeline; break;
                                    case 128: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H128].pipeline; break;
                                    case 256: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H256].pipeline; break;
                                    default:
                                              {
                                                  LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                                  LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                                  LM_GGML_ABORT("add template specialization for this size");
                                              }
                                }
                            } break;
                        case LM_GGML_TYPE_Q8_0:
                            {
                                switch (ne00) {
                                    case 64:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H64 ].pipeline; break;
                                    case 80:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H80 ].pipeline; break;
                                    case 96:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H96 ].pipeline; break;
                                    case 112: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H112].pipeline; break;
                                    case 128: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H128].pipeline; break;
                                    case 256: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H256].pipeline; break;
                                    default:
                                              {
                                                  LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                                  LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                                  LM_GGML_ABORT("add template specialization for this size");
                                              }
                                }
                            } break;
                        default:
                            {
                                LM_GGML_LOG_ERROR("unsupported type: %d\n", src1->type);
                                LM_GGML_LOG_ERROR("add template specialization for this type\n");
                                LM_GGML_ABORT("add template specialization for this type");
                            }
                    }
                } else {
                    use_vec_kernel = true;

                    switch (ne00) {
                        case 128:
                            {
                                switch (src1->type) {
                                    case LM_GGML_TYPE_F16:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H128].pipeline; break;
                                    case LM_GGML_TYPE_BF16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H128].pipeline; break;
                                    case LM_GGML_TYPE_Q4_0: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H128].pipeline; break;
                                    case LM_GGML_TYPE_Q4_1: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H128].pipeline; break;
                                    case LM_GGML_TYPE_Q5_0: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H128].pipeline; break;
                                    case LM_GGML_TYPE_Q5_1: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H128].pipeline; break;
                                    case LM_GGML_TYPE_Q8_0: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H128].pipeline; break;
                                    default:
                                        {
                                            LM_GGML_LOG_ERROR("unsupported type: %d\n", src1->type);
                                            LM_GGML_LOG_ERROR("add template specialization for this type\n");
                                            LM_GGML_ABORT("add template specialization for this type");
                                        }
                                }
                            } break;
                        case 256:
                            {
                                switch (src1->type) {
                                    case LM_GGML_TYPE_F16:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H256].pipeline; break;
                                    case LM_GGML_TYPE_BF16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H256].pipeline; break;
                                    case LM_GGML_TYPE_Q4_0: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H256].pipeline; break;
                                    case LM_GGML_TYPE_Q4_1: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H256].pipeline; break;
                                    case LM_GGML_TYPE_Q5_0: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H256].pipeline; break;
                                    case LM_GGML_TYPE_Q5_1: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H256].pipeline; break;
                                    case LM_GGML_TYPE_Q8_0: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H256].pipeline; break;
                                    default:
                                        {
                                            LM_GGML_LOG_ERROR("unsupported type: %d\n", src1->type);
                                            LM_GGML_LOG_ERROR("add template specialization for this type\n");
                                            LM_GGML_ABORT("add template specialization for this type");
                                        }
                                }
                            } break;
                        default:
                                  {
                                      LM_GGML_LOG_ERROR("unsupported size: %lld\n", ne00);
                                      LM_GGML_LOG_ERROR("add template specialization for this size\n");
                                      LM_GGML_ABORT("add template specialization for this size");
                                  }
                    }
                }

                lm_ggml_metal_kargs_flash_attn_ext args = {
                    /*.ne01          =*/ ne01,
                    /*.ne02          =*/ ne02,
                    /*.ne03          =*/ ne03,
                    /*.nb01          =*/ nb01,
                    /*.nb02          =*/ nb02,
                    /*.nb03          =*/ nb03,
                    /*.ne11          =*/ ne11,
                    /*.ne_12_2       =*/ ne12,
                    /*.ne_12_3       =*/ ne13,
                    /*.nb_12_1       =*/ nb11,
                    /*.nb_12_2       =*/ nb12,
                    /*.nb_12_3       =*/ nb13,
                    /*.nb31          =*/ nb31,
                    /*.ne1           =*/ ne1,
                    /*.ne2           =*/ ne2,
                    /*.scale         =*/ scale,
                    /*.max_bias      =*/ max_bias,
                    /*.m0            =*/ m0,
                    /*.m1            =*/ m1,
                    /*.n_head_log2   =*/ n_head_log2,
                    /*.logit_softcap =*/ logit_softcap,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args)     atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0     atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1     atIndex:2];
                [encoder setBuffer:id_src2 offset:offs_src2     atIndex:3];
                if (id_src3) {
                    [encoder setBuffer:id_src3 offset:offs_src3 atIndex:4];
                } else {
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:4];
                }
                [encoder setBuffer:id_dst offset:offs_dst       atIndex:5];

                if (!use_vec_kernel) {
                    // half8x8 kernel
                    const int64_t nqptg = 8;  // queries per threadgroup    !! sync with kernel template arguments !!
                    const int64_t ncpsg = 32; // cache values per simdgroup !! sync with kernel template arguments !!

                    LM_GGML_ASSERT(nqptg <= 32);
                    LM_GGML_ASSERT(nqptg  % 8  == 0);
                    LM_GGML_ASSERT(ncpsg  % 32 == 0);

                    // 2*(2*ncpsg + nqptg)*(nsg)
                    // ncpsg soft_max values + ncpsg mask values + a diagonal scaling matrix (in float)
                    //
                    // 16*32*(nsg)
                    // the shared memory needed for the simdgroups to load the KV cache
                    // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
                    //
#define FATTN_SMEM(nsg) (LM_GGML_PAD((nqptg*(ne00 + 2*(2*ncpsg + nqptg)*(nsg)) + 16*32*(nsg))*(sizeof(float)/2), 16))

                    int64_t nsgmax = 2;

                    while (true) {
                        const size_t smem = FATTN_SMEM(nsgmax);
                        if (smem > device.maxThreadgroupMemoryLength) {
                            break;
                        }
                        nsgmax *= 2;
                    }
                    nsgmax /= 2;

                    // simdgroups per threadgroup (a.k.a. warps)
                    const int64_t nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;

                    const size_t smem = FATTN_SMEM(nsg);

                    //printf("smem: %zu, max: %zu, nsg = %d\n", smem, device.maxThreadgroupMemoryLength, (int) nsg);
                    LM_GGML_ASSERT(smem <= device.maxThreadgroupMemoryLength);
                    [encoder setThreadgroupMemoryLength:smem atIndex:0];
#undef FATTN_SMEM
                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nqptg - 1)/nqptg, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                } else {
                    // half4x4 kernel
                    const int64_t nqptg = 1;  // queries per threadgroup    !! sync with kernel template arguments !!
                    const int64_t ncpsg = 32; // cache values per simdgroup !! sync with kernel template arguments !!

                    LM_GGML_ASSERT(nqptg <= 32);
                    LM_GGML_ASSERT(nqptg  % 1  == 0);
                    LM_GGML_ASSERT(ncpsg  % 32 == 0);

                    // ne00 + 2*ncpsg*(nsg)
                    // for each query, we load it as f16 in shared memory (ne00)
                    // and store the soft_max values and the mask
                    //
                    // ne00*(nsg)
                    // each simdgroup has a full f16 head vector in shared mem to accumulate results
                    //
#define FATTN_SMEM(nsg) (LM_GGML_PAD((nqptg*(ne00 + 2*ncpsg*(nsg)) + ne00*(nsg))*(sizeof(float)/2), 16))

                    int64_t nsgmax = 2;

                    while (true) {
                        const size_t smem = FATTN_SMEM(nsgmax);
                        if (smem > device.maxThreadgroupMemoryLength) {
                            break;
                        }
                        nsgmax *= 2;
                    }
                    nsgmax /= 2;

                    // simdgroups per threadgroup (a.k.a. warps)
                    const int64_t nsgt = MAX(2, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32)));

                    int64_t nsg = 1;
                    while (nsg <= nsgt) {
                        nsg *= 2;
                    }
                    nsg /= 2;

                    const size_t smem = FATTN_SMEM(nsg);

                    //printf("smem: %zu, max: %zu, nsg = %d\n", smem, device.maxThreadgroupMemoryLength, (int) nsg);
                    LM_GGML_ASSERT(smem <= device.maxThreadgroupMemoryLength);
                    [encoder setThreadgroupMemoryLength:smem atIndex:0];
#undef FATTN_SMEM
                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nqptg - 1)/nqptg, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                }
            } break;
        case LM_GGML_OP_DUP:
        case LM_GGML_OP_CPY:
        case LM_GGML_OP_CONT:
            {
                id<MTLComputePipelineState> pipeline = nil;

                switch (src0t) {
                    case LM_GGML_TYPE_F32:
                        {
                            LM_GGML_ASSERT(ne0 % lm_ggml_blck_size(dst->type) == 0);

                            switch (dstt) {
                                case LM_GGML_TYPE_F32:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F32].pipeline; break;
                                case LM_GGML_TYPE_F16:    pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_F16].pipeline; break;
                                case LM_GGML_TYPE_BF16:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_BF16].pipeline; break;
                                case LM_GGML_TYPE_Q8_0:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0].pipeline; break;
                                case LM_GGML_TYPE_Q4_0:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0].pipeline; break;
                                case LM_GGML_TYPE_Q4_1:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1].pipeline; break;
                                case LM_GGML_TYPE_Q5_0:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0].pipeline; break;
                                case LM_GGML_TYPE_Q5_1:   pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1].pipeline; break;
                                case LM_GGML_TYPE_IQ4_NL: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    case LM_GGML_TYPE_F16:
                        {
                            switch (dstt) {
                                case LM_GGML_TYPE_F32:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F32].pipeline; break;
                                case LM_GGML_TYPE_F16:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_F16_F16].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    case LM_GGML_TYPE_BF16:
                        {
                            switch (dstt) {
                                case LM_GGML_TYPE_F32:  pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_F32].pipeline; break;
                                case LM_GGML_TYPE_BF16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_BF16_BF16].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    case LM_GGML_TYPE_Q4_0:
                        {
                            switch (dstt) {
                                case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F32].pipeline; break;
                                case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F16].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    case LM_GGML_TYPE_Q4_1:
                        {
                            switch (dstt) {
                                case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F32].pipeline; break;
                                case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F16].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    case LM_GGML_TYPE_Q5_0:
                        {
                            switch (dstt) {
                                case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F32].pipeline; break;
                                case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F16].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    case LM_GGML_TYPE_Q5_1:
                        {
                            switch (dstt) {
                                case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F32].pipeline; break;
                                case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F16].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    case LM_GGML_TYPE_Q8_0:
                        {
                            switch (dstt) {
                                case LM_GGML_TYPE_F32: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F32].pipeline; break;
                                case LM_GGML_TYPE_F16: pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F16].pipeline; break;
                                default: LM_GGML_ABORT("not implemented");
                            };
                        } break;
                    default: LM_GGML_ABORT("not implemented");
                }

                lm_ggml_metal_kargs_cpy args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                LM_GGML_ASSERT(ne00 % lm_ggml_blck_size(src0->type) == 0);
                int nth = MIN(1024, ne00/lm_ggml_blck_size(src0->type));

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

            } break;
        case LM_GGML_OP_SET:
            {
                LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
                LM_GGML_ASSERT(lm_ggml_is_contiguous(dst) && lm_ggml_is_contiguous(src0));

                // src0 and dst as viewed during set
                const size_t dst_nb0 = lm_ggml_element_size(src0);

                const size_t dst_nb1 = ((int32_t *) dst->op_params)[0];
                const size_t dst_nb2 = ((int32_t *) dst->op_params)[1];
                const size_t dst_nb3 = ((int32_t *) dst->op_params)[2];
                const size_t offset  = ((int32_t *) dst->op_params)[3];
                const bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

                if (!inplace) {
                    memcpy(((char *)  dst->data), ((char *) src0->data), lm_ggml_nbytes(dst));
                }

                const int im0 = (ne10 == 0 ? 0 : ne10-1);
                const int im1 = (ne11 == 0 ? 0 : ne11-1);
                const int im2 = (ne12 == 0 ? 0 : ne12-1);
                const int im3 = (ne13 == 0 ? 0 : ne13-1);

                LM_GGML_ASSERT(offset + im0*dst_nb0  + im1*dst_nb1  + im2*dst_nb2  + im3*dst_nb3  <= lm_ggml_nbytes(dst));

                id<MTLComputePipelineState> pipeline = nil;

                switch (src0t) {
                    case LM_GGML_TYPE_F32:
                        LM_GGML_ASSERT(nb10 == sizeof(float));
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SET_F32].pipeline; break;
                    case LM_GGML_TYPE_I32:
                        LM_GGML_ASSERT(nb10 == sizeof(int32_t));
                        pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_SET_I32].pipeline; break;
                    default: LM_GGML_ABORT("fatal error");
                }

                lm_ggml_metal_kargs_set args = {
                    /*.ne10    =*/ ne10,
                    /*.ne11    =*/ ne11,
                    /*.ne12    =*/ ne12,
                    /*.nb10    =*/ nb10,
                    /*.nb11    =*/ nb11,
                    /*.nb12    =*/ nb12,
                    /*.nb13    =*/ nb13,
                    /*.nb1     =*/ dst_nb1,
                    /*.nb2     =*/ dst_nb2,
                    /*.nb3     =*/ dst_nb3,
                    /*.offs    =*/ offset,
                    /*.inplace =*/ inplace,
                };

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne10);

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0    atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1    atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst     atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(ne11, ne12, ne13) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case LM_GGML_OP_POOL_2D:
            {
                LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
                LM_GGML_ASSERT(src0t == LM_GGML_TYPE_F32 && src0t == dstt);

                const int32_t * opts = dst->op_params;
                enum lm_ggml_op_pool op = opts[0];

                id<MTLComputePipelineState> pipeline = nil;
                switch (src0t) {
                    case LM_GGML_TYPE_F32: {
                        switch(op) {
                            case LM_GGML_OP_POOL_AVG:
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_POOL_2D_AVG_F32].pipeline; break;
                            case LM_GGML_OP_POOL_MAX:
                                pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_POOL_2D_MAX_F32].pipeline; break;
                            default: LM_GGML_ASSERT(false && "not implemented");
                        }
                    } break;
                    default: LM_GGML_ASSERT(false && "not implemented");
                }

                const int32_t k0 = opts[1];
                const int32_t k1 = opts[2];
                const int32_t s0 = opts[3];
                const int32_t s1 = opts[4];
                const int32_t p0 = opts[5];
                const int32_t p1 = opts[6];

                const int64_t IH = src0->ne[1];
                const int64_t IW = src0->ne[0];

                const int64_t N  = dst->ne[3];
                const int64_t OC = dst->ne[2];
                const int64_t OH = dst->ne[1];
                const int64_t OW = dst->ne[0];

                const int64_t parallel_elements = N * OC * OH * OW;
                const int64_t n_threads = MIN((int64_t)[pipeline maxTotalThreadsPerThreadgroup], parallel_elements);
                const int64_t n_tg = (parallel_elements + n_threads - 1) / n_threads;

                lm_ggml_metal_kargs_pool_2d args_pool_2d = {
                    /* .k0 = */ k0,
                    /* .k1 = */ k1,
                    /* .s0 = */ s0,
                    /* .s1 = */ s1,
                    /* .p0 = */ p0,
                    /* .p1 = */ p1,
                    /* .IH = */ IH,
                    /* .IW = */ IW,
                    /* .OH = */ OH,
                    /* .OW = */ OW,
                    /* .parallel_elements = */ parallel_elements
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args_pool_2d length:sizeof(args_pool_2d) atIndex:2];

                [encoder dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(n_threads, 1, 1)];
            } break;
            case LM_GGML_OP_ARGMAX:
            {
                LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(lm_ggml_is_contiguous_1(src0));
                LM_GGML_ASSERT(nb00 == lm_ggml_type_size(src0->type));

                const int64_t nrows = lm_ggml_nrows(src0);

                int nth = 32; // SIMD width
                while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
                    nth *= 2;
                }

                id<MTLComputePipelineState> pipeline = ctx->kernels[LM_GGML_METAL_KERNEL_TYPE_ARGMAX].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:3];
                [encoder setThreadgroupMemoryLength:32*sizeof(float)   atIndex:0];
                [encoder setThreadgroupMemoryLength:32*sizeof(int32_t) atIndex:1];

                [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
       default:
            {
                LM_GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, lm_ggml_op_name(dst->op));
                LM_GGML_ABORT("fatal error");
            }
    }
}

static enum lm_ggml_status lm_ggml_metal_graph_compute(
            lm_ggml_backend_t   backend,
        struct lm_ggml_cgraph * gf) {
    struct lm_ggml_backend_metal_context        * ctx     = backend->context;
    struct lm_ggml_backend_metal_device_context * ctx_dev = backend->device->context;

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = 128;

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        const bool should_capture = ctx->capture_next_compute;
        if (should_capture) {
            ctx->capture_next_compute = false;

            if (!ctx->capture_started) {
                // create capture scope
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:ctx_dev->mtl_device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithFormat:@"/tmp/perf-metal.gputrace"]];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    LM_GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // the main thread commits the first few commands immediately
        // command_buffer[n_cb]
        {
            id<MTLCommandBuffer> command_buffer = [ctx->queue commandBufferWithUnretainedReferences];
            ctx->command_buffers[n_cb] = command_buffer;

            [command_buffer enqueue];
            ctx->encode_async(n_cb);
        }

        // prepare the rest of the command buffers asynchronously
        // command_buffer[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> command_buffer = [ctx->queue commandBufferWithUnretainedReferences];
            ctx->command_buffers[cb_idx] = command_buffer;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [command_buffer enqueue];
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // wait for completion and check status of each command buffer
        // needed to detect if the device ran out-of-memory for example (#1881)
        {
            id<MTLCommandBuffer> command_buffer = ctx->command_buffers[n_cb];
            [command_buffer waitUntilCompleted];

            MTLCommandBufferStatus status = [command_buffer status];
            if (status != MTLCommandBufferStatusCompleted) {
                LM_GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                if (status == MTLCommandBufferStatusError) {
                    LM_GGML_LOG_INFO("error: %s\n", [[command_buffer error].localizedDescription UTF8String]);
                }

                return LM_GGML_STATUS_FAILED;
            }
        }

        for (int i = 0; i < n_cb; ++i) {
            id<MTLCommandBuffer> command_buffer = ctx->command_buffers[i];
            [command_buffer waitUntilCompleted];

            MTLCommandBufferStatus status = [command_buffer status];
            if (status != MTLCommandBufferStatusCompleted) {
                LM_GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                if (status == MTLCommandBufferStatusError) {
                    LM_GGML_LOG_INFO("error: %s\n", [[command_buffer error].localizedDescription UTF8String]);
                }

                return LM_GGML_STATUS_FAILED;
            }

            id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->command_buffers[i + 1] : nil);
            if (!next_buffer) {
                continue;
            }

            const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
            if (next_queued) {
                continue;
            }

            if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                LM_GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                return LM_GGML_STATUS_ABORTED;
            }

            [next_buffer commit];
        }

        if (!should_capture && ctx->capture_started) {
            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
        }
    }

    return LM_GGML_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

static void lm_ggml_backend_metal_buffer_free_buffer(lm_ggml_backend_buffer_t buffer) {
    struct lm_ggml_backend_metal_buffer_context * ctx = (struct lm_ggml_backend_metal_buffer_context *)buffer->context;

    for (int i = 0; i < ctx->n_buffers; i++) {
        [ctx->buffers[i].metal release];
    }

    lm_ggml_backend_metal_buffer_rset_free(ctx);
    lm_ggml_backend_metal_device_rel(buffer->buft->device->context);

    if (ctx->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ctx->all_data, ctx->all_size);
#else
        free(ctx->all_data);
#endif
    }

    free(ctx);
}

static void * lm_ggml_backend_metal_buffer_get_base(lm_ggml_backend_buffer_t buffer) {
    struct lm_ggml_backend_metal_buffer_context * ctx = (struct lm_ggml_backend_metal_buffer_context *)buffer->context;

    return ctx->all_data;
}

static void lm_ggml_backend_metal_buffer_memset_tensor(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    LM_GGML_UNUSED(buffer);
}

static void lm_ggml_backend_metal_buffer_set_tensor(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    LM_GGML_UNUSED(buffer);
}

static void lm_ggml_backend_metal_buffer_get_tensor(lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    LM_GGML_UNUSED(buffer);
}

static bool lm_ggml_backend_metal_buffer_cpy_tensor(lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst) {
    if (lm_ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, lm_ggml_nbytes(src));
        return true;
    }
    return false;

    LM_GGML_UNUSED(buffer);
}

static void lm_ggml_backend_metal_buffer_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    struct lm_ggml_backend_metal_buffer_context * ctx = (struct lm_ggml_backend_metal_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

static struct lm_ggml_backend_buffer_i lm_ggml_backend_metal_buffer_i = {
    /* .free_buffer     = */ lm_ggml_backend_metal_buffer_free_buffer,
    /* .get_base        = */ lm_ggml_backend_metal_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ lm_ggml_backend_metal_buffer_memset_tensor,
    /* .set_tensor      = */ lm_ggml_backend_metal_buffer_set_tensor,
    /* .get_tensor      = */ lm_ggml_backend_metal_buffer_get_tensor,
    /* .cpy_tensor      = */ lm_ggml_backend_metal_buffer_cpy_tensor,
    /* .clear           = */ lm_ggml_backend_metal_buffer_clear,
    /* .reset           = */ NULL,
};

// default buffer type

static const char * lm_ggml_backend_metal_buffer_type_get_name(lm_ggml_backend_buffer_type_t buft) {
    return "Metal";

    LM_GGML_UNUSED(buft);
}

static void lm_ggml_backend_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef LM_GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        LM_GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            LM_GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        LM_GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    LM_GGML_UNUSED(device);
    LM_GGML_UNUSED(size_aligned);
}

static lm_ggml_backend_buffer_t lm_ggml_backend_metal_buffer_type_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    struct lm_ggml_backend_metal_buffer_context * ctx = calloc(1, sizeof(struct lm_ggml_backend_metal_buffer_context));

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    struct lm_ggml_backend_metal_device_context * ctx_dev = (struct lm_ggml_backend_metal_device_context *)buft->device->context;
    id<MTLDevice> device = lm_ggml_backend_metal_device_acq(ctx_dev);

    ctx->all_data = lm_ggml_metal_host_malloc(size_aligned);
    ctx->all_size = size_aligned;
    ctx->owned = true;
    ctx->n_buffers = 1;

    if (ctx->all_data != NULL) {
        ctx->buffers[0].data  = ctx->all_data;
        ctx->buffers[0].size  = size;
        ctx->buffers[0].metal = nil;

        if (size_aligned > 0) {
            ctx->buffers[0].metal = [device newBufferWithBytesNoCopy:ctx->all_data
                                            length:size_aligned
                                            options:MTLResourceStorageModeShared
                                            deallocator:nil];
        }
    }

    if (size_aligned > 0 && (ctx->all_data == NULL || ctx->buffers[0].metal == nil)) {
        LM_GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(ctx);
        lm_ggml_backend_metal_device_rel(ctx_dev);
        return NULL;
    }

    if (!lm_ggml_backend_metal_buffer_rset_init(ctx, ctx_dev, device)) {
        LM_GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(ctx);
        lm_ggml_backend_metal_device_rel(ctx_dev);
        return NULL;
    }

    //lm_ggml_backend_metal_log_allocated_size(device, size_aligned);

    return lm_ggml_backend_buffer_init(buft, lm_ggml_backend_metal_buffer_i, ctx, size);
}

static size_t lm_ggml_backend_metal_buffer_type_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    return 32;
    LM_GGML_UNUSED(buft);
}

static size_t lm_ggml_backend_metal_buffer_type_get_max_size(lm_ggml_backend_buffer_type_t buft) {
    id<MTLDevice> device = lm_ggml_backend_metal_device_acq(buft->device->context);
    const size_t max_size = device.maxBufferLength;
    lm_ggml_backend_metal_device_rel(buft->device->context);

    return max_size;

    LM_GGML_UNUSED(buft);
}

static bool lm_ggml_backend_metal_buffer_type_is_host(lm_ggml_backend_buffer_type_t buft) {
    return true;

    LM_GGML_UNUSED(buft);
}

lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type(void) {
    static struct lm_ggml_backend_buffer_type lm_ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ lm_ggml_backend_metal_buffer_type_get_name,
            /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_alloc_buffer,
            /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_get_alignment,
            /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_get_max_size,
            /* .get_alloc_size   = */ NULL, // defaults to lm_ggml_nbytes
            /* .is_host          = */ lm_ggml_backend_metal_buffer_type_is_host,
        },
        /* .device  = */ &g_lm_ggml_backend_metal_device,
        /* .context = */ NULL,
    };

    return &lm_ggml_backend_buffer_type_metal;
}

static const char * lm_ggml_backend_metal_buffer_from_ptr_type_get_name(lm_ggml_backend_buffer_type_t buft) {
    return "Metal_Mapped";

    LM_GGML_UNUSED(buft);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_from_ptr_type(void) {
    static struct lm_ggml_backend_buffer_type lm_ggml_backend_buffer_from_ptr_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ lm_ggml_backend_metal_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_alloc_buffer,
            /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_get_alignment,
            /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_get_max_size,
            /* .get_alloc_size   = */ NULL, // defaults to lm_ggml_nbytes
            /* .is_host          = */ lm_ggml_backend_metal_buffer_type_is_host,
        },
        /* .device  = */ &g_lm_ggml_backend_metal_device,
        /* .context = */ NULL,
    };

    return &lm_ggml_backend_buffer_from_ptr_type_metal;
}

// TODO: obsoleted by lm_ggml_backend_metal_device_buffer_from_ptr
lm_ggml_backend_buffer_t lm_ggml_backend_metal_buffer_from_ptr(void * data, size_t size, size_t max_size) {
    struct lm_ggml_backend_metal_buffer_context * ctx = calloc(1, sizeof(struct lm_ggml_backend_metal_buffer_context));

    ctx->all_data = data;
    ctx->all_size = size;
    ctx->owned = false;
    ctx->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) data % size_page;
        data  = (void *) ((char *) data - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    struct lm_ggml_backend_metal_device_context * ctx_dev = &g_lm_ggml_ctx_dev_main;
    id<MTLDevice> device = lm_ggml_backend_metal_device_acq(ctx_dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= device.maxBufferLength) {
        ctx->buffers[ctx->n_buffers].data  = data;
        ctx->buffers[ctx->n_buffers].size  = size;
        ctx->buffers[ctx->n_buffers].metal = nil;

        if (size_aligned > 0) {
            ctx->buffers[ctx->n_buffers].metal = [device newBufferWithBytesNoCopy:data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                LM_GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                return false;
            }
        }

        lm_ggml_backend_metal_log_allocated_size(device, size_aligned);

        ++ctx->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = device.maxBufferLength - size_ovlp;
        const size_t size_view = device.maxBufferLength;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            ctx->buffers[ctx->n_buffers].data  = (void *) ((uint8_t *) data + i);
            ctx->buffers[ctx->n_buffers].size  = size_step_aligned;
            ctx->buffers[ctx->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                ctx->buffers[ctx->n_buffers].metal = [device newBufferWithBytesNoCopy:(void *) ((uint8_t *) data + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (ctx->buffers[ctx->n_buffers].metal == nil) {
                    LM_GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }
            }

            lm_ggml_backend_metal_log_allocated_size(device, size_step_aligned);

            if (i + size_step < size) {
                LM_GGML_LOG_INFO("\n");
            }

            ++ctx->n_buffers;
        }
    }

    if (!lm_ggml_backend_metal_buffer_rset_init(ctx, ctx_dev, device)) {
        LM_GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(ctx);
        lm_ggml_backend_metal_device_rel(ctx_dev);
        return NULL;
    }

    return lm_ggml_backend_buffer_init(lm_ggml_backend_metal_buffer_from_ptr_type(), lm_ggml_backend_metal_buffer_i, ctx, size);
}

// backend

static const char * lm_ggml_backend_metal_name(lm_ggml_backend_t backend) {
    return "Metal";

    LM_GGML_UNUSED(backend);
}

static void lm_ggml_backend_metal_free(lm_ggml_backend_t backend) {
    struct lm_ggml_backend_metal_context        * ctx     = backend->context;
    struct lm_ggml_backend_metal_device_context * ctx_dev = backend->device->context;

    lm_ggml_backend_metal_device_rel(ctx_dev);
    lm_ggml_metal_free(ctx);

    free(backend);
}

static enum lm_ggml_status lm_ggml_backend_metal_graph_compute(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph) {
    return lm_ggml_metal_graph_compute(backend, cgraph);
}

static void lm_ggml_backend_metal_set_n_cb(lm_ggml_backend_t backend, int n_cb) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    struct lm_ggml_backend_metal_context * ctx = (struct lm_ggml_backend_metal_context *)backend->context;

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, LM_GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            LM_GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        id<MTLCommandBuffer> command_buffer  = ctx->command_buffers[cb_idx];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        int node_start = 0;
        int node_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            node_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            node_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        const bool should_capture = ctx->capture_next_compute;

        for (int idx = node_start; idx < node_end; ++idx) {
            if (should_capture) {
                [encoder pushDebugGroup:[NSString stringWithCString:lm_ggml_op_desc(lm_ggml_graph_node(ctx->gf, idx)) encoding:NSUTF8StringEncoding]];
            }

            lm_ggml_metal_encode_node(backend, idx, encoder);

            if (should_capture) {
                [encoder popDebugGroup];
            }
        }

        [encoder endEncoding];

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [command_buffer commit];
        }
    });
}

static struct lm_ggml_backend_i lm_ggml_backend_metal_i = {
    /* .get_name                = */ lm_ggml_backend_metal_name,
    /* .free                    = */ lm_ggml_backend_metal_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ lm_ggml_backend_metal_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static lm_ggml_guid_t lm_ggml_backend_metal_guid(void) {
    static lm_ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

// TODO: remove in the future
lm_ggml_backend_t lm_ggml_backend_metal_init(void) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_reg_dev_get(lm_ggml_backend_metal_reg(), 0);

    struct lm_ggml_backend_metal_context * ctx = lm_ggml_metal_init(dev);
    if (ctx == NULL) {
        LM_GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    lm_ggml_backend_t backend = malloc(sizeof(struct lm_ggml_backend));

    *backend = (struct lm_ggml_backend) {
        /* .guid      = */ lm_ggml_backend_metal_guid(),
        /* .interface = */ lm_ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    lm_ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool lm_ggml_backend_is_metal(lm_ggml_backend_t backend) {
    return backend != NULL && lm_ggml_guid_matches(backend->guid, lm_ggml_backend_metal_guid());
}

void lm_ggml_backend_metal_set_abort_callback(lm_ggml_backend_t backend, lm_ggml_abort_callback abort_callback, void * user_data) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    struct lm_ggml_backend_metal_context * ctx = (struct lm_ggml_backend_metal_context *)backend->context;

    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool lm_ggml_backend_metal_supports_family(lm_ggml_backend_t backend, int family) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    struct lm_ggml_backend_metal_device_context * ctx_dev = backend->device->context;

    return [ctx_dev->mtl_device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void lm_ggml_backend_metal_capture_next_compute(lm_ggml_backend_t backend) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    struct lm_ggml_backend_metal_context * ctx = (struct lm_ggml_backend_metal_context *)backend->context;
    ctx->capture_next_compute = true;
}

// backend device

static const char * lm_ggml_backend_metal_device_get_name(lm_ggml_backend_dev_t dev) {
    return "Metal";

    LM_GGML_UNUSED(dev);
}

static const char * lm_ggml_backend_metal_device_get_description(lm_ggml_backend_dev_t dev) {
    // acq/rel just to populate ctx->name in case it hasn't been done yet
    struct lm_ggml_backend_metal_device_context * ctx_dev = (struct lm_ggml_backend_metal_device_context *)dev->context;
    lm_ggml_backend_metal_device_acq(ctx_dev);
    lm_ggml_backend_metal_device_rel(ctx_dev);

    return ctx_dev->name;
}

static void lm_ggml_backend_metal_device_get_memory(lm_ggml_backend_dev_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        struct lm_ggml_backend_metal_device_context * ctx_dev = (struct lm_ggml_backend_metal_device_context *)dev->context;
        id<MTLDevice> device = lm_ggml_backend_metal_device_acq(ctx_dev);

        *total = device.recommendedMaxWorkingSetSize;
        *free  = *total - device.currentAllocatedSize;

        lm_ggml_backend_metal_device_rel(ctx_dev);
    } else {
        *free = 1;
        *total = 1;
    }
}

static enum lm_ggml_backend_dev_type lm_ggml_backend_metal_device_get_type(lm_ggml_backend_dev_t dev) {
    return LM_GGML_BACKEND_DEVICE_TYPE_GPU;

    LM_GGML_UNUSED(dev);
}

static void lm_ggml_backend_metal_device_get_props(lm_ggml_backend_dev_t dev, struct lm_ggml_backend_dev_props * props) {
    props->name        = lm_ggml_backend_metal_device_get_name(dev);
    props->description = lm_ggml_backend_metal_device_get_description(dev);
    props->type        = lm_ggml_backend_metal_device_get_type(dev);
    lm_ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = (struct lm_ggml_backend_dev_caps) {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static lm_ggml_backend_t lm_ggml_backend_metal_device_init(lm_ggml_backend_dev_t dev, const char * params) {
    struct lm_ggml_backend_metal_context * ctx = lm_ggml_metal_init(dev);
    if (ctx == NULL) {
        LM_GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    lm_ggml_backend_t backend = malloc(sizeof(struct lm_ggml_backend));

    *backend = (struct lm_ggml_backend) {
        /* .guid      = */ lm_ggml_backend_metal_guid(),
        /* .interface = */ lm_ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    lm_ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    LM_GGML_UNUSED(params);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_device_get_buffer_type(lm_ggml_backend_dev_t dev) {
    return lm_ggml_backend_metal_buffer_type();

    LM_GGML_UNUSED(dev);
}

static lm_ggml_backend_buffer_t lm_ggml_backend_metal_device_buffer_from_ptr(lm_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    struct lm_ggml_backend_metal_buffer_context * ctx = calloc(1, sizeof(struct lm_ggml_backend_metal_buffer_context));

    ctx->all_data = ptr;
    ctx->all_size = size;
    ctx->owned = false;
    ctx->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    struct lm_ggml_backend_metal_device_context * ctx_dev = (struct lm_ggml_backend_metal_device_context *)dev->context;
    id<MTLDevice> device = lm_ggml_backend_metal_device_acq(ctx_dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= device.maxBufferLength) {
        ctx->buffers[ctx->n_buffers].data  = ptr;
        ctx->buffers[ctx->n_buffers].size  = size;
        ctx->buffers[ctx->n_buffers].metal = nil;

        if (size_aligned > 0) {
            ctx->buffers[ctx->n_buffers].metal = [device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                LM_GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                return false;
            }
        }

        lm_ggml_backend_metal_log_allocated_size(device, size_aligned);

        ++ctx->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = device.maxBufferLength - size_ovlp;
        const size_t size_view = device.maxBufferLength;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            ctx->buffers[ctx->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            ctx->buffers[ctx->n_buffers].size  = size_step_aligned;
            ctx->buffers[ctx->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                ctx->buffers[ctx->n_buffers].metal = [device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (ctx->buffers[ctx->n_buffers].metal == nil) {
                    LM_GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }
            }

            lm_ggml_backend_metal_log_allocated_size(device, size_step_aligned);

            if (i + size_step < size) {
                LM_GGML_LOG_INFO("\n");
            }

            ++ctx->n_buffers;
        }
    }

    if (!lm_ggml_backend_metal_buffer_rset_init(ctx, ctx_dev, device)) {
        LM_GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(ctx);
        lm_ggml_backend_metal_device_rel(ctx_dev);
        return NULL;
    }

    return lm_ggml_backend_buffer_init(lm_ggml_backend_metal_buffer_from_ptr_type(), lm_ggml_backend_metal_buffer_i, ctx, size);
}

static bool lm_ggml_backend_metal_device_supports_op(lm_ggml_backend_dev_t dev, const struct lm_ggml_tensor * op) {
    struct lm_ggml_backend_metal_device_context * ctx_dev = dev->context;

    return lm_ggml_metal_supports_op(ctx_dev, op);
}

static bool lm_ggml_backend_metal_device_supports_buft(lm_ggml_backend_dev_t dev, lm_ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == lm_ggml_backend_metal_buffer_type_get_name ||
            buft->iface.get_name == lm_ggml_backend_metal_buffer_from_ptr_type_get_name;

    LM_GGML_UNUSED(dev);
}

static bool lm_ggml_backend_metal_device_offload_op(lm_ggml_backend_dev_t dev, const struct lm_ggml_tensor * op) {
    return false;

    LM_GGML_UNUSED(dev);
    LM_GGML_UNUSED(op);
}

static struct lm_ggml_backend_device_i lm_ggml_backend_metal_device_i = {
    /* .get_name             = */ lm_ggml_backend_metal_device_get_name,
    /* .get_description      = */ lm_ggml_backend_metal_device_get_description,
    /* .get_memory           = */ lm_ggml_backend_metal_device_get_memory,
    /* .get_type             = */ lm_ggml_backend_metal_device_get_type,
    /* .get_props            = */ lm_ggml_backend_metal_device_get_props,
    /* .init_backend         = */ lm_ggml_backend_metal_device_init,
    /* .get_buffer_type      = */ lm_ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ lm_ggml_backend_metal_device_buffer_from_ptr,
    /* .supports_op          = */ lm_ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ lm_ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ lm_ggml_backend_metal_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend registry

static const char * lm_ggml_backend_metal_reg_get_name(lm_ggml_backend_reg_t reg) {
    return "Metal";

    LM_GGML_UNUSED(reg);
}

static size_t lm_ggml_backend_metal_reg_device_count(lm_ggml_backend_reg_t reg) {
    return 1;

    LM_GGML_UNUSED(reg);
}

static lm_ggml_backend_dev_t lm_ggml_backend_metal_reg_device_get(lm_ggml_backend_reg_t reg, size_t index) {
    LM_GGML_ASSERT(index == 0);

    return &g_lm_ggml_backend_metal_device;

    LM_GGML_UNUSED(reg);
    LM_GGML_UNUSED(index);
}

static struct lm_ggml_backend_feature g_lm_ggml_backend_metal_features[] = {
#if defined(LM_GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
#if defined(LM_GGML_METAL_USE_BF16)
    { "BF16", "1" },
#endif
    { nil, nil },
};

static struct lm_ggml_backend_feature * lm_ggml_backend_metal_get_features(lm_ggml_backend_reg_t reg) {
    return g_lm_ggml_backend_metal_features;

    LM_GGML_UNUSED(reg);
}

static void * lm_ggml_backend_metal_get_proc_address(lm_ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "lm_ggml_backend_get_features") == 0) {
        return (void *)lm_ggml_backend_metal_get_features;
    }

    return NULL;

    LM_GGML_UNUSED(reg);
}
static struct lm_ggml_backend_reg_i lm_ggml_backend_metal_reg_i = {
    /* .get_name         = */ lm_ggml_backend_metal_reg_get_name,
    /* .device_count     = */ lm_ggml_backend_metal_reg_device_count,
    /* .device_get       = */ lm_ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ lm_ggml_backend_metal_get_proc_address,
};

lm_ggml_backend_reg_t lm_ggml_backend_metal_reg(void) {
    // TODO: make this thread-safe somehow?
    {
        g_lm_ggml_backend_metal_reg = (struct lm_ggml_backend_reg) {
            /* .api_version = */ LM_GGML_BACKEND_API_VERSION,
            /* .iface       = */ lm_ggml_backend_metal_reg_i,
            /* .context     = */ NULL,
        };

        g_lm_ggml_backend_metal_device = (struct lm_ggml_backend_device) {
            /* .iface   = */ lm_ggml_backend_metal_device_i,
            /* .reg     = */ &g_lm_ggml_backend_metal_reg,
            /* .context = */ &g_lm_ggml_ctx_dev_main,
        };
    }

    return &g_lm_ggml_backend_metal_reg;
}

LM_GGML_BACKEND_DL_IMPL(lm_ggml_backend_metal_reg)
