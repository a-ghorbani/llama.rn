#include "ggml.h"
#include "gguf.h"
#include "clip.h"

#include <climits>
#include <cstdarg>
#include <cinttypes>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <memory>

// Internal header for clip.cpp

#define KEY_FTYPE               "general.file_type"
#define KEY_NAME                "general.name"
#define KEY_DESCRIPTION         "general.description"
#define KEY_PROJ_TYPE           "clip.projector_type"
#define KEY_HAS_AUDIO_ENC       "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC      "clip.has_vision_encoder"
#define KEY_USE_GELU            "clip.use_gelu"
#define KEY_USE_SILU            "clip.use_silu"

#define KEY_N_EMBD              "clip.%s.embedding_length"
#define KEY_N_FF                "clip.%s.feed_forward_length"
#define KEY_N_BLOCK             "clip.%s.block_count"
#define KEY_PROJ_DIM            "clip.%s.projection_dim"
#define KEY_N_HEAD              "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS      "clip.%s.attention.layer_norm_epsilon"

// vision-specific
#define KEY_IMAGE_SIZE          "clip.vision.image_size"
#define KEY_PATCH_SIZE          "clip.vision.patch_size"
#define KEY_IMAGE_MEAN          "clip.vision.image_mean"
#define KEY_IMAGE_STD           "clip.vision.image_std"
#define KEY_FEATURE_LAYER       "clip.vision.feature_layer"
#define KEY_PROJ_SCALE_FACTOR   "clip.vision.projector.scale_factor"
#define KEY_SPATIAL_MERGE_SIZE  "clip.vision.spatial_merge_size"

#define KEY_MM_PATCH_MERGE_TYPE   "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS  "clip.vision.image_grid_pinpoints"
#define KEY_IMAGE_CROP_RESOLUTION "clip.vision.image_crop_resolution"
#define KEY_WIN_ATTN_PATTERN      "clip.vision.n_wa_pattern"
#define KEY_ATTN_WINDOW_SIZE      "clip.vision.window_size"
#define KEY_MINICPMV_VERSION      "clip.minicpmv_version"

// audio-specific
#define KEY_A_NUM_MEL_BINS      "clip.audio.num_mel_bins"
#define KEY_A_PROJ_STACK_FACTOR "clip.audio.projector.stack_factor"


//
// tensor name constants
//

#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_CLASS_EMBD      "v.class_embd"
#define TN_PATCH_EMBD      "v.patch_embd.weight"  // not rename tensor with ".0" postfix for backwrad compat
#define TN_PATCH_EMBD_1    "v.patch_embd.weight.1"
#define TN_PATCH_BIAS      "v.patch_embd.bias"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_ATTN_K_NORM     "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM     "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s" // layer norm
#define TN_LN_2            "%s.blk.%d.ln2.%s" // layer norm
#define TN_LS_1            "%s.blk.%d.ls1.%s" // layer scale
#define TN_LS_2            "%s.blk.%d.ls2.%s" // layer scale
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_LLAVA_PROJ      "mm.%d.%s"
#define TN_MVLM_PROJ_MLP   "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG   "mm.model.peg.%d.%s"
#define TN_IMAGE_NEWLINE   "model.image_newline"
#define TN_MM_INP_NORM     "mm.input_norm.weight"
#define TN_MM_INP_PROJ     "mm.input_projection.weight" // gemma3
#define TN_MM_SOFT_EMB_N   "mm.soft_emb_norm.weight"    // gemma3
#define TN_MM_PROJECTOR    "mm.model.fc.weight"         // idefics3
#define TN_MM_PATCH_MERGER "mm.patch_merger.weight"     // mistral small 3.1
#define TN_TOK_IMG_BREAK   "v.token_embd.img_break"     // pixtral
#define TN_TOK_GLM_BOI     "adapter.boi"                // glm-edge (these embeddings are not in text model)
#define TN_TOK_GLM_EOI     "adapter.eoi"                // glm-edge (these embeddings are not in text model)

// mimicpmv
#define TN_MINICPMV_POS_EMBD_K "resampler.pos_embed_k"
#define TN_MINICPMV_QUERY      "resampler.query"
#define TN_MINICPMV_PROJ       "resampler.proj.weight"
#define TN_MINICPMV_KV_PROJ    "resampler.kv.weight"
#define TN_MINICPMV_ATTN       "resampler.attn.%s.%s"
#define TN_MINICPMV_LN         "resampler.ln_%s.%s"

#define TN_GLM_ADAPER_CONV      "adapter.conv.%s"
#define TN_GLM_ADAPTER_LINEAR   "adapter.linear.linear.%s"
#define TN_GLM_ADAPTER_NORM_1   "adapter.linear.norm1.%s"
#define TN_GLM_ADAPTER_D_H_2_4H "adapter.linear.dense_h_to_4h.%s"
#define TN_GLM_ADAPTER_GATE     "adapter.linear.gate.%s"
#define TN_GLM_ADAPTER_D_4H_2_H "adapter.linear.dense_4h_to_h.%s"

// ultravox
#define TN_CONV1D       "a.conv1d.%d.%s"
#define TN_MM_AUDIO_MLP "mm.a.mlp.%d.%s"
#define TN_MM_AUDIO_FC  "mm.a.fc.%s" // fully connected layer
#define TN_MM_NORM_PRE  "mm.a.norm_pre.%s"
#define TN_MM_NORM_MID  "mm.a.norm_mid.%s"

// align x to upper multiple of n
#define CLIP_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_MINICPMV,
    PROJECTOR_TYPE_GLM_EDGE,
    PROJECTOR_TYPE_QWEN2VL,
    PROJECTOR_TYPE_GEMMA3,
    PROJECTOR_TYPE_IDEFICS3,
    PROJECTOR_TYPE_PIXTRAL,
    PROJECTOR_TYPE_QWEN25VL,
    PROJECTOR_TYPE_ULTRAVOX,
    PROJECTOR_TYPE_INTERNVL,
    PROJECTOR_TYPE_LLAMA4,
    PROJECTOR_TYPE_QWEN2A,
    PROJECTOR_TYPE_QWEN25O, // will be replaced by QWEN2A or QWEN25VL depending on clip_ctx
    PROJECTOR_TYPE_VOXTRAL,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_MLP,       "mlp" },
    { PROJECTOR_TYPE_LDP,       "ldp" },
    { PROJECTOR_TYPE_LDPV2,     "ldpv2"},
    { PROJECTOR_TYPE_MINICPMV,  "resampler"},
    { PROJECTOR_TYPE_GLM_EDGE,  "adapter"},
    { PROJECTOR_TYPE_QWEN2VL,   "qwen2vl_merger"},
    { PROJECTOR_TYPE_QWEN25VL,  "qwen2.5vl_merger"},
    { PROJECTOR_TYPE_GEMMA3,    "gemma3"},
    { PROJECTOR_TYPE_IDEFICS3,  "idefics3"},
    { PROJECTOR_TYPE_PIXTRAL,   "pixtral"},
    { PROJECTOR_TYPE_ULTRAVOX,  "ultravox"},
    { PROJECTOR_TYPE_INTERNVL,  "internvl"},
    { PROJECTOR_TYPE_LLAMA4,    "llama4"},
    { PROJECTOR_TYPE_QWEN2A,    "qwen2a"},
    { PROJECTOR_TYPE_QWEN25O,   "qwen2.5o"},
    { PROJECTOR_TYPE_VOXTRAL,   "voxtral"},
};

static projector_type clip_projector_type_from_string(const std::string & str) {
    for (const auto & pair : PROJECTOR_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return PROJECTOR_TYPE_UNKNOWN;
}

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

//
// logging
//

static void clip_log_callback_default(enum lm_ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct clip_logger_state {
    lm_ggml_log_level verbosity_thold;
    lm_ggml_log_callback log_callback;
    void * log_callback_user_data;
};

extern struct clip_logger_state g_logger_state;

static void clip_log_internal_v(enum lm_ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

static void clip_log_internal(enum lm_ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    clip_log_internal_v(level, format, args);
    va_end(args);
}

#define LOG_TMPL(level, ...) \
    do { \
        if ((level) >= g_logger_state.verbosity_thold) { \
            clip_log_internal((level), __VA_ARGS__); \
        } \
    } while (0)
#define LOG_INF(...) LOG_TMPL(LM_GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(LM_GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(LM_GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_DBG(...) LOG_TMPL(LM_GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(LM_GGML_LOG_LEVEL_CONT,  __VA_ARGS__)

//
// cpp wrappers
//

// wrapper for clip_image_size
struct clip_image_size_deleter {
    void operator()(clip_image_size * val) { clip_image_size_free(val); }
};
typedef std::unique_ptr<clip_image_size, clip_image_size_deleter> clip_image_size_ptr;

// wrapper for clip_image_u8
struct clip_image_u8_deleter {
    void operator()(clip_image_u8 * val) { clip_image_u8_free(val); }
};
typedef std::unique_ptr<clip_image_u8, clip_image_u8_deleter> clip_image_u8_ptr;

// wrapper for clip_image_f32
struct clip_image_f32_deleter {
    void operator()(clip_image_f32 * val) { clip_image_f32_free(val); }
};
typedef std::unique_ptr<clip_image_f32, clip_image_f32_deleter> clip_image_f32_ptr;

struct clip_image_u8_batch {
    std::vector<clip_image_u8_ptr> entries;
};

struct clip_image_f32_batch {
    std::vector<clip_image_f32_ptr> entries;
    bool is_audio = false;

    // for llava-uhd style models, we need to know the grid size
    // note: entries.size() == grid_x * grid_y + 1 (one overview image)
    int grid_x = 0;
    int grid_y = 0;

    clip_image_f32_batch clone() const {
        clip_image_f32_batch new_batch{
            /* entries  */ {},
            /* is_audio */ is_audio,
            /* grid_x   */ grid_x,
            /* grid_y   */ grid_y,
        };
        new_batch.entries.reserve(entries.size());
        for (const auto & entry : entries) {
            new_batch.entries.emplace_back(new clip_image_f32(*entry));
        }
        return new_batch;
    }
};

//
// common utils
//

static std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    LM_GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    LM_GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

// split string by a `std::string delim` instead of `char delim`
static std::vector<std::string> string_split_str(std::string s, const std::string & delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);
    return tokens;
}

//
// gguf utils
//

static std::string lm_gguf_data_to_str(enum lm_gguf_type type, const void * data, int i) {
    switch (type) {
        case LM_GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case LM_GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case LM_GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case LM_GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case LM_GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case LM_GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case LM_GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case LM_GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case LM_GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case LM_GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case LM_GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return string_format("unknown type %d", type);
    }
}

static std::string lm_gguf_kv_to_str(const struct lm_gguf_context * ctx_gguf, int i) {
    const enum lm_gguf_type type = lm_gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case LM_GGUF_TYPE_STRING:
            return lm_gguf_get_val_str(ctx_gguf, i);
        case LM_GGUF_TYPE_ARRAY:
            {
                const enum lm_gguf_type arr_type = lm_gguf_get_arr_type(ctx_gguf, i);
                int arr_n = lm_gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == LM_GGUF_TYPE_STRING ? nullptr : lm_gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == LM_GGUF_TYPE_STRING) {
                        std::string val = lm_gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        string_replace_all(val, "\\", "\\\\");
                        string_replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == LM_GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << lm_gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return lm_gguf_data_to_str(type, lm_gguf_get_val_data(ctx_gguf, i), 0);
    }
}

//
// debugging
//

static void print_tensor_shape(lm_ggml_tensor * t) {
    printf("%s.shape = [", t->name);
    for (int i = 0; i < lm_ggml_n_dims(t); ++i) {
        printf("%" PRId64, t->ne[i]);
        if (i < lm_ggml_n_dims(t) - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

static void print_tensor_data(lm_ggml_tensor * t, uint8_t * data, int64_t n) {
    lm_ggml_type type = t->type;
    int64_t * ne = t->ne;
    size_t * nb = t->nb;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("%s.data: [\n", t->name);
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("     ..., \n");
                i2 = ne[2] - n;
            }
            printf("     [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("      ..., \n");
                    i1 = ne[1] - n;
                }
                printf("      [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == LM_GGML_TYPE_F16) {
                        v = lm_ggml_fp16_to_fp32(*(lm_ggml_fp16_t *) &data[i]);
                    } else if (type == LM_GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == LM_GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == LM_GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == LM_GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        LM_GGML_ABORT("fatal error");
                    }
                    printf("%8.4f", v);
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("     ],\n");
        }
        printf("    ]\n");
    }
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx);
