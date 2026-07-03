#ifndef RN_KV_CHECKPOINT_HPP
#define RN_KV_CHECKPOINT_HPP

#include "llama.h"

#include <cstdint>
#include <vector>

namespace rnllama {

// In-memory KV checkpoint for the legacy completion path.
//
// A single checkpoint pinned at the latest prompt/turn boundary (captured at the end
// of prefill, before generation) lets recurrent/hybrid models reuse the cache across
// turns instead of wiping it when a mid-sequence seq_rm fails. It serializes the full
// per-sequence state (attention + recurrent R/S tensors) via llama_state_seq_get_data_ext
// / set_data_ext. SWA models are not handled: their seq_rm returns true even after the
// window evicted cells, so the wipe-and-restore path is never reached there.
//
// It stores:
//   - the serialized state blob,
//   - boundary_pos: the number of tokens captured, i.e. the cache covers [0, boundary_pos),
//   - boundary_tokens: those tokens, so a restore can validate that the new prompt
//     extends this prefix before trusting the snapshot.
//
// Re-captured at each turn's prompt boundary only; it does not roll forward during decode.
struct KvCheckpoint {
    bool valid = false;
    llama_pos boundary_pos = 0;
    std::vector<llama_token> boundary_tokens;
    std::vector<uint8_t> data;

    void invalidate() {
        valid = false;
        boundary_pos = 0;
        // Swap-clear so the memory is actually released; clear() keeps the vector
        // capacity, which for `data` can be a ~54-114 MB blob on hybrid-SSM models.
        std::vector<llama_token>().swap(boundary_tokens);
        std::vector<uint8_t>().swap(data);
    }

    // Snapshot the state of `seq_id`, which must currently cover exactly
    // tokens[0, n_boundary). Host-side snapshot (LLAMA_STATE_SEQ_FLAGS_NONE).
    //
    // `backend_serializable` is the caller's verdict on whether this backend can
    // round-trip per-seq state (OpenCL with kv_unified==false or flash-attn enabled
    // cannot). When false we refuse rather than capture a bad/partial blob, and
    // invalidate any prior snapshot.
    bool save(llama_context *ctx, llama_seq_id seq_id,
              const std::vector<llama_token> &tokens, llama_pos n_boundary,
              bool backend_serializable = true) {
        if (!backend_serializable) {
            invalidate();
            return false;
        }
        if (ctx == nullptr || n_boundary <= 0 || (size_t)n_boundary > tokens.size()) {
            return false;
        }
        const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE;
        const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
        if (size == 0) {
            invalidate();
            return false;
        }
        data.resize(size);
        const size_t n = llama_state_seq_get_data_ext(ctx, data.data(), size, seq_id, flags);
        if (n != size) {
            invalidate();
            return false;
        }
        boundary_pos = n_boundary;
        boundary_tokens.assign(tokens.begin(), tokens.begin() + n_boundary);
        valid = true;
        return true;
    }

    // True if the captured boundary tokens are a prefix of `prompt`.
    bool is_prefix_of(const std::vector<llama_token> &prompt) const {
        if (!valid || (size_t)boundary_pos > prompt.size()) {
            return false;
        }
        for (size_t i = 0; i < (size_t)boundary_pos; i++) {
            if (boundary_tokens[i] != prompt[i]) {
                return false;
            }
        }
        return true;
    }

    // Restore the snapshot into `seq_id`. llama_state_seq_set_data first removes the
    // whole destination sequence, then writes back exactly [0, boundary_pos), so no
    // stale cells survive past the boundary. Returns true on success.
    bool restore(llama_context *ctx, llama_seq_id seq_id) const {
        if (!valid || ctx == nullptr || data.empty()) {
            return false;
        }
        const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE;
        const size_t n = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), seq_id, flags);
        return n == data.size();
    }
};

} // namespace rnllama

#endif /* RN_KV_CHECKPOINT_HPP */
