// Reproduce SWA (sliding-window attention) KV-cache corruption in the LEGACY
// rn-completion `loadPrompt` reuse path. This is the bug seen on Gemma 3 / 3n
// (e.g. gemma-3n-E2B) where a chat goes garbage/incoherent after a long turn.
//
// ----------------------------------------------------------------------------
// WHY THE EXISTING S5 SCENARIO DOES NOT REPRODUCE IT
// ----------------------------------------------------------------------------
// Corruption needs THREE preconditions at once; chat_scenarios_test S5 meets none:
//   (1) the model is actually iSWA  -> tiny-random-llama is dense, n_swa == 0.
//   (2) turn-1 length EXCEEDS n_swa  -> S5's ~500-token filler < Gemma's window,
//       so the window never slides past the shared prefix and nothing is evicted.
//   (3) the test ASSERTS the eviction precondition (seq_pos_min advanced past the
//       reused prefix). Without that assert, a silent non-repro looks like a pass.
//
// ----------------------------------------------------------------------------
// THE MECHANISM
// ----------------------------------------------------------------------------
// For an iSWA model the SWA cache only physically retains the last `n_swa`
// positions: after a long turn-1 it holds [seq_pos_min, seq_pos_max], NOT [0, N).
// `find_common_prefix_length` still reports a shared prefix that reaches into the
// evicted region. Crucially, iSWA `seq_rm(0, n_past, -1)` ALWAYS returns true
// (llama-kv-cache-iswa.cpp -> AND of two unified caches that never fail), so the
// hybrid/recurrent WIPE fallback in loadPrompt NEVER fires. The path therefore
// "reuses" KV cells for positions that no longer exist in the SWA layers: the
// new tokens' SWA layers attend to a hole instead of the real prefix.
//
// The fix (server parity, server-context.cpp pos_min_thold check): before reusing
// a prefix, compare llama_memory_seq_pos_min() against the oldest position the new
// decode must attend to; if the cells were evicted, reset n_past to 0 and reprocess.
//
// ----------------------------------------------------------------------------
// OBSERVED ON gemma-4-E2B (n_swa=512): STATE BUG IS REAL, OUTPUT OFTEN SURVIVES
// ----------------------------------------------------------------------------
// Important nuance found while building this. The state-level violation is
// ALWAYS present (loadPrompt reuses cells far below the SWA-retained minimum),
// but the OUTPUT frequently stays correct. Gemma interleaves SWA layers with a
// few GLOBAL (full-attention) layers, and the global layers never evict by
// window, so they still hold [0, n_past) correctly and compensate for the SWA
// hole. The corruption therefore SHOWS only when the SWA-local contribution
// matters more than the global layers can cover -> it is data-dependent and
// intermittent. This is exactly why hunting for "garbage output" did not
// reproduce it: the reliable signal is the STATE invariant, not the text.
//
// ----------------------------------------------------------------------------
// HOW THIS TEST CAPTURES IT
// ----------------------------------------------------------------------------
// Primary signal is DETERMINISTIC and does not depend on generation quality:
//   * Construct a NEW-chat turn-2 that shares only a short early prefix S.
//   * After a long turn-1, assert seq_pos_min advanced past |S| (S is evicted
//     from the SWA cache) -> precondition proven, not assumed.
//   * Read n_past after turn-2 loadPrompt. The new tokens sit near position |S|
//     and their SWA window reaches back to position 0, so every reused cell is
//     needed AND evicted. Correct behavior => loadPrompt must reset (n_past == 0).
//     Bug => n_past > 0 (reuse of evicted cells). THIS is the reliable check.
// Secondary signal (greedy, also deterministic): the same turn-2 prompt run WITH
// clearCache vs WITHOUT. If outputs diverge, corruption is visible; if they match
// (the common case, thanks to global-layer compensation) the text is INCONCLUSIVE
// and we defer to the n_past check. We never treat a text match as proof of safety.
//
// Convention (matches the other tests): PASS = correct/fixed, FAIL = bug present.
// The n_past check FAILS today (bug reproduced) and flips to PASS once loadPrompt
// gains the SWA seq_pos_min guard.
//
// Run:  ./swa_cache_corruption_test <gemma3-swa-model.gguf>   (env GPU_LAYERS=99)

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdlib>

#include "rn-llama.h"
#include "rn-completion.h"
#include "common.h"

using namespace rnllama;

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

// Fraction of the LONGER string's leading chars that match. 1.0 == identical
// prefix; collapses fast when corruption flips early tokens.
static double prefix_similarity(const std::string& a, const std::string& b) {
    size_t n = std::min(a.size(), b.size()), i = 0;
    while (i < n && a[i] == b[i]) i++;
    size_t denom = std::max(a.size(), b.size());
    return denom == 0 ? 1.0 : (double)i / (double)denom;
}

static bool looks_like_garbage(const std::string& s) {
    if (s.empty()) return true;
    int printable = 0;
    for (unsigned char c : s) if (std::isprint(c) || c == '\n' || (c & 0x80)) printable++;
    if ((double)printable / s.size() < 0.70) return true;
    std::vector<std::string> words; std::string w;
    for (char c : s) { if (std::isspace((unsigned char)c)) { if(!w.empty()){words.push_back(lower(w)); w.clear();} } else w += c; }
    if (!w.empty()) words.push_back(lower(w));
    if (words.size() >= 8) {
        size_t best = 0;
        for (auto& a : words) { size_t cnt = std::count(words.begin(), words.end(), a); best = std::max(best, cnt); }
        if ((double)best / words.size() > 0.60) return true;
    }
    return false;
}

// One completion turn through the real legacy flow (params.prompt path, like
// reuse_test). Returns generated text; out_npast = n_past right after loadPrompt.
static std::string run_turn(llama_rn_context& ctx, const std::string& prompt,
                            int n_predict, int& out_npast, const char* label) {
    ctx.params.prompt = prompt;
    ctx.params.n_predict = n_predict;
    auto* c = ctx.completion;
    c->rewind();
    c->initSampling();
    c->beginCompletion();
    c->loadPrompt({});
    out_npast = (int)c->n_past;
    int gen = 0;
    while (c->has_next_token && gen < n_predict) {
        auto t = c->doCompletion();
        if (t.tok == -1) break;
        gen++;
    }
    std::string text = c->generated_text;
    c->endCompletion();
    std::cout << "  [" << label << "] n_past(reused)=" << out_npast
              << " gen=" << gen << " resp=\"" << text.substr(0, 70) << "\"\n";
    return text;
}

static int g_pass = 0, g_fail = 0;
static void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::cout << (ok ? "  PASS " : "  FAIL ") << name;
    if (!detail.empty()) std::cout << "  [" << detail << "]";
    std::cout << "\n";
    ok ? g_pass++ : g_fail++;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: swa_cache_corruption_test <gemma3-swa-model.gguf>\n"
                     "  Needs a real iSWA model (Gemma 3 / 3n). A dense model is skipped.\n";
        return 2;
    }
    const std::string model = argv[1];
    if (!std::filesystem::exists(model)) { std::cout << "missing model: " << model << "\n"; return 2; }

    // NOTE: the test harness builds with Metal DISABLED (see tests/CMakeLists.txt),
    // so there is no GPU backend here -> default to CPU. Turn-1 is a single batched
    // prompt ingestion (only ~32 tokens are generated), so CPU is fine.
    const char* gl = getenv("GPU_LAYERS");
    const int gpu_layers = gl ? atoi(gl) : 0;

    std::cout << "===== SWA cache corruption repro :: " << model << " =====\n";

    llama_rn_context ctx;
    common_params params;
    params.model.path = model;
    params.n_ctx = 8192;            // must exceed turn-1 length so ctx_shift never fires
    params.n_batch = 512;
    params.cpuparams.n_threads = 6;
    params.n_gpu_layers = gpu_layers;
    params.sampling.temp = 0.0f;    // greedy -> turn-2 with/without reset is deterministic
    if (!ctx.loadModel(params)) { std::cout << "load failed\n"; return 2; }

    const llama_model* lmodel = llama_get_model(ctx.ctx);
    const int n_swa = llama_model_n_swa(lmodel);
    std::cout << "n_swa = " << n_swa << "  (0 == dense, no sliding window)\n";
    if (n_swa <= 0) {
        std::cout << "SKIP: model is not SWA, this bug cannot occur here. "
                     "Run with a Gemma 3 / 3n model to reproduce.\n";
        return 0;
    }

    // Shared early prefix S, carrying a secret only recallable if its KV survives.
    const std::string SECRET = "ZX9QW";
    const std::string S =
        "System note: the secret access code is " + SECRET +
        ". Always remember this code for later. ";

    // Build turn-1 filler so the tokenized turn-1 prompt exceeds n_swa by ~3x,
    // guaranteeing the window slides well past S and evicts it from the SWA cache.
    const size_t target_tokens = (size_t)n_swa * 3 + 256;
    std::string filler;
    const std::string sentence =
        "The river carried small round stones past the old wooden bridge while the wind moved through the tall dry grass. ";
    auto tok_len = [&](const std::string& s) {
        return common_tokenize(ctx.ctx, s, true, true).size();
    };
    while (tok_len(S + filler) < target_tokens) filler += sentence;

    // Wrap in Gemma turn markers so generation actually flows (a raw prompt under
    // greedy hits <end_of_turn> immediately and yields 0 tokens — on the clean
    // reference too — which would make the behavioral checks meaningless). Both
    // prompts share the leading "<start_of_turn>user\n" + S, so the reused prefix
    // is real.
    const std::string U = "<start_of_turn>user\n";
    const std::string M = "<end_of_turn>\n<start_of_turn>model\n";
    const std::string p1 = U + S + filler + "\nSummarize the passage above in one sentence." + M;
    const std::string p2 = U + S + "Ignore the passage. In one short sentence, what is the secret access code?" + M;

    const size_t s_tokens = tok_len(U + S);   // the shared prefix p1/p2 have in common
    std::cout << "shared-prefix tokens ~= " << s_tokens
              << "  turn-1 prompt tokens ~= " << tok_len(p1) << "\n";

    // ---- Turn 1: long, cold. Populates + slides the SWA window. ----
    std::cout << "\n-- turn 1 (long, cold) --\n";
    int np1 = 0;
    run_turn(ctx, p1, 8, np1, "turn1");

    auto* kv = llama_get_memory(ctx.ctx);
    const llama_pos pos_min = llama_memory_seq_pos_min(kv, 0);
    const llama_pos pos_max = llama_memory_seq_pos_max(kv, 0);
    std::cout << "  after turn1: SWA seq_pos_min=" << pos_min
              << " seq_pos_max=" << pos_max << "\n";

    // Precondition: the shared prefix S must have been EVICTED from the SWA cache.
    // (seq_pos_min reports the SWA cache min for iSWA models.)
    const bool evicted = pos_min > (llama_pos)s_tokens;
    check("PRECONDITION: SWA window evicted the shared prefix (seq_pos_min > |S|)",
          evicted,
          "pos_min=" + std::to_string(pos_min) + " |S|=" + std::to_string(s_tokens));
    if (!evicted) {
        std::cout << "INCONCLUSIVE: window did not slide past S. Increase target_tokens "
                     "(needs turn-1 length > n_swa) and rerun.\n";
        std::cout << "\n===== " << g_pass << " passed, " << g_fail << " failed =====\n";
        return 2;
    }

    // ---- Turn 2 (BUGGY path): NEW chat, NO clearCache. ----
    // Shares only S with the cache; new tokens land near position |S| and their
    // SWA window reaches back to position 0 -> every reused cell is needed AND
    // evicted. Correct behavior: loadPrompt resets reuse to n_past == 0.
    std::cout << "\n-- turn 2a: new chat, NO clearCache (reuses evicted prefix) --\n";
    int np2 = 0;
    std::string b_noreset = run_turn(ctx, p2, 96, np2, "turn2 no-reset");

    // PRIMARY (deterministic) regression assertion. FAILS today = bug reproduced.
    check("loadPrompt must NOT reuse an SWA-evicted prefix (expect n_past==0)",
          np2 == 0,
          "n_past=" + std::to_string(np2) + " but SWA min retained pos=" + std::to_string(pos_min));

    // ---- Turn 2 (REFERENCE): same prompt, WITH clearCache -> fresh reprocess. ----
    std::cout << "\n-- turn 2b: same prompt, WITH clearCache (clean reference) --\n";
    ctx.clearCache(false);
    int np2c = 0;
    std::string b_reset = run_turn(ctx, p2, 96, np2c, "turn2 reset");

    // SECONDARY (deterministic, greedy): the clean reference must actually
    // generate, otherwise the comparison is vacuous. Only assert when it does.
    auto nonempty = [](const std::string& s){
        return s.find_first_not_of(" \t\r\n") != std::string::npos;
    };
    if (!nonempty(b_reset)) {
        std::cout << "  [behavioral INCONCLUSIVE] clean reference generated nothing "
                     "(format/sampling) — rely on the deterministic n_past check above.\n";
    } else {
        // Correct reuse would yield the same greedy tokens as a clean reprocess;
        // a corrupted SWA prefix diverges sharply.
        const double sim = prefix_similarity(b_noreset, b_reset);
        std::cout << "  prefix_similarity(no-reset, reset) = " << sim << "\n";
        check("SWA reuse output matches clean reprocess (greedy)", sim > 0.80,
              "similarity=" + std::to_string(sim));
        std::cout << "  [info] no-reset garbage? " << (looks_like_garbage(b_noreset) ? "yes" : "no")
                  << "   clean recalls secret? "
                  << (lower(b_reset).find(lower(SECRET)) != std::string::npos ? "yes" : "no") << "\n";
    }

    std::cout << "\n===== " << g_pass << " passed, " << g_fail << " failed =====\n";
    std::cout << "(FAIL today = bug reproduced; checks flip to PASS once loadPrompt\n"
                 " gains the seq_pos_min SWA guard.)\n";
    return g_fail == 0 ? 0 : 1;
}
