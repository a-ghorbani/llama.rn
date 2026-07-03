// KV-cache reuse + KV-checkpoint tests for the legacy rn-completion path.
//
// Three layers, all driving the real JSI sequence (rewind -> initSampling ->
// beginCompletion -> loadPrompt -> doCompletion loop -> endCompletion):
//
//   1. UNIT SECTION (always runs, gates the exit code). KvCheckpoint invariants
//      (is_prefix_of, save() arg-check asymmetry, backend refusal, invalidate()
//      really frees) plus the completion-level cancel/drop post-states and the
//      endCompletion undecoded-tail-trim regression. Uses the bundled tiny
//      model (../tiny-random-llama.gguf) for the parts that need a live context.
//
//   2. GATED VALIDATION (argv[1] = a real hybrid/SSM model, e.g. LFM2 /
//      Qwen3.5). The clearCache leak gate, the perf/correctness phase-B gates,
//      and the new restore-failure / ctx-shift / embedding-interleave tests.
//      On a pure-attention model these skip loudly instead of passing.
//
//   3. DESCRIPTIVE SCENARIOS (no-arg runs, observability only): does a clean
//      extension reuse, does mid-divergence wipe, does a faithful round-trip
//      reuse. These print, they do not assert.
//
// Extra mode: ./reuse_test <model> size   -> checkpoint footprint per n_ctx.

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>

#include "rn-llama.h"
#include "rn-completion.h"
#include "common.h"

#include "test-helpers.hpp"

using namespace rnllama;
using namespace rntest;

static const char* TINY_MODEL = "../tiny-random-llama.gguf";

// Default tiny model; overridden with argv[1] to test a real architecture.
static const char* MODEL = TINY_MODEL;

// ---------------------------------------------------------------- ctx setup

static bool load_chat_ctx(llama_rn_context& ctx, const char* model_path,
                          int n_ctx = 2048, int n_batch = 512) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.cpuparams.n_threads = 6;
    params.n_gpu_layers = 0;
    params.sampling.temp = 0.0f;   // greedy -> deterministic token comparison
    params.sampling.top_k = 1;
    return ctx.loadModel(params);
}

// One chat turn through the real flow (renders msgs with the chat template).
static TurnResult chat_turn(llama_rn_context& ctx, const std::vector<Msg>& msgs,
                            int n_predict, bool checkpoint) {
    TurnOptions o;
    o.n_predict = n_predict;
    o.kv_checkpoint = checkpoint;
    return run_turn(ctx, format_chat(ctx, msgs), o);
}

// ============================================================================
// 1. UNIT SECTION -- KvCheckpoint + completion checkpoint-state invariants.
//    Always runs and always gates the exit code.
// ============================================================================

static void unit_is_prefix_of() {
    std::cout << "\n-- unit: is_prefix_of matrix --\n";
    KvCheckpoint kv;
    kv.valid = true;
    kv.boundary_pos = 4;
    kv.boundary_tokens = {1, 2, 3, 4};

    check("is_prefix_of: shorter prompt {1,2,3} -> false", !kv.is_prefix_of({1, 2, 3}));
    check("is_prefix_of: exact prompt {1,2,3,4} -> true", kv.is_prefix_of({1, 2, 3, 4}));
    check("is_prefix_of: extension {1,2,3,4,5} -> true", kv.is_prefix_of({1, 2, 3, 4, 5}));
    check("is_prefix_of: first-token mismatch {9,..} -> false", !kv.is_prefix_of({9, 2, 3, 4, 5}));
    check("is_prefix_of: mismatch at boundary-1 -> false", !kv.is_prefix_of({1, 2, 3, 9, 5}));
    kv.valid = false;
    check("is_prefix_of: valid=false -> false", !kv.is_prefix_of({1, 2, 3, 4, 5}));
}

static void unit_invalidate_frees_manual() {
    std::cout << "\n-- unit: invalidate() releases memory (manual blob) --\n";
    KvCheckpoint kv;
    kv.valid = true;
    kv.boundary_pos = 1024;
    kv.boundary_tokens.assign(1024, 7);
    kv.data.assign(4u << 20, 0xAB);   // 4 MB stand-in for a real hybrid-SSM blob
    kv.invalidate();
    check("invalidate: valid=false, boundary_pos=0", !kv.valid && kv.boundary_pos == 0);
    check("invalidate: data capacity == 0 (swap-cleared)", kv.data.capacity() == 0,
          "capacity=" + std::to_string(kv.data.capacity()));
    check("invalidate: boundary_tokens capacity == 0 (swap-cleared)",
          kv.boundary_tokens.capacity() == 0,
          "capacity=" + std::to_string(kv.boundary_tokens.capacity()));
}

// Model-backed unit checks: undecoded-tail trim, save() semantics, drop/cancel.
static void unit_with_model(const char* model_path) {
    std::cout << "\n-- unit: model-backed checks (" << model_path << ") --\n";
    llama_rn_context ctx;
    if (!load_chat_ctx(ctx, model_path, 512, 128)) {
        check("unit: test model loads", false, model_path);
        return;
    }
    auto* c = ctx.completion;

    // ---- B3 regression: endCompletion must trim the never-decoded tail. The
    // final sampled token is pushed to embd but only decoded by the NEXT
    // nextToken call; on a stop it must not survive in embd, or next-turn prefix
    // reuse skips its decode (KV hole / missing recurrent update). Before the
    // fix, seq_pos_max+1 == embd.size()-1 here. ----
    TurnOptions o;
    o.n_predict = 3;   // greedy (temp=0 from load_chat_ctx)
    auto t = run_turn(ctx, "alpha beta gamma delta epsilon", o);
    auto* mem = llama_get_memory(ctx.ctx);
    const llama_pos cache_end = llama_memory_seq_pos_max(mem, 0) + 1;
    check("endCompletion trims undecoded tail (seq_pos_max+1 == embd.size)",
          cache_end == (llama_pos)c->embd.size(),
          "cache_end=" + std::to_string(cache_end) +
          " embd=" + std::to_string(c->embd.size()) +
          " predicted=" + std::to_string(t.predicted));

    // ---- save() success on a live sequence ----
    std::cout << "\n-- unit: save() arg-check asymmetry --\n";
    KvCheckpoint kv;
    const std::vector<llama_token> toks = c->embd;   // cache covers exactly [0, n_past)
    const llama_pos n = c->n_past;
    if (n <= 0) {
        check("unit: live sequence available for save()", false);
        return;
    }
    check("save: succeeds on live seq", kv.save(ctx.ctx, 0, toks, n, true) && kv.valid,
          "n_boundary=" + std::to_string(n) + " blob=" + std::to_string(kv.data.size()) + "B");

    // Arg-check failures must return false WITHOUT invalidating the existing
    // snapshot. This asymmetry is deliberate and load-bearing: a bad call site
    // must not cost the conversation its only restore point.
    const size_t blob_size = kv.data.size();
    const std::vector<llama_token> btoks = kv.boundary_tokens;
    check("save(ctx=nullptr): false, snapshot kept",
          !kv.save(nullptr, 0, toks, n, true) && kv.valid && kv.data.size() == blob_size);
    check("save(n_boundary=0): false, snapshot kept",
          !kv.save(ctx.ctx, 0, toks, 0, true) && kv.valid && kv.data.size() == blob_size);
    check("save(n_boundary>tokens.size): false, snapshot kept",
          !kv.save(ctx.ctx, 0, toks, (llama_pos)toks.size() + 1, true) &&
          kv.valid && kv.data.size() == blob_size);
    check("save arg-check failures left boundary tokens untouched",
          kv.boundary_tokens == btoks && kv.boundary_pos == n);

    // ---- backend refusal is the OPPOSITE asymmetry: it must invalidate. A
    // stale blob surviving a "this backend can't serialize" verdict would later
    // restore wrong state (the OpenCL kv_unified/flash-attn case). ----
    std::cout << "\n-- unit: backend refusal invalidates --\n";
    check("save(backend_serializable=false): false AND prior snapshot invalidated",
          !kv.save(ctx.ctx, 0, toks, n, false) && !kv.valid &&
          kv.data.capacity() == 0 && kv.boundary_tokens.capacity() == 0);

    // ---- invalidate() after a REAL save must also free ----
    std::cout << "\n-- unit: invalidate() releases memory (real blob) --\n";
    KvCheckpoint kv2;
    if (kv2.save(ctx.ctx, 0, toks, n, true)) {
        kv2.invalidate();
        check("invalidate after real save: capacities == 0",
              !kv2.valid && kv2.data.capacity() == 0 && kv2.boundary_tokens.capacity() == 0);
    } else {
        check("unit: second save for invalidate test", false);
    }

    // ---- cancelKvCheckpointSave / dropKvCheckpoint exact post-states ----
    std::cout << "\n-- unit: cancelKvCheckpointSave / dropKvCheckpoint post-states --\n";
    auto arm = [&]() {
        c->kv_checkpoint.valid = true;
        c->kv_checkpoint.boundary_pos = 3;
        c->kv_checkpoint.boundary_tokens = {1, 2, 3};
        c->kv_checkpoint.data.assign(1024, 0x5A);
        c->kv_checkpoint_save_pending = true;
        c->kv_checkpoint_pending_pos = 7;
        c->kv_checkpoint_pending_tokens = {1, 2, 3, 4, 5, 6, 7};
    };

    arm();
    c->cancelKvCheckpointSave();
    check("cancel: pending capture cleared+freed",
          !c->kv_checkpoint_save_pending && c->kv_checkpoint_pending_pos == 0 &&
          c->kv_checkpoint_pending_tokens.capacity() == 0);
    check("cancel: stored snapshot KEPT",
          c->kv_checkpoint.valid && c->kv_checkpoint.boundary_pos == 3 &&
          c->kv_checkpoint.data.size() == 1024 &&
          c->kv_checkpoint.boundary_tokens.size() == 3);

    arm();
    c->dropKvCheckpoint();
    check("drop: snapshot cleared+freed",
          !c->kv_checkpoint.valid && c->kv_checkpoint.boundary_pos == 0 &&
          c->kv_checkpoint.data.capacity() == 0 &&
          c->kv_checkpoint.boundary_tokens.capacity() == 0);
    check("drop: pending capture cleared+freed",
          !c->kv_checkpoint_save_pending && c->kv_checkpoint_pending_pos == 0 &&
          c->kv_checkpoint_pending_tokens.capacity() == 0);
}

static void run_unit_section(const char* fallback_model) {
    std::cout << "\n========== UNIT SECTION (always gates exit code) ==========\n";
    unit_is_prefix_of();
    unit_invalidate_frees_manual();

    const char* unit_model = nullptr;
    if (std::filesystem::exists(TINY_MODEL)) unit_model = TINY_MODEL;
    else if (fallback_model && std::filesystem::exists(fallback_model)) unit_model = fallback_model;

    if (unit_model) {
        unit_with_model(unit_model);
    } else {
        check("unit: a model is available for model-backed unit tests", false,
              std::string("missing ") + TINY_MODEL);
    }
}

// ============================================================================
// 2. GATED VALIDATION (real hybrid/SSM model via argv[1])
// ----------------------------------------------------------------------------
// On a hybrid/recurrent model the legacy path wipes the cache on turn 2 when
// the re-rendered prompt diverges mid-sequence (e.g. Qwen3.5 strips the model's
// empty <think></think>). kv_checkpoint='memory' restores the prompt-boundary
// checkpoint instead of wiping. Validated below:
//   (1) PERF       : turn-2 reused > wiped baseline (restore actually fired).
//   (2) CORRECTNESS: restore output is coherent and deterministic.
//   plus leak / restore-failure / ctx-shift / embedding-interleave gates.
// Skips (does not pass) on a pure-attention model, which never hits the wipe path.
// ============================================================================

// Drive a two-turn chat in a fresh context (no state shared with any other pass)
// and return turn 2, with the checkpoint off or on. A fresh context per pass is
// the only way to get a deterministic baseline on SWA models, where clearCache
// does not fully reset the sliding-window state.
static TurnResult phaseb_two_turns(const char* model_path, bool checkpoint,
                                   int gen1, int gen2, size_t& out_cache1) {
    llama_rn_context ctx;
    if (!load_chat_ctx(ctx, model_path)) {
        std::cout << "load failed\n";
        TurnResult r; r.reused = -1; return r;
    }

    const std::vector<Msg> base = {{"user", "Hi, my name is Ali. Tell me a short greeting."}};
    auto t1 = chat_turn(ctx, base, gen1, checkpoint);
    out_cache1 = ctx.completion->embd.size();
    std::vector<Msg> conv = base;
    conv.push_back({"assistant", t1.text});
    conv.push_back({"user", "Write three sentences about why the ocean is blue."});
    return chat_turn(ctx, conv, gen2, checkpoint);
}

// Prove the checkpoint survives a turn that overshoots the trimmed boundary. A
// "regenerate" turn (turn 2 re-sends the same single message that is already fully
// cached) enters prefill with n_past already >= the trimmed boundary, so the batch cap
// never fires and the decode loop rolls n_past to the full prompt, including the
// trailing generation scaffold. Saving that un-trimmed boundary would make next turn's
// is_prefix_of fail -> restore rejected -> wipe -> r3 == 0. Keeping the existing
// (trimmed) checkpoint instead lets the next real turn (turn 3) still restore (r3 > 0).
// r2 is the regenerate turn (already high reuse); r3 is the load-bearing assertion.
static bool phaseb_restore_chain(const char* model_path, int& r2, int& r3) {
    r2 = r3 = -1;
    llama_rn_context ctx;
    if (!load_chat_ctx(ctx, model_path)) { std::cout << "load failed\n"; return false; }

    const std::vector<Msg> first = {{"user", "Hi, my name is Ali. Give a one-line greeting."}};
    auto t1 = chat_turn(ctx, first, 24, /*checkpoint*/true);          // turn 1 (cold)
    // turn 2: REGENERATE the same first message -> prompt is fully cached -> overshoot.
    auto t2 = chat_turn(ctx, first, 24, /*checkpoint*/true);          // turn 2 (overshoot path)
    r2 = t2.reused;
    // turn 3: a real follow-up. With the bug this WIPES (r3==0); with the fix it restores.
    std::vector<Msg> conv = first;
    conv.push_back({"assistant", t2.text});
    conv.push_back({"user", "Name one primary color, single word."});
    auto t3 = chat_turn(ctx, conv, 16, /*checkpoint*/true);           // turn 3 (must restore)
    r3 = t3.reused;
    return true;
}

// Cross-chat leak guard. With kv_checkpoint='memory', chat A establishes a name + a
// boundary checkpoint; clearCache(false) must invalidate that checkpoint so a new chat
// B (which shares the leading chat-template prefix) can neither restore a stale snapshot
// nor leak the name. Returns 0 on PASS, 1 on FAIL, 2 on load failure.
static int run_clearcache_leak_test(const char* model_path) {
    std::cout << "\n========== clearCache LEAK TEST :: " << model_path << " ==========\n";
    llama_rn_context ctx;
    if (!load_chat_ctx(ctx, model_path)) { std::cout << "load failed\n"; return 2; }

    auto tA = chat_turn(ctx, {{"user", "Remember this fact: my name is Ali. Reply with just OK."}},
                        24, /*checkpoint*/true);
    const bool ckpt_after_A = ctx.completion->kv_checkpoint.valid;

    ctx.clearCache(false);
    const bool ckpt_after_clear = ctx.completion->kv_checkpoint.valid;

    auto tB = chat_turn(ctx, {{"user", "What is my name? If you do not know, reply UNKNOWN."}},
                        24, /*checkpoint*/true);

    // Word-boundary match so "realize"/"alias" don't count as a leak of "ali".
    const bool no_leak = !contains_word(tB.text, "ali");
    const bool no_stale_restore = !ckpt_after_clear; // clearCache must have dropped it

    std::cout << "  chatA: checkpoint valid=" << ckpt_after_A << " resp=\"" << tA.text.substr(0, 40) << "\"\n";
    std::cout << "  after clearCache: checkpoint valid=" << ckpt_after_clear
              << " (must be 0)\n";
    std::cout << "  chatB: reused=" << tB.reused << " resp=\"" << tB.text.substr(0, 60) << "\"\n";
    const bool ok = no_leak && no_stale_restore;
    std::cout << "  LEAK GATE: no_leak=" << no_leak << " no_stale_restore=" << no_stale_restore
              << "  => " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

static int run_phaseb_validation(const char* model_path) {
    std::cout << "\n========== KV-CHECKPOINT VALIDATION :: " << model_path << " ==========\n";
    {
        // Arch probe (separate throwaway context).
        llama_rn_context probe;
        common_params pp; pp.model.path = model_path; pp.n_ctx = 512; pp.n_gpu_layers = 0;
        if (!probe.loadModel(pp)) { std::cout << "load failed\n"; return 2; }
        const bool recurrent = llama_model_is_recurrent(probe.model) || llama_model_is_hybrid(probe.model);
        const bool swa = llama_model_n_swa(probe.model) > 0;
        if (!recurrent && !swa) {
            std::cout << "\n*** SKIP: '" << model_path << "' is a PURE-ATTENTION model. ***\n"
                      << "*** Its seq_rm always succeeds, so the wipe-and-restore path is NOT\n"
                      << "*** exercised here. Run on a hybrid/SSM/SWA model for a meaningful result.\n";
            return 3; // skip, not pass
        }
    }

    const int GEN1 = 32;
    const int GEN2 = 32;

    // ---- Baseline pass: kv_checkpoint='off' (today's behavior, wipe on divergence).
    //      Used only for the GATE(1) PERF comparison (r2_off); its token output is no
    //      longer compared against memory (see GATE(2) rationale below). ----
    size_t cache1_off = 0;
    auto off2 = phaseb_two_turns(model_path, /*checkpoint*/false, GEN1, GEN2, cache1_off);
    if (off2.reused < 0) return 2;
    std::cout << "  [off] turn1 cache=" << cache1_off << "  turn2 reused=" << off2.reused
              << " resp=\"" << off2.text.substr(0, 50) << "\"\n";

    // ---- Memory pass: kv_checkpoint='memory' (restore on divergence), run twice in
    //      fresh contexts. This test targets recurrent/hybrid (SSM) models, where a
    //      checkpoint restore rebuilds the shared-prefix SSM state under a different
    //      batch layout than a fresh reprocess. By FP non-associativity that state is
    //      not bit-identical to the 'off' reprocess, which can flip near-tie greedy
    //      tokens -- so "memory == off baseline" is the wrong correctness bar. The real
    //      guarantee is that restore output is coherent and deterministic (stable
    //      run-to-run), so the two memory runs must be token-identical to each other.
    size_t cache1_on = 0;
    auto mem2 = phaseb_two_turns(model_path, /*checkpoint*/true, GEN1, GEN2, cache1_on);
    if (mem2.reused < 0) return 2;
    std::cout << "  [mem] turn1 cache=" << cache1_on << "  turn2 reused=" << mem2.reused
              << " resp=\"" << mem2.text.substr(0, 50) << "\"\n";

    size_t cache1_on2 = 0;
    auto mem2b = phaseb_two_turns(model_path, /*checkpoint*/true, GEN1, GEN2, cache1_on2);
    if (mem2b.reused < 0) return 2;
    std::cout << "  [mem2] turn1 cache=" << cache1_on2 << "  turn2 reused=" << mem2b.reused
              << " resp=\"" << mem2b.text.substr(0, 50) << "\"\n";

    // ---- Skip loud if the baseline did NOT wipe. If 'off' already reused (r2_off > 0),
    //      seq_rm succeeded and the restore path is never reached on this model/prompt;
    //      a "reused>0 with memory" verdict would then be trivially true and prove
    //      nothing. Report SKIP (exit 3) instead of a fake pass. ----
    if (off2.reused > 0) {
        std::cout << "\n*** SKIP: baseline (kv_checkpoint='off') reused " << off2.reused
                  << " > 0, i.e. it did NOT wipe.\n"
                  << "*** The restore path is not exercisable on this model/prompt, so the\n"
                  << "*** PERF gate cannot be validated here. Not counted as a pass.\n";
        return 3;
    }

    // ---- (1) PERF: REQUIRE restore_exercised (memory reused strictly MORE than the
    //      wiped baseline). reused>0 alone is insufficient -- a dead restore path that
    //      silently wiped would still show reused>0 from the cold reprocess. ----
    const bool restore_exercised = (mem2.reused > off2.reused);
    bool perf_ok = restore_exercised;
    std::cout << "\n  GATE(1) PERF       : " << (perf_ok ? "PASS" : "FAIL")
              << "  (memory reused=" << mem2.reused << " vs baseline reused=" << off2.reused
              << (restore_exercised ? " ; RESTORE prevented the wipe"
                                    : " ; restore did NOT beat the wipe -> dead path") << ")\n";

    // ---- (1b) SECOND-REUSE chain: a turn-2 save that pinned the boundary past the
    //      trimmed point would break turn 3's restore. Require both the first and
    //      second reuse turn to restore (reused > 0). ----
    int rc2 = 0, rc3 = 0;
    const bool chain_ran = phaseb_restore_chain(model_path, rc2, rc3);
    const bool chain_ok = chain_ran && rc2 > 0 && rc3 > 0;
    std::cout << "  GATE(1b) 2ND-REUSE : " << (chain_ok ? "PASS" : "FAIL")
              << "  (turn2 reused=" << rc2 << " ; turn3 reused=" << rc3
              << " ; both must be >0 -> turn-2 save kept the TRIMMED boundary)\n";
    perf_ok = perf_ok && chain_ok;

    // ---- (2) CORRECTNESS: restore output must be coherent and deterministic. On the
    //      recurrent/hybrid (SSM) models this test targets, a restore rebuilds the
    //      shared-prefix SSM state under a different batch layout than the 'off' full
    //      reprocess, so by FP non-associativity the two are not bit-identical and
    //      near-tie greedy tokens can flip -- "memory == off" is the wrong bar. Instead
    //      assert that the restored generation is coherent/non-empty and stable: two
    //      independent memory runs must produce identical tokens (mem-vs-mem, not
    //      mem-vs-off). ----
    const auto& pb = mem2.token_ids;
    const auto& pb2 = mem2b.token_ids;
    size_t n = std::min(pb.size(), pb2.size());
    size_t first_diff = n;
    for (size_t i = 0; i < n; i++) if (pb[i] != pb2[i]) { first_diff = i; break; }
    const bool mem_deterministic = (pb.size() == pb2.size()) && (first_diff == n);
    const bool mem_coherent = !pb.empty() && !looks_like_garbage(mem2.text);
    const bool correctness_ok = mem_deterministic && mem_coherent;

    std::cout << "  GATE(2) CORRECTNESS: " << (correctness_ok ? "PASS" : "FAIL")
              << "  (mem-vs-mem deterministic=" << mem_deterministic
              << " coherent=" << mem_coherent
              << " ; compared " << n << " tokens; mem=" << pb.size()
              << " mem2=" << pb2.size();
    if (!mem_deterministic) std::cout << " first_diff_at=" << (first_diff == n ? -1 : (long)first_diff);
    std::cout << ")\n";

    if (!mem_deterministic && first_diff < n) {
        std::cout << "  diverge at token " << first_diff
                  << ": mem=" << pb[first_diff] << " mem2=" << pb2[first_diff] << "\n";
        std::cout << "  mem  full : \"" << mem2.text << "\"\n";
        std::cout << "  mem2 full : \"" << mem2b.text << "\"\n";
    }

    bool gate_pass = perf_ok && correctness_ok;
    std::cout << "\n  ===== KV-CHECKPOINT VALIDATION: "
              << (gate_pass ? "PASS (reused>off AND restore deterministic+coherent)"
                            : "FAIL") << " =====\n";
    return gate_pass ? 0 : 1;
}

// ---- Restore-failure fallback: a corrupted snapshot must degrade to a clean
// WIPE (coherent output, no crash) and the wipe turn must re-save a fresh
// checkpoint. ----
static void run_restore_failure_test(const char* model_path) {
    std::cout << "\n========== RESTORE-FAILURE FALLBACK :: " << model_path << " ==========\n";
    llama_rn_context ctx;
    if (!load_chat_ctx(ctx, model_path)) { check("restore-failure: model loads", false); return; }
    auto* c = ctx.completion;

    std::vector<Msg> conv = {{"user", "Hi, my name is Ali. Tell me a short greeting."}};
    auto t1 = chat_turn(ctx, conv, 24, /*checkpoint*/true);
    if (!c->kv_checkpoint.valid) {
        std::cout << "  SKIP: no checkpoint saved on turn 1 (arch not checkpoint-eligible).\n";
        return;
    }

    // Corrupt the snapshot blob. The magic still matches (leading bytes of a real
    // blob), so restore() reaches the state reader and must fail on truncation.
    c->kv_checkpoint.data.resize(8);

    conv.push_back({"assistant", t1.text});
    conv.push_back({"user", "Write one sentence about why the ocean is blue."});
    auto t2 = chat_turn(ctx, conv, 32, /*checkpoint*/true);

    if (t2.action == ReuseAction::NORMAL_REUSE || t2.action == ReuseAction::COLD) {
        std::cout << "  SKIP: seq_rm succeeded on this model/prompt; the corrupt snapshot was "
                     "never consulted (action=" << action_name(t2.action) << ").\n";
        return;
    }
    check("corrupt restore falls back to WIPE", t2.action == ReuseAction::WIPE,
          std::string("action=") + action_name(t2.action));
    check("output after failed restore is coherent", !looks_like_garbage(t2.text),
          "resp=\"" + t2.text.substr(0, 50) + "\"");
    check("fresh checkpoint saved on the wipe turn",
          c->kv_checkpoint.valid && c->kv_checkpoint.data.size() > 8,
          "valid=" + std::to_string(c->kv_checkpoint.valid) +
          " blob=" + std::to_string(c->kv_checkpoint.data.size()) + "B");
}

// ---- Context-shift invalidation: once generation shifts the cache, every
// checkpoint position is renumbered; the snapshot must be dropped and the next
// turn must not RESTORE from it. ----
static void run_ctx_shift_test(const char* model_path) {
    std::cout << "\n========== CTX-SHIFT INVALIDATION :: " << model_path << " ==========\n";
    llama_rn_context ctx;
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 256;              // small so generation overflows quickly
    params.n_batch = 128;
    params.cpuparams.n_threads = 6;
    params.n_gpu_layers = 0;
    params.ctx_shift = true;
    params.sampling.temp = 0.0f;
    params.sampling.top_k = 1;
    params.sampling.ignore_eos = true;   // force generation to actually reach the shift
    if (!ctx.loadModel(params)) { check("ctx-shift: model loads", false); return; }
    auto* c = ctx.completion;

    std::vector<Msg> conv = {{"user", "Hi, my name is Ali."}};
    auto t1 = chat_turn(ctx, conv, 8, /*checkpoint*/true);
    if (!c->kv_checkpoint.valid) {
        std::cout << "  SKIP: no checkpoint saved on turn 1 (arch not checkpoint-eligible).\n";
        return;
    }

    // Turn 2 generates past n_ctx -> nextToken shifts the cache mid-generation.
    conv.push_back({"assistant", t1.text});
    conv.push_back({"user", "Tell me a very long story about the sea."});
    auto t2 = chat_turn(ctx, conv, 320, /*checkpoint*/true);
    std::cout << "  turn2 predicted=" << t2.predicted << " (n_ctx=256, shift expected)\n";
    if (!c->truncated && t2.predicted < 200) {
        std::cout << "  SKIP: generation ended before the context shifted; nothing to assert.\n";
        return;
    }
    check("ctx shift drops the checkpoint (valid==false)", !c->kv_checkpoint.valid);

    // Turn 3 must not restore from a renumbered snapshot.
    conv.push_back({"assistant", t2.text.substr(0, 120)});
    conv.push_back({"user", "Name one primary color, single word."});
    auto t3 = chat_turn(ctx, conv, 8, /*checkpoint*/true);
    check("turn after ctx shift does not RESTORE", t3.action != ReuseAction::RESTORE,
          std::string("action=") + action_name(t3.action));
}

// ---- Embedding interleave: embedding() runs loadPrompt with
// allow_kv_checkpoint=false, so it must neither overwrite nor drop the chat's
// checkpoint (no-clobber), and the next divergent chat turn must still be able
// to RESTORE it. ----
static void run_embedding_preservation_test(const char* model_path) {
    std::cout << "\n========== EMBEDDING NO-CLOBBER :: " << model_path << " ==========\n";
    llama_rn_context ctx;
    if (!load_chat_ctx(ctx, model_path)) { check("embedding: model loads", false); return; }
    auto* c = ctx.completion;

    std::vector<Msg> conv = {{"user", "Hi, my name is Ali. Tell me a short greeting."}};
    auto t1 = chat_turn(ctx, conv, 24, /*checkpoint*/true);
    if (!c->kv_checkpoint.valid) {
        std::cout << "  SKIP: no checkpoint saved on turn 1 (arch not checkpoint-eligible).\n";
        return;
    }
    const size_t boundary_size = c->kv_checkpoint.boundary_tokens.size();
    const size_t blob_size = c->kv_checkpoint.data.size();

    // Interleave an embedding request (the model does not need to be an embedding
    // model for the loadPrompt path to run; unsupported models just return zeros).
    ctx.params.prompt = "totally unrelated embedding text about volcanoes";
    common_params ep = ctx.params;
    ep.embedding = true;
    try {
        c->embedding(ep);
    } catch (const std::exception& e) {
        std::cout << "  SKIP: embedding() unusable on this model (" << e.what() << ").\n";
        return;
    }

    check("embedding() preserves the checkpoint blob (no-clobber)",
          c->kv_checkpoint.valid &&
          c->kv_checkpoint.boundary_tokens.size() == boundary_size &&
          c->kv_checkpoint.data.size() == blob_size,
          "valid=" + std::to_string(c->kv_checkpoint.valid) +
          " boundary=" + std::to_string(c->kv_checkpoint.boundary_tokens.size()) +
          "/" + std::to_string(boundary_size));

    // A divergent follow-up on the ORIGINAL conversation. The preserved snapshot
    // is only worth keeping if this turn can actually restore from it.
    conv.push_back({"assistant", t1.text});
    conv.push_back({"user", "Write one sentence about why the ocean is blue."});
    auto t2 = chat_turn(ctx, conv, 32, /*checkpoint*/true);
    check("chat turn after embedding() RESTOREs the preserved checkpoint",
          t2.action == ReuseAction::RESTORE,
          std::string("action=") + action_name(t2.action) +
          " reused=" + std::to_string(t2.reused));
}

// ============================================================================
// 3. DESCRIPTIVE SCENARIOS (observability only, no asserts)
// ============================================================================

// Drive one completion turn. If generate>0, run the decode loop to populate the
// KV cache + embd (turn 1). Returns n_past observed right after loadPrompt.
static int describe_turn(llama_rn_context& ctx, const std::string& prompt,
                         int generate, const char* label) {
    TurnOptions o;
    o.n_predict = generate;
    auto r = run_turn(ctx, prompt, o);
    std::cout << "  [" << label << "] n_past(reused)=" << r.reused
              << "  generated=" << r.predicted
              << "  embd_now=" << ctx.completion->embd.size() << "\n";
    return r.reused;
}

static void scenario(const std::string& title, bool kv_unified,
                     const std::string& p1, int gen1,
                     const std::string& p2) {
    std::cout << "\n===== " << title << "  (kv_unified=" << kv_unified << ") =====\n";
    if (!std::filesystem::exists(MODEL)) { std::cout << "model missing\n"; return; }
    llama_rn_context ctx;
    common_params params;
    params.model.path = MODEL;
    params.n_ctx = 512;
    params.n_batch = 128;
    params.cpuparams.n_threads = 2;
    params.n_gpu_layers = 0;
    params.kv_unified = kv_unified;
    if (!ctx.loadModel(params)) { std::cout << "load failed\n"; return; }

    // Turn 1: process p1 and GENERATE so the KV cache + embd are populated.
    describe_turn(ctx, p1, gen1, "turn1 (cold)");
    // Turn 2: process p2 — measure how much prefix is reused.
    describe_turn(ctx, p2, 0, "turn2 (reuse?)");
}

// Feed turn-1's exact cached tokens back as turn-2's prefix (detokenize the cache
// verbatim + a suffix). If the history round-trips exactly, n_past == full cache and
// the removal is an empty range, so seq_rm should succeed even on hybrid/SSM models.
static void faithful_scenario(const std::string& p1, int gen1, const std::string& suffix) {
    std::cout << "\n===== FAITHFUL ROUND-TRIP (does clean extension reuse on this arch?) =====\n";
    llama_rn_context ctx;
    common_params params;
    params.model.path = MODEL;
    params.n_ctx = 1024;
    params.n_batch = 256;
    params.cpuparams.n_threads = 4;
    params.n_gpu_layers = 0;
    if (!ctx.loadModel(params)) { std::cout << "load failed\n"; return; }

    describe_turn(ctx, p1, gen1, "turn1 (cold)");
    std::vector<llama_token> full = ctx.completion->embd;   // exact cached sequence
    std::string faithful = common_detokenize(ctx.ctx, full, true);

    int reused = describe_turn(ctx, faithful + suffix, 0, "turn2 (faithful)");
    bool ok = reused >= (int)full.size() - 2;   // allow tiny detok/retok drift
    std::cout << "  >>> cache=" << full.size() << " reused=" << reused
              << "  => " << (ok ? "REUSE WORKS" : "STILL WIPES") << "\n";
}

static void descriptive_scenarios() {
    // P1 base. P2a = clean extension (SmolLM3-like, prompt is a strict prefix).
    // P2b = mid-divergence (Qwen3.5-like: shares early prefix, differs in middle).
    const std::string P1  = "alpha beta gamma delta epsilon zeta eta theta iota";
    const std::string P2a = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
    const std::string P2b = "alpha beta gamma DIVERGE here now totally different tail words";

    for (bool ku : {false, true}) {
        scenario("CLEAN EXTENSION (expect high reuse)", ku, P1, 8, P2a);
        scenario("MID DIVERGENCE (partial reuse, or WIPE?)", ku, P1, 8, P2b);
    }
    faithful_scenario(P1, 8, " kappa lambda mu nu xi omicron");
    std::cout << "\nWatch the [REUSE] log line from rn-completion.cpp for seq_rm ok/FAIL.\n";
}

// ============================================================================
// Checkpoint footprint measurement (./reuse_test <model> size)
// ----------------------------------------------------------------------------
// llama_state_seq_get_size is the exact #bytes llama_state_seq_get_data would
// copy for one sequence. Reported across context lengths to show scaling
// (recurrent state is ~constant; attention KV scales with positions).
// ============================================================================
static void measure_checkpoint(const char* model_path) {
    llama_rn_context ctx;
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 4096;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.n_gpu_layers = 0;
    if (!ctx.loadModel(params)) { std::cout << "load failed\n"; return; }
    std::cout << "\n===== CHECKPOINT SIZE (llama_state_seq_get_size, seq 0) :: "
              << model_path << " =====\n";
    for (int target : {0, 128, 512, 1024, 2048}) {
        std::string prompt;
        for (int w = 0; w < target; w++) prompt += "word ";
        ctx.clearCache(true);
        TurnOptions o;
        o.n_predict = 1;
        run_turn(ctx, prompt, o);
        int n = ctx.completion->n_past;
        size_t bytes = llama_state_seq_get_size(ctx.ctx, 0);
        std::cout << "  positions=" << n << "  checkpoint=" << bytes
                  << " bytes  (" << (bytes / 1048576.0) << " MB)\n";
    }
}

// ============================================================================
// Benchmark (./reuse_test <model> bench [off|memory] [n_turns])
// ----------------------------------------------------------------------------
// Quantifies the checkpoint tradeoff on a growing multi-turn chat: what the
// user pays (RAM: the serialized blob) for what they get (latency: prefill /
// time-to-first-token saved by restoring instead of reprocessing history).
//
// No filter -> runs both modes in one process and prints a per-turn speedup
// table (deterministic, greedy, so the two modes are directly comparable).
// With a mode filter -> runs only that mode and prints its peak RSS, so a caller
// (run-android.sh) can diff peak RSS across two separate processes -- the only
// way to get a true peak-RSS delta, since VmHWM is process-wide.
// ============================================================================
static size_t read_status_kb(const char* field) {
    // Linux/Android: /proc/self/status VmHWM (peak) / VmRSS (current), in kB.
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, field, strlen(field)) == 0) {
            sscanf(line + strlen(field), " %zu", &kb);
            break;
        }
    }
    fclose(f);
    return kb;
}
static double peak_rss_mb() {
    size_t kb = read_status_kb("VmHWM:");        // Linux/Android
    if (kb) return kb / 1024.0;
    struct rusage ru;                            // macOS fallback (ru_maxrss = bytes)
    if (getrusage(RUSAGE_SELF, &ru) == 0) return ru.ru_maxrss / 1048576.0;
    return 0.0;
}

struct BenchTurn { int prompt_toks; double ttft_ms; int reused; ReuseAction action; };

static std::vector<BenchTurn> bench_conversation(llama_rn_context& ctx, bool kv, int n_turns) {
    ctx.clearCache(true);
    std::vector<Msg> conv;
    std::vector<BenchTurn> out;
    static const char* USER[] = {
        "Hi! Tell me a short fact about the ocean.",
        "Interesting. Now tell me one about mountains.",
        "Nice. And one about rivers?",
        "Great. What about deserts?",
        "Cool. Tell me about forests now.",
        "Thanks. One more about volcanoes.",
        "And glaciers?",
        "Finally, one about coral reefs.",
    };
    const int nprompt = (int)(sizeof(USER) / sizeof(USER[0]));
    for (int t = 0; t < n_turns; t++) {
        conv.push_back({"user", USER[t % nprompt]});
        TurnOptions o; o.n_predict = 24; o.kv_checkpoint = kv;
        std::string prompt = format_chat(ctx, conv);
        int ptoks = (int)common_tokenize(ctx.ctx, prompt.c_str(), true, true).size();
        auto r = run_turn(ctx, prompt, o);
        out.push_back({ptoks, r.ttft_ms, r.reused, r.action});
        conv.push_back({"assistant", r.text});
    }
    return out;
}

static int run_bench(const char* model_path, const std::string& mode_filter, int n_turns) {
    auto load = [&](llama_rn_context& ctx) {
        common_params p; p.model.path = model_path;
        p.n_ctx = 4096; p.n_batch = 512;
        p.cpuparams.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        const char* env = getenv("RNLLAMA_BENCH_NGL");
        p.n_gpu_layers = env ? atoi(env) : 0;
        return ctx.loadModel(p);
    };

    std::cout << "\n===== BENCH :: " << model_path
              << "  (turns=" << n_turns << ", threads="
              << std::max(1u, std::thread::hardware_concurrency() / 2) << ") =====\n";

    // Single-mode: report per-turn TTFT + peak RSS + blob size for cross-process diff.
    if (mode_filter == "off" || mode_filter == "memory") {
        const bool kv = (mode_filter == "memory");
        llama_rn_context ctx;
        if (!load(ctx)) { std::cout << "load failed\n"; return 2; }
        auto turns = bench_conversation(ctx, kv, n_turns);
        size_t blob = kv ? llama_state_seq_get_size(ctx.ctx, 0) : 0;
        std::cout << "MODE=" << mode_filter << "\n";
        for (size_t i = 0; i < turns.size(); i++)
            std::cout << "  turn " << (i + 1) << "  prompt_toks=" << turns[i].prompt_toks
                      << "  reused=" << turns[i].reused
                      << "  action=" << action_name(turns[i].action)
                      << "  ttft_ms=" << turns[i].ttft_ms << "\n";
        std::cout << "PEAK_RSS_MB=" << peak_rss_mb() << "\n";
        std::cout << "CHECKPOINT_BLOB_MB=" << (blob / 1048576.0) << "\n";
        return 0;
    }

    // Both modes in one process: per-turn speedup table.
    llama_rn_context ctx_off;   if (!load(ctx_off)) { std::cout << "load failed\n"; return 2; }
    auto off = bench_conversation(ctx_off, false, n_turns);
    llama_rn_context ctx_mem;   if (!load(ctx_mem)) { std::cout << "load failed\n"; return 2; }
    auto mem = bench_conversation(ctx_mem, true, n_turns);
    size_t blob = llama_state_seq_get_size(ctx_mem.ctx, 0);

    std::cout << "\n turn | p_toks | off_ttft_ms | mem_ttft_ms | mem_action | speedup\n";
    std::cout << " -----+--------+-------------+-------------+------------+--------\n";
    double sum_off = 0, sum_mem = 0;
    for (size_t i = 0; i < off.size() && i < mem.size(); i++) {
        double sp = mem[i].ttft_ms > 0 ? off[i].ttft_ms / mem[i].ttft_ms : 0;
        sum_off += off[i].ttft_ms; sum_mem += mem[i].ttft_ms;
        char row[256];
        snprintf(row, sizeof(row), " %4zu | %6d | %11.1f | %11.1f | %-10s | %5.2fx\n",
                 i + 1, off[i].prompt_toks, off[i].ttft_ms, mem[i].ttft_ms,
                 action_name(mem[i].action), sp);
        std::cout << row;
    }
    std::cout << " -----+--------+-------------+-------------+------------+--------\n";
    std::cout << " total off=" << sum_off << "ms  memory=" << sum_mem << "ms  overall speedup="
              << (sum_mem > 0 ? sum_off / sum_mem : 0) << "x\n";
    std::cout << " checkpoint blob = " << (blob / 1048576.0) << " MB (marginal RAM cost, ~constant on recurrent)\n";
    return 0;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    if (argc > 2 && std::string(argv[2]) == "size") {
        measure_checkpoint(argv[1]);
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "bench") {
        // ./reuse_test <model> bench [off|memory] [n_turns]
        std::string mode = (argc > 3) ? argv[3] : "";
        int n_turns = 6;
        if (argc > 4) n_turns = atoi(argv[4]);
        else if (!mode.empty() && mode != "off" && mode != "memory") { n_turns = atoi(mode.c_str()); mode = ""; }
        return run_bench(argv[1], mode, n_turns > 0 ? n_turns : 6);
    }

    // The unit section ALWAYS runs and always gates the exit code.
    run_unit_section(argc > 1 ? argv[1] : nullptr);

    bool ran_gated = false;
    if (argc > 1) {
        // A real model was provided: run the KV-checkpoint validation as the primary
        // gated section.
        MODEL = argv[1];
        std::cout << "\nmodel: " << MODEL << "\n";

        int leak_rc = run_clearcache_leak_test(MODEL);
        if (leak_rc == 2) return 2;
        check("clearCache leak gate", leak_rc == 0);

        int rc = run_phaseb_validation(MODEL);
        if (rc == 2) return 2;
        if (rc == 3) {
            std::cout << "(validation skipped; falling through to descriptive scenarios)\n";
        } else {
            check("KV-checkpoint validation gates (perf + 2nd-reuse + correctness)", rc == 0);
            ran_gated = true;
        }

        // These self-SKIP (without failing) when the arch never saves a checkpoint.
        run_restore_failure_test(MODEL);
        run_ctx_shift_test(MODEL);
        run_embedding_preservation_test(MODEL);
    }

    if (argc <= 1 || !ran_gated) {
        std::cout << "\nmodel: " << MODEL << "\n";
        descriptive_scenarios();
    }

    std::cout << "\n===== reuse_test: " << g_pass << " passed, " << g_fail << " failed =====\n";
    return g_fail == 0 ? 0 : 1;
}
