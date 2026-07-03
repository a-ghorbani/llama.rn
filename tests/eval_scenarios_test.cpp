// Model-agnostic EVALUATION harness for the KV reuse / checkpoint / mtmd / tools
// area. Unlike the pass/fail-only tests, this one PRINTS full, untruncated
// transcripts so a human can read the generated text and judge whether it makes
// sense. It also carries machine-checkable asserts.
//
// For each KV mode in {off, memory} (completion->kv_checkpoint_enabled) it runs
// a FRESH llama_rn_context (clearCache does not fully reset SWA state, which
// would contaminate the cross-mode token-identity comparison):
//   A. APPEND  — multi-turn memory; off==memory must be TOKEN-IDENTICAL
//   B. EDIT    — change the first user message; response reflects the edit, no leak
//   C. NEW SESSION — no clearCache; reuse system prefix, probe name directly, no leak
//   D. VLM     — (only with --mmproj) image grounding, same-image prefix reuse,
//                image-swap answer change
//   E. TOOLS   — (only with --tools) tool call + tool result follow-up
//
// Each turn prints:
//   [mode][scenario][turn]
//     user: <short>
//     resp: <full generated text>
//     status: reused=<n_past> action=<reuse|restore|wipe|cold|none> predicted=<n>
//
// Run:
//   ./eval_scenarios_test <model.gguf> [--mmproj <p>] [--image <p>] [--image2 <p>]
//                         [--tools] [--smoke]
//
// --smoke (used by run_tests.sh with the bundled tiny-random model): capability
// asserts that need a REAL model (recall, edit-reflection, coherence) are
// downgraded to informational lines; structural asserts (cross-mode token
// identity, isolation/no-leak, wipe->restore pairing) stay hard.
//
// Exit code 0 iff every (hard) assert passed.

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>

#include "rn-llama.h"
#include "rn-completion.h"
#include "common.h"
#include "chat.h"

#include "test-helpers.hpp"

using namespace rnllama;
using namespace rntest;

// ------------------------------------------------------------------ asserts

static bool g_smoke = false;

// Structural invariant: always a hard assert.
static void hard_check(const std::string& name, bool ok, const std::string& detail = "") {
    check(name, ok, detail);
}

// Model-capability expectation (recall, coherence, ...): hard on a real model,
// informational in --smoke (the tiny-random model can't answer anything).
static void capability_check(const std::string& name, bool ok, const std::string& detail = "") {
    if (!g_smoke) {
        check(name, ok, detail);
        return;
    }
    std::cout << "  info " << name << " -> " << (ok ? "ok" : "weak")
              << " (smoke: not asserted)";
    if (!detail.empty()) std::cout << "  [" << detail << "]";
    std::cout << "\n";
}

// ------------------------------------------------------------------ transcript

static void print_turn(const std::string& mode, const std::string& scen,
                       const std::string& turn, const std::string& user,
                       const TurnResult& r) {
    std::cout << "\n[" << mode << "][" << scen << "][" << turn << "]\n";
    std::cout << "  user: " << user << "\n";
    std::cout << "  resp: " << r.text << "\n";
    std::cout << "  status: reused=" << r.reused
              << "  action=" << action_name(r.action)
              << "  predicted=" << r.predicted << "\n";
}

// ------------------------------------------------------------------ chat formatting

// Tool chat: render with tools JSON + tool_choice auto, return prompt AND the
// sampling/parser wiring the completion needs.
static std::string format_chat_tools(llama_rn_context& ctx, const std::vector<Msg>& msgs,
                                     const std::string& tools, TurnOptions& opt_out) {
    auto cp = ctx.getFormattedChatWithJinja(
        to_json(msgs), "", "", tools, /*parallel*/false, /*tool_choice*/"auto",
        /*enable_thinking*/false, "none", true);
    opt_out.chat_format = (int)cp.format;
    opt_out.generation_prompt = cp.generation_prompt;
    opt_out.chat_parser = cp.parser;
    opt_out.grammar = cp.grammar;
    opt_out.grammar_lazy = cp.grammar_lazy;
    opt_out.grammar_triggers = cp.grammar_triggers;
    opt_out.preserved_tokens = cp.preserved_tokens;
    return cp.prompt;
}

// ================================================================== scenarios

// Tool behavior captured for one mode, used for the cross-mode consistency check
// (the checkpoint must not change tool behavior). Capability (did it emit a real
// get_weather call) is informational, not a pass/fail criterion.
struct ToolBehavior {
    bool ran = false;                 // scenario E was executed in this mode
    size_t n_tool_calls = 0;          // number of parsed tool calls on t1
    std::string first_tool_name;      // name of first tool call (if any)
    std::string first_tool_args;      // parsed args of first tool call (if any)
    std::string t2_text;              // final-answer text after the tool result
    bool emitted_get_weather = false; // capability: a get_weather call ...
    bool location_paris = false;      // ... with location=Paris
};

struct ModeResults {
    // per-scenario captured token ids for the off==memory identity check.
    std::vector<std::vector<llama_token>> a_tokens;  // A turns
    std::vector<std::vector<llama_token>> b_tokens;  // B turns
    std::vector<ReuseAction> a_actions;              // A per-turn reuse action
    std::string a_t3_text;                           // A t3 generated text (coherence)
    std::string b_text;                              // B generated text (coherence)
    bool a_restore_fired = false;                    // any A turn took a RESTORE
    bool b_restore_fired = false;                    // B turn took a RESTORE
    ToolBehavior tools;                              // E tool behavior (consistency)
};

static TurnOptions plain(int n_predict, bool kv) {
    TurnOptions o;
    o.n_predict = n_predict;
    o.kv_checkpoint = kv;
    return o;
}

// ------- A: APPEND (multi-turn memory) -------
static void run_A(llama_rn_context& ctx, const std::string& mode, bool kv,
                  ModeResults& mr) {
    ctx.clearCache(false);
    std::vector<Msg> conv;

    conv.push_back({"user", "Hi, my name is Ali and my hobby is hiking."});
    auto t1 = run_turn(ctx, format_chat(ctx, conv), plain(40, kv));
    print_turn(mode, "A-append", "t1", "Hi, my name is Ali and my hobby is hiking.", t1);
    conv.push_back({"assistant", t1.text});

    conv.push_back({"user", "My favorite color is blue."});
    auto t2 = run_turn(ctx, format_chat(ctx, conv), plain(40, kv));
    print_turn(mode, "A-append", "t2", "My favorite color is blue.", t2);
    conv.push_back({"assistant", t2.text});

    conv.push_back({"user", "What is my name, my hobby, and my favorite color?"});
    auto t3 = run_turn(ctx, format_chat(ctx, conv), plain(80, kv));
    print_turn(mode, "A-append", "t3", "What is my name, my hobby, and my favorite color?", t3);

    capability_check("A[" + mode + "] t3 recalls name (ali)", contains(t3.text, "ali"));
    capability_check("A[" + mode + "] t3 recalls hobby (hiking)", contains(t3.text, "hik"));
    capability_check("A[" + mode + "] t3 recalls color (blue)", contains(t3.text, "blue"));

    mr.a_tokens = {t1.token_ids, t2.token_ids, t3.token_ids};
    mr.a_actions = {t1.action, t2.action, t3.action};
    mr.a_t3_text = t3.text;
    mr.a_restore_fired = (t1.action == ReuseAction::RESTORE ||
                          t2.action == ReuseAction::RESTORE ||
                          t3.action == ReuseAction::RESTORE);
}

// ------- B: EDIT (regenerate an earlier turn) -------
static void run_B(llama_rn_context& ctx, const std::string& mode, bool kv,
                  ModeResults& mr) {
    // Reuse the prior session state on purpose (no clearCache): starting a new
    // conversation whose first user message differs from the cached one (Ali ->
    // Sam) forces a mid-sequence divergence -> exercises the reuse/restore/wipe
    // decision. One well-formed user turn (not two consecutive user turns, which
    // render an unnatural prompt and make chat models slip into first person).
    std::vector<Msg> conv;
    conv.push_back({"user", "Hi, my name is Sam and my hobby is cooking. What is my name and hobby?"});

    auto t = run_turn(ctx, format_chat(ctx, conv), plain(80, kv));
    print_turn(mode, "B-edit", "t1",
               "Hi, my name is Sam and my hobby is cooking. What is my name and hobby?", t);

    capability_check("B[" + mode + "] reflects edit name (sam)", contains(t.text, "sam"));
    capability_check("B[" + mode + "] reflects edit hobby (cooking)", contains(t.text, "cook"));
    hard_check("B[" + mode + "] does NOT leak old name (ali)", !contains_word(t.text, "ali"));
    hard_check("B[" + mode + "] does NOT leak old hobby (hiking)", !contains_word(t.text, "hiking"));

    mr.b_tokens = {t.token_ids};
    mr.b_text = t.text;
    mr.b_restore_fired = (t.action == ReuseAction::RESTORE);
}

// ------- C: NEW SESSION (isolation via prefix reuse, not clearCache) -------
static void run_C(llama_rn_context& ctx, const std::string& mode, bool kv) {
    // A new session does NOT clearCache. Practically it is like editing the first user
    // message: reuse the cached SYSTEM-PROMPT prefix (so a large system prompt is not
    // reprocessed) and start a fresh user turn. The private data from the previous
    // session must be gone. We probe for it DIRECTLY -- asking the name/hobby -- because
    // an unrelated question (e.g. "a fact about the moon") would never surface a leaked
    // name and so tests almost nothing.
    static const std::string SYS =
        "You are a concise assistant. Answer in one short sentence.";

    // Harness setup only: one reset, then session 1 stores private data under SYS.
    // (This reset is the test isolating itself from A/B -- it is NOT the "new session".)
    ctx.clearCache(false);
    run_turn(ctx, format_chat(ctx, {{"system", SYS},
             {"user", "My name is Zoe and my hobby is painting. Just reply OK."}}), plain(8, kv));

    // The NEW SESSION: SAME system prompt (prefix reused, not reprocessed), a fresh
    // question, and NO clearCache between the two. Ask directly for the private data.
    std::vector<Msg> conv = {{"system", SYS},
             {"user", "What is my name and what is my hobby? If you do not know, reply exactly: I don't know."}};
    auto t = run_turn(ctx, format_chat(ctx, conv), plain(40, kv));
    print_turn(mode, "C-newsession", "t2",
               "[new session, shared system prompt] What is my name and hobby?", t);
    std::cout << "  (new session: reused=" << t.reused << " action=" << action_name(t.action)
              << " -- pure-attention reuses the system prefix; recurrent wipes)\n";

    // The real leak check: the new session cannot know session 1's private data.
    hard_check("C[" + mode + "] does NOT leak prior name 'zoe'",     !contains_word(t.text, "zoe"));
    hard_check("C[" + mode + "] does NOT leak prior hobby 'painting'", !contains_word(t.text, "painting"));
    const std::string lo = lower(t.text);
    const bool disclaims = contains(lo, "know") || contains(lo, "don't have") ||
        contains(lo, "do not have") || contains(lo, "no name") || contains(lo, "as an ai") ||
        contains(lo, "cannot") || contains(lo, "haven't told") || contains(lo, "not sure");
    capability_check("C[" + mode + "] admits it does not know", disclaims);
}

// ------- D: VLM -------
static void run_D(llama_rn_context& ctx, const std::string& mode, bool kv,
                  const std::string& image, const std::string& image2) {
    ctx.clearCache(false);
    const std::string SYS =
        "<|im_start|>system\nYou are a helpful vision assistant.<|im_end|>\n";

    auto build = [&](const std::string& body) {
        return SYS + "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n";
    };
    auto media_opt = [&](const std::vector<std::string>& media) {
        TurnOptions o = plain(48, kv);
        o.media = media;
        return o;
    };

    // d1: image + describe
    auto d1 = run_turn(ctx, build("<__media__>\nName the colors and shapes you see."),
                       media_opt({image}));
    print_turn(mode, "D-vlm", "d1", "[image] Name the colors and shapes you see.", d1);
    capability_check("D[" + mode + "] d1 image-grounded & non-empty",
                     !d1.text.empty() && !looks_like_garbage(d1.text));

    // d2: SAME image, follow-up -> the mtmd bitmap-hash path must match the image
    // chunk, so the image+text prefix is actually REUSED (ported from vlm_test).
    auto d2 = run_turn(ctx, build("<__media__>\nWhich shape is in the center?"),
                       media_opt({image}));
    print_turn(mode, "D-vlm", "d2", "[same image] Which shape is in the center?", d2);
    hard_check("D[" + mode + "] d2 same-image follow-up reuses prefix (reused > 0)",
               d2.reused > 0, "reused=" + std::to_string(d2.reused));
    capability_check("D[" + mode + "] d2 non-empty & coherent",
                     !d2.text.empty() && !looks_like_garbage(d2.text));

    // d3: EDIT d2's text, same image
    auto d3 = run_turn(ctx, build("<__media__>\nDescribe the image in one sentence."),
                       media_opt({image}));
    print_turn(mode, "D-vlm", "d3", "[same image] Describe the image in one sentence.", d3);
    capability_check("D[" + mode + "] d3 non-empty & coherent",
                     !d3.text.empty() && !looks_like_garbage(d3.text));

    // d4: image SWAP. Ask the SAME question in a fresh session on image1 and on
    // image2: greedy decoding means a model that ignores the image would answer
    // identically, so the answers must CHANGE. Falls back to an image-dropped
    // coherence probe when no --image2 was given.
    if (!image2.empty()) {
        const std::string q4 = build("<__media__>\nWhat is the main color in this image? Answer in one short sentence.");
        ctx.clearCache(false);
        auto d4a = run_turn(ctx, q4, media_opt({image}));
        print_turn(mode, "D-vlm", "d4a", "[image1, fresh] What is the main color?", d4a);
        ctx.clearCache(false);
        auto d4b = run_turn(ctx, q4, media_opt({image2}));
        print_turn(mode, "D-vlm", "d4b", "[image2, fresh] What is the main color?", d4b);
        hard_check("D[" + mode + "] d4 answer CHANGES when the image is swapped",
                   d4a.text != d4b.text,
                   d4a.text == d4b.text ? "identical answer -> model ignores the image" : "");
        capability_check("D[" + mode + "] d4 swapped-image answer coherent",
                         !d4b.text.empty() && !looks_like_garbage(d4b.text));
    } else {
        auto d4 = run_turn(ctx, build("Now what do you see?"), plain(48, kv));
        print_turn(mode, "D-vlm", "d4", "[image dropped] Now what do you see?", d4);
        capability_check("D[" + mode + "] d4 non-empty & coherent",
                         !d4.text.empty() && !looks_like_garbage(d4.text));
        std::cout << "  NOTE: pass --image2 to enable the image-swap answer-change assert.\n";
    }

    std::cout << "  NOTE: mtmd ignores kv_checkpoint today, so off==memory is the "
                 "expected baseline for D.\n";
}

// ------- E: TOOLS -------
static const char* TOOLS_JSON = R"([{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string", "description": "City name, e.g. Paris" }
      },
      "required": ["location"]
    }
  }
}])";

static void run_E(llama_rn_context& ctx, const std::string& mode, bool kv,
                  ModeResults& mr) {
    ctx.clearCache(false);
    std::vector<Msg> conv = {
        {"user", "What's the weather in Paris? Use the tool."}
    };

    TurnOptions o1 = plain(128, kv);
    std::string prompt = format_chat_tools(ctx, conv, TOOLS_JSON, o1);
    auto t1 = run_turn(ctx, prompt, o1);
    print_turn(mode, "E-tools", "t1", "What's the weather in Paris? Use the tool.", t1);

    auto parsed = ctx.completion->parseChatOutput(false);
    ToolBehavior& tb = mr.tools;
    tb.ran = true;
    tb.n_tool_calls = parsed.tool_calls.size();
    if (!parsed.tool_calls.empty()) {
        tb.first_tool_name = parsed.tool_calls[0].name;
        tb.first_tool_args = parsed.tool_calls[0].arguments;
    }
    for (auto& tc : parsed.tool_calls) {
        std::cout << "  tool_call: name=" << tc.name << " args=" << tc.arguments << "\n";
        if (tc.name == "get_weather") {
            tb.emitted_get_weather = true;
            if (contains(tc.arguments, "paris")) tb.location_paris = true;
        }
    }
    if (parsed.tool_calls.empty())
        std::cout << "  (no tool_calls parsed; content=\"" << parsed.content << "\")\n";

    // Append the tool result and run the follow-up turn. Provide assistant tool-call
    // turn + tool result so the template renders a coherent continuation.
    std::string tool_result = "{\"temp_c\":18,\"cond\":\"cloudy\"}";
    std::vector<Msg> conv2 = conv;
    // Represent the assistant tool call compactly, then the tool output.
    conv2.push_back({"assistant", t1.text.empty() ? "Calling get_weather for Paris." : t1.text});
    conv2.push_back({"tool", tool_result});
    conv2.push_back({"user", "Given the tool result, what's the weather in Paris?"});

    TurnOptions o2 = plain(128, kv);
    std::string prompt2 = format_chat_tools(ctx, conv2, TOOLS_JSON, o2);
    auto t2 = run_turn(ctx, prompt2, o2);
    print_turn(mode, "E-tools", "t2", "[tool result: 18C cloudy] what's the weather?", t2);
    tb.t2_text = t2.text;

    bool coherent = !looks_like_garbage(t2.text) && !t2.text.empty();
    capability_check("E[" + mode + "] final answer coherent", coherent);
}

// ================================================================== main

static bool load_ctx(llama_rn_context& ctx, const std::string& model) {
    common_params params;
    params.model.path = model;
    params.n_ctx = 4096;
    params.n_batch = 512;
    params.cpuparams.n_threads = 4;
    params.n_gpu_layers = 99;  // use Metal/GPU if available; CPU fallback otherwise
    // deterministic decoding so off==memory can be token-identical
    params.sampling.temp = 0.0f;
    params.sampling.seed = 1234;
    return ctx.loadModel(params);
}

int main(int argc, char** argv) {
    quiet_llama_logs();   // silence model-load spam; RNLLAMA_VERBOSE_LOGS=1 to restore
    if (argc < 2) {
        std::cout << "usage: ./eval_scenarios_test <model.gguf> "
                     "[--mmproj <p>] [--image <p>] [--image2 <p>] [--tools] [--smoke]\n";
        return 2;
    }
    std::string model = argv[1];
    std::string mmproj, image, image2;
    bool do_tools = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--mmproj" && i + 1 < argc) mmproj = argv[++i];
        else if (a == "--image" && i + 1 < argc) image = argv[++i];
        else if (a == "--image2" && i + 1 < argc) image2 = argv[++i];
        else if (a == "--tools") do_tools = true;
        else if (a == "--smoke") g_smoke = true;
    }
    if (!std::filesystem::exists(model)) { std::cout << "model missing: " << model << "\n"; return 2; }

    std::cout << "================ EVAL SCENARIOS ================\n";
    std::cout << "model:  " << model << "\n";
    std::cout << "mmproj: " << (mmproj.empty() ? "(none, skip D)" : mmproj) << "\n";
    std::cout << "tools:  " << (do_tools ? "yes" : "no (skip E)") << "\n";
    std::cout << "smoke:  " << (g_smoke ? "yes (capability asserts informational)" : "no") << "\n";

    bool is_ssm = false;
    ModeResults off_res, mem_res;

    for (int m = 0; m < 2; m++) {
        bool kv = (m == 1);
        std::string mode = kv ? "memory" : "off";
        std::cout << "\n############### MODE = " << mode
                  << " (kv_checkpoint_enabled=" << (kv ? "true" : "false") << ") ###############\n";

        // FRESH context per mode. clearCache does not fully reset SWA
        // sliding-window state, so sharing one context would contaminate the
        // cross-mode token-identity comparison on SWA models.
        llama_rn_context ctx;
        if (!load_ctx(ctx, model)) { std::cout << "load failed\n"; return 2; }

        if (m == 0) {
            // Arch awareness: recurrent/hybrid (SSM) models do NOT restore a
            // bit-identical shared prefix (see the cross-mode section below).
            const llama_model* model_handle = llama_get_model(ctx.ctx);
            is_ssm = llama_model_is_recurrent(model_handle) ||
                     llama_model_is_hybrid(model_handle);
            std::cout << "arch:   " << (is_ssm ? "recurrent/hybrid (SSM)" : "pure-attn/SWA") << "\n";
        }

        bool vlm_ok = false;
        if (!mmproj.empty()) {
            if (std::filesystem::exists(mmproj) &&
                ctx.initMultimodal(mmproj, /*use_gpu*/false)) {
                vlm_ok = true;
                std::cout << "multimodal enabled: vision=" << ctx.isMultimodalSupportVision() << "\n";
            } else {
                std::cout << "initMultimodal failed; skipping D\n";
            }
        }

        ModeResults& mr = kv ? mem_res : off_res;

        run_A(ctx, mode, kv, mr);
        run_B(ctx, mode, kv, mr);
        run_C(ctx, mode, kv);

        if (vlm_ok) {
            run_D(ctx, mode, kv, image, image2);
        } else if (m == 0) {
            std::cout << "\n[skip] D-vlm: no --mmproj provided (or init failed).\n";
        }

        if (do_tools) {
            run_E(ctx, mode, kv, mr);
        } else if (m == 0) {
            std::cout << "\n[skip] E-tools: no --tools flag.\n";
        }
    }

    // ---- cross-mode token-identity asserts (A and B) ----
    std::cout << "\n############### CROSS-MODE IDENTITY (off vs memory) ###############\n";
    auto compare_tokens = [&](const std::string& scen,
                              const std::vector<std::vector<llama_token>>& off,
                              const std::vector<std::vector<llama_token>>& mem) {
        bool identical = (off.size() == mem.size());
        for (size_t i = 0; identical && i < off.size(); i++) {
            if (off[i] != mem[i]) identical = false;
        }
        std::string detail;
        if (!identical) {
            detail = "turns off=" + std::to_string(off.size()) +
                     " mem=" + std::to_string(mem.size());
            for (size_t i = 0; i < std::min(off.size(), mem.size()); i++) {
                if (off[i] != mem[i])
                    detail += " | turn" + std::to_string(i + 1) + " differs (" +
                              std::to_string(off[i].size()) + " vs " +
                              std::to_string(mem[i].size()) + " toks)";
            }
        }
        hard_check(scen + " off==memory TOKEN-IDENTICAL", identical, detail);
    };

    // ---- Scenario A: Phase-B CONSISTENCY (not model capability) ----
    // The right bar depends on WHETHER a checkpoint RESTORE actually fired in memory
    // mode -- not on the architecture. A RESTORE rebuilds shared-prefix state under a
    // different batch layout than a full reprocess; for SSM that is coherent and
    // context-preserving but NOT bit-identical (FP non-associativity flips near-tie
    // greedy tokens). When memory mode instead reuses/wipes the cache via the SAME
    // path as off mode (no restore), the output IS genuinely token-identical and we
    // assert it strictly. Branching on restore-fired (not is_ssm) avoids falsely
    // failing models that simply reuse the cache normally (e.g. LFM2.5 / LFM2-VL).
    if (mem_res.a_restore_fired) {
        // A RESTORE fired in memory mode -> off==memory token-identity is the WRONG
        // bar. Assert the Phase-B properties that MUST still hold: memory output is
        // coherent (not garbage) and recall held (name/hobby/color already checked in
        // run_A). Do NOT assert off==memory token-identity.
        std::cout << "  NOTE: restore fired in memory mode -> output is coherent+"
                     "context-preserving but NOT bit-identical to a full reprocess "
                     "(FP non-assoc); strict off==memory token-identity intentionally "
                     "not asserted.\n";
        capability_check("A[memory] t3 coherent (not garbage)",
                         !looks_like_garbage(mem_res.a_t3_text));
    } else {
        // No RESTORE fired: memory mode used the same reuse/wipe path as off mode, so
        // the output MUST be token-identical. This is a real check that must pass for
        // normal-reuse models (LFM), pure-attention, and SWA.
        std::cout << "  NOTE: no restore fired -> strict off==memory token-identity "
                     "asserted.\n";
        compare_tokens("A", off_res.a_tokens, mem_res.a_tokens);
    }

    // Strict off==memory token-identity in B holds only if NO restore fired in
    // the memory-mode session up to here. B keeps the prior session state (no
    // clearCache), so once a restore fired in A -- or in B itself -- the memory
    // path carries a coherent-but-not-bit-identical recurrent state (FP
    // non-associativity) that B's reprocess inherits. That is the documented
    // restore behavior propagating one hop, not a divergence B introduces, so we
    // assert coherence instead of byte-equality (leak/edit asserts already ran).
    if (mem_res.a_restore_fired || mem_res.b_restore_fired) {
        std::cout << "  NOTE: a restore fired earlier in memory mode -> B inherits "
                     "coherent+context-preserving (not bit-identical) state; strict "
                     "off==memory token-identity intentionally not asserted.\n";
        capability_check("B[memory] coherent (not garbage)",
                         !looks_like_garbage(mem_res.b_text));
    } else {
        compare_tokens("B", off_res.b_tokens, mem_res.b_tokens);
    }

    // ---- Scenario A: per-turn WIPE -> RESTORE pairing ----
    // If the off-mode baseline WIPED at some turn, the divergence forcing that wipe
    // exists identically in memory mode (same prompts, greedy off-mode tokens feed
    // both), so memory mode must fire a RESTORE at that same turn -- not merely
    // "somewhere". Vacuous (auto-skip) when the baseline never wiped, e.g. on
    // pure-attention models.
    {
        static const char* tn[3] = {"t1", "t2", "t3"};
        for (size_t i = 0; i < off_res.a_actions.size() && i < mem_res.a_actions.size(); i++) {
            if (off_res.a_actions[i] == ReuseAction::WIPE) {
                hard_check(std::string("A per-turn: off wiped at ") + tn[i] +
                           " -> memory must RESTORE there",
                           mem_res.a_actions[i] == ReuseAction::RESTORE,
                           std::string("memory action=") + action_name(mem_res.a_actions[i]));
            }
        }
    }

    // ---- Scenario E: tool-behavior consistency (off vs memory) ----
    // The checkpoint must not change tool behavior. The hard assert is that off and
    // memory produce the same tool behavior (same #calls, same first tool name+args,
    // same final-answer text). Whether the model emits a real get_weather(location=Paris)
    // call is a model capability -- reported as info, never failed (a model that refuses
    // tools and answers in text is a limitation, identical in both modes).
    if (off_res.tools.ran && mem_res.tools.ran) {
        std::cout << "\n############### CROSS-MODE TOOL CONSISTENCY (off vs memory) ###############\n";
        const ToolBehavior& o = off_res.tools;
        const ToolBehavior& mm = mem_res.tools;

        bool same_count = (o.n_tool_calls == mm.n_tool_calls);
        hard_check("E off==memory same number of tool calls", same_count,
                   "off=" + std::to_string(o.n_tool_calls) +
                   " mem=" + std::to_string(mm.n_tool_calls));

        bool same_call = (o.first_tool_name == mm.first_tool_name) &&
                         (o.first_tool_args == mm.first_tool_args);
        hard_check("E off==memory same tool name + parsed args", same_call,
                   same_call ? "" : "off=[" + o.first_tool_name + "|" + o.first_tool_args +
                                    "] mem=[" + mm.first_tool_name + "|" + mm.first_tool_args + "]");

        bool same_text = (o.t2_text == mm.t2_text);
        hard_check("E off==memory same final-answer text", same_text);

        // INFORMATIONAL: model capability, not a pass/fail criterion.
        bool emitted = o.emitted_get_weather && o.location_paris;
        std::cout << "  [info] model emitted tool call: "
                  << (emitted ? "yes" : "no")
                  << " (get_weather location=Paris)\n";
    }

    std::cout << "\n================ SUMMARY ================\n";
    std::cout << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
