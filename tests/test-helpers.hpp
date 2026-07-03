// Shared helpers for the C++ tests in this directory.
//
// Header-only and dependency-light: string/JSON utilities, the garbage/leak
// detectors, the pass/fail check() counters, and the one-turn completion driver
// that mirrors the real JSI call sequence (rewind -> initSampling ->
// beginCompletion -> loadPrompt -> doCompletion loop -> endCompletion).

#ifndef RNLLAMA_TEST_HELPERS_HPP
#define RNLLAMA_TEST_HELPERS_HPP

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "rn-llama.h"
#include "rn-completion.h"
#include "common.h"

namespace rntest {

using ReuseAction = rnllama::llama_rn_context_completion::ReuseAction;

// ------------------------------------------------------------------ strings

struct Msg { std::string role, content; };

inline std::string json_escape(const std::string& s) {
    // Full JSON string escaping. Weak models (e.g. mamba-130m) can emit raw
    // control characters; feeding that back as a chat message must not produce
    // a JSON string that fails to parse.
    static const char* hex = "0123456789abcdef";
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { // any other control char -> \u00XX
                    o += "\\u00";
                    o += hex[(c >> 4) & 0xF];
                    o += hex[c & 0xF];
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

// messages -> JSON array of {role, content}. Tool-result messages use role
// "tool" and are passed through verbatim.
inline std::string to_json(const std::vector<Msg>& msgs) {
    std::string j = "[";
    for (size_t i = 0; i < msgs.size(); i++) {
        j += "{\"role\":\"" + msgs[i].role + "\",\"content\":\"" +
             json_escape(msgs[i].content) + "\"}";
        if (i + 1 < msgs.size()) j += ",";
    }
    return j + "]";
}

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

inline bool contains(const std::string& hay, const std::string& needle) {
    return lower(hay).find(lower(needle)) != std::string::npos;
}

// Word-boundary match for NEGATIVE leakage checks: "ali"/"sam" must not match
// "alias"/"realize"/"same"/"salmon". Boundary = non-alphabetic neighbor.
inline bool contains_word(const std::string& hay, const std::string& needle) {
    const std::string h = lower(hay), n = lower(needle);
    for (size_t pos = 0; (pos = h.find(n, pos)) != std::string::npos; pos += n.size()) {
        const bool left_ok = pos == 0 || !std::isalpha((unsigned char)h[pos - 1]);
        const size_t end = pos + n.size();
        const bool right_ok = end >= h.size() || !std::isalpha((unsigned char)h[end]);
        if (left_ok && right_ok) return true;
    }
    return false;
}

// Garbage detector: empty, mostly-non-printable (mojibake/binary), OR
// degenerate word repetition (a corrupt KV state typically loops one token).
// Short valid answers like "4" or "Ali" are NOT garbage.
inline bool looks_like_garbage(const std::string& s) {
    if (s.empty()) return true;
    int printable = 0;
    for (unsigned char c : s) if (std::isprint(c) || c == '\n' || (c & 0x80)) printable++;
    if ((double)printable / s.size() < 0.70) return true;
    std::vector<std::string> words; std::string w;
    for (char c : s) {
        if (std::isspace((unsigned char)c)) { if (!w.empty()) { words.push_back(lower(w)); w.clear(); } }
        else w += c;
    }
    if (!w.empty()) words.push_back(lower(w));
    if (words.size() >= 8) {
        size_t best = 0;
        for (auto& a : words) {
            size_t cnt = std::count(words.begin(), words.end(), a);
            best = std::max(best, cnt);
        }
        if ((double)best / words.size() > 0.60) return true;
    }
    return false;
}

inline const char* action_name(ReuseAction a) {
    switch (a) {
        case ReuseAction::NONE:         return "none";
        case ReuseAction::COLD:         return "cold";
        case ReuseAction::NORMAL_REUSE: return "reuse";
        case ReuseAction::RESTORE:      return "restore";
        case ReuseAction::WIPE:         return "wipe";
    }
    return "?";
}

// ------------------------------------------------------------------ checks

inline int g_pass = 0, g_fail = 0;

inline void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::cout << (ok ? "  PASS " : "  FAIL ") << name;
    if (!detail.empty()) std::cout << "  [" << detail << "]";
    std::cout << "\n";
    if (ok) g_pass++; else g_fail++;
}

// ------------------------------------------------------------------ chat formatting

// Plain chat (no tools): render the prompt with the model's (or default) template.
inline std::string format_chat(rnllama::llama_rn_context& ctx, const std::vector<Msg>& msgs) {
    auto cp = ctx.getFormattedChatWithJinja(
        to_json(msgs), /*chat_template*/"", /*json_schema*/"", /*tools*/"",
        /*parallel_tool_calls*/false, /*tool_choice*/"",
        /*enable_thinking*/false, /*reasoning_format*/"none",
        /*add_generation_prompt*/true);
    return cp.prompt;
}

// ------------------------------------------------------------------ turn driver

struct TurnOptions {
    int n_predict = 32;
    bool kv_checkpoint = false;              // completion->kv_checkpoint_enabled
    std::vector<std::string> media;          // media paths (multimodal path)
    // Chat-format wiring for tool calling; defaults mean plain content chat.
    int chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string generation_prompt;
    std::string chat_parser;
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string> preserved_tokens;
};

struct TurnResult {
    std::string text;                        // generated text
    int reused = 0;                          // n_past right after loadPrompt
    ReuseAction action = ReuseAction::NONE;  // how loadPrompt resolved the cache
    int predicted = 0;
    std::vector<llama_token> token_ids;      // generated token ids (identity checks)
    double load_ms = 0.0;                    // loadPrompt time (reuse/restore/wipe decision + restore memcpy)
    double ttft_ms = 0.0;                    // loadPrompt start -> first generated token (prefill cost)
};

// Drive one completion turn through the real JSI flow.
inline TurnResult run_turn(rnllama::llama_rn_context& ctx, const std::string& prompt,
                           const TurnOptions& opt = {}) {
    ctx.params.prompt = prompt;
    ctx.params.n_predict = opt.n_predict;

    auto* c = ctx.completion;
    c->rewind();
    // Set after rewind; the stored snapshot itself persists across turns.
    c->kv_checkpoint_enabled = opt.kv_checkpoint;

    // Wire grammar/tool-call sampling state (matches JSIParams reset + apply).
    auto& sp = ctx.params.sampling;
    sp.grammar = {};
    sp.grammar_triggers.clear();
    sp.preserved_tokens.clear();
    sp.generation_prompt = opt.generation_prompt;
    sp.grammar_lazy = opt.grammar_lazy;
    if (!opt.grammar.empty()) {
        sp.grammar = {COMMON_GRAMMAR_TYPE_USER, opt.grammar};
    }
    sp.grammar_triggers = opt.grammar_triggers;
    for (const auto& t : opt.preserved_tokens) {
        auto ids = common_tokenize(ctx.ctx, t.c_str(), false, true);
        if (ids.size() == 1) sp.preserved_tokens.insert(ids[0]);
    }

    c->initSampling();
    c->beginCompletion(opt.chat_format, COMMON_REASONING_FORMAT_NONE,
                       opt.generation_prompt, opt.chat_parser);

    TurnResult r;
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point a) {
        return std::chrono::duration<double, std::milli>(clk::now() - a).count();
    };
    const auto t0 = clk::now();
    c->loadPrompt(opt.media);
    r.load_ms = ms_since(t0);
    r.reused = c->n_past;
    r.action = c->last_reuse_action;

    bool first = true;
    while (c->has_next_token && r.predicted < opt.n_predict) {
        auto t = c->doCompletion();
        // The prompt tail is decoded inside the first doCompletion (prefill), so the
        // first returned token marks time-to-first-token -- the user-facing prefill cost.
        if (first) { r.ttft_ms = ms_since(t0); first = false; }
        if (t.tok == -1) break;
        r.token_ids.push_back(t.tok);
        r.predicted++;
    }
    r.text = c->generated_text;
    c->endCompletion();
    return r;
}

} // namespace rntest

#endif // RNLLAMA_TEST_HELPERS_HPP
