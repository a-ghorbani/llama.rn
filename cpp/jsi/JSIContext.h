#pragma once
#include "JSINativeHeaders.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rnllama_jsi {
    // Registered contexts are reachable only through shared ownership: every task
    // holds a reference while it runs, and release destroys a context only once
    // it is the sole owner.
    template<typename T>
    class ContextManager {
    private:
        std::unordered_map<int, std::shared_ptr<T>> contextMap;
        std::mutex contextMutex;

    public:
        void add(int contextId, std::shared_ptr<T> context) {
            std::lock_guard<std::mutex> lock(contextMutex);
            contextMap[contextId] = std::move(context);
        }

        std::shared_ptr<T> take(int contextId) {
            std::lock_guard<std::mutex> lock(contextMutex);
            auto it = contextMap.find(contextId);
            if (it == contextMap.end()) {
                return nullptr;
            }
            auto context = std::move(it->second);
            contextMap.erase(it);
            return context;
        }

        std::shared_ptr<T> get(int contextId) {
            std::lock_guard<std::mutex> lock(contextMutex);
            auto it = contextMap.find(contextId);
            return (it != contextMap.end()) ? it->second : nullptr;
        }

        size_t size() {
            std::lock_guard<std::mutex> lock(contextMutex);
            return contextMap.size();
        }

        std::vector<std::shared_ptr<T>> takeAll() {
            std::lock_guard<std::mutex> lock(contextMutex);
            std::vector<std::shared_ptr<T>> items;
            items.reserve(contextMap.size());
            for (auto& entry : contextMap) {
                items.push_back(std::move(entry.second));
            }
            contextMap.clear();
            return items;
        }
    };

    using ContextRef = std::shared_ptr<rnllama::llama_rn_context>;

    extern ContextManager<rnllama::llama_rn_context> g_llamaContexts;
}
