#include "JSITaskManager.h"

namespace rnllama_jsi {

TaskManager& TaskManager::getInstance() {
    static TaskManager instance;
    return instance;
}

void TaskManager::startTask(int contextId) {
    std::lock_guard<std::mutex> lock(mutex);
    activeTasks[contextId] += 1;
    totalTasks += 1;
}

void TaskManager::finishTask(int contextId) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = activeTasks.find(contextId);
    if (it != activeTasks.end()) {
        it->second -= 1;
        if (it->second <= 0) {
            activeTasks.erase(it);
        }
    }

    if (totalTasks > 0) {
        totalTasks -= 1;
    }
}

void TaskManager::beginShutdown() {
    shuttingDown.store(true, std::memory_order_relaxed);
}

void TaskManager::reset() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        activeTasks.clear();
        totalTasks = 0;
    }
    shuttingDown.store(false, std::memory_order_relaxed);
}

bool TaskManager::isShuttingDown() const {
    return shuttingDown.load(std::memory_order_relaxed);
}

TaskFinishGuard::TaskFinishGuard(int contextId, bool tracked)
    : contextId(contextId), tracked(tracked) {}

TaskFinishGuard::~TaskFinishGuard() {
    if (tracked) {
        TaskManager::getInstance().finishTask(contextId);
    }
}

} // namespace rnllama_jsi
