// Standard headers BEFORE pybind11
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "eval_thread.h"
#include "batch_queue.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace alphago {

PersistentEvaluator::PersistentEvaluator() {
    thread_ = std::thread(&PersistentEvaluator::thread_main, this);
}

PersistentEvaluator::~PersistentEvaluator() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void PersistentEvaluator::start(BatchQueue* queue, const NetBatchEvalFn* eval_fn) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_ = queue;
        eval_fn_ = eval_fn;
        session_active_ = true;
        num_batches_.store(0);
        total_items_.store(0);
    }
    cv_.notify_one();
}

void PersistentEvaluator::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        session_active_ = false;
    }
    cv_.notify_one();

    // Wait for the evaluator to finish the current session
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return queue_ == nullptr; });
}

void PersistentEvaluator::thread_main() {
    while (true) {
        // Wait for work or shutdown
        BatchQueue* queue;
        const NetBatchEvalFn* eval_fn;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return session_active_ || shutdown_; });
            if (shutdown_) return;
            queue = queue_;
            eval_fn = eval_fn_;
        }

        // Process batches until session ends
        while (true) {
            auto batch = queue->collect_batch();

            if (batch.empty()) {
                // Check if session is still active
                std::lock_guard<std::mutex> lock(mutex_);
                if (!session_active_) {
                    // Final drain
                    batch = queue->collect_batch();
                    if (batch.empty()) {
                        // Signal that we're done
                        queue_ = nullptr;
                        eval_fn_ = nullptr;
                        cv_.notify_one();
                        break;
                    }
                } else {
                    continue;
                }
            }

            // Prepare inputs
            std::vector<const Board*> boards;
            std::vector<uint8_t> colors;
            boards.reserve(batch.size());
            colors.reserve(batch.size());
            for (auto& req : batch) {
                boards.push_back(&req.board);
                colors.push_back(req.color);
            }

            // Acquire GIL and call Python neural net
            std::vector<NetOutput> outputs;
            {
                py::gil_scoped_acquire acquire;
                outputs = (*eval_fn)(boards, colors);
            }

            num_batches_.fetch_add(1, std::memory_order_relaxed);
            total_items_.fetch_add(static_cast<int>(batch.size()), std::memory_order_relaxed);

            // Dispatch results
            for (size_t i = 0; i < batch.size(); i++) {
                batch[i].promise.set_value(std::move(outputs[i]));
            }
        }
    }
}

} // namespace alphago
