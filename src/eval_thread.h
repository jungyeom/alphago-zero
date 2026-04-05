#pragma once

#include "mcts.h"
#include "batch_queue.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

namespace alphago {

/**
 * Persistent evaluator thread for MCTS search_parallel.
 *
 * Problem: Creating a new evaluator thread per MCTS call causes PyTorch
 * to allocate per-thread CUDA state (streams, caching allocator TLS) that
 * is never freed, leaking ~0.25MB per call.
 *
 * Solution: Keep a single long-lived evaluator thread that is reused
 * across search_parallel calls. The thread sleeps between calls and
 * wakes when new work is submitted.
 */
class PersistentEvaluator {
public:
    PersistentEvaluator();
    ~PersistentEvaluator();

    /// Start a new evaluation session. The evaluator thread will
    /// collect batches from `queue` and call `eval_fn` until stop() is called.
    void start(BatchQueue* queue, const NetBatchEvalFn* eval_fn);

    /// Stop the current session (blocks until evaluator finishes draining).
    void stop();

    /// Stats from the last session.
    int num_batches() const { return num_batches_.load(); }
    int total_items() const { return total_items_.load(); }

private:
    void thread_main();

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;

    // Session state (protected by mutex_)
    BatchQueue* queue_ = nullptr;
    const NetBatchEvalFn* eval_fn_ = nullptr;
    bool session_active_ = false;
    bool shutdown_ = false;

    // Stats
    std::atomic<int> num_batches_{0};
    std::atomic<int> total_items_{0};
};

/**
 * Shared evaluator: one BatchQueue + one PersistentEvaluator shared across
 * multiple concurrent MCTSSearch instances.  All games' worker threads
 * submit to the same queue → larger GPU batches → better utilization.
 */
class SharedEvaluator {
public:
    SharedEvaluator(int min_batch, int max_batch, int timeout_us);

    /// Start processing. Call before launching any search_parallel_shared.
    void start(const NetBatchEvalFn& eval_fn);

    /// Stop processing. Call after all searches have joined.
    void stop();

    BatchQueue& queue() { return *queue_; }
    int num_batches() const { return evaluator_->num_batches(); }
    int total_items() const { return evaluator_->total_items(); }

private:
    std::unique_ptr<BatchQueue> queue_;
    std::unique_ptr<PersistentEvaluator> evaluator_;
    NetBatchEvalFn active_fn_;  // stored by value to keep Python callable alive
};

} // namespace alphago
