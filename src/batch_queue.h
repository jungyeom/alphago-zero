#pragma once

#include "board.h"
#include "mcts.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <chrono>
#include <vector>

namespace alphago {

struct EvalRequest {
    Board board;
    uint8_t color;
    std::promise<NetOutput> promise;

    // Move-only (promise is not copyable)
    EvalRequest() = default;
    EvalRequest(EvalRequest&&) = default;
    EvalRequest& operator=(EvalRequest&&) = default;
};

class BatchQueue {
public:
    BatchQueue(int min_batch_size, int max_batch_size, int timeout_us)
        : min_batch_size_(min_batch_size)
        , max_batch_size_(max_batch_size)
        , timeout_(timeout_us)
        , shutdown_(false)
    {}

    // Called by worker threads. Blocks until result is ready.
    NetOutput submit(const Board& board, uint8_t color) {
        EvalRequest req;
        req.board = board.copy_light();  // lightweight copy (skip position_history_)
        req.color = color;
        auto future = req.promise.get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(req));
        }
        cv_producer_.notify_one();

        // Block until the evaluator processes our request
        return future.get();
    }

    // Called by the evaluator thread. Blocks until batch is ready or timeout.
    // Returns empty vector only on shutdown with empty queue.
    std::vector<EvalRequest> collect_batch() {
        std::vector<EvalRequest> batch;

        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for at least 1 item or shutdown
        cv_producer_.wait_for(lock, timeout_, [this] {
            return !queue_.empty() || shutdown_;
        });

        if (queue_.empty()) {
            return batch;  // empty — either timeout with nothing or shutdown
        }

        // If we have items but less than min_batch_size, wait a bit more
        // for more items to arrive (but not too long)
        if (static_cast<int>(queue_.size()) < min_batch_size_ && !shutdown_) {
            auto short_wait = std::chrono::microseconds(50);
            cv_producer_.wait_for(lock, short_wait, [this] {
                return static_cast<int>(queue_.size()) >= min_batch_size_ || shutdown_;
            });
        }

        // Drain up to max_batch_size
        int count = std::min(max_batch_size_, static_cast<int>(queue_.size()));
        for (int i = 0; i < count; i++) {
            batch.push_back(std::move(queue_.front()));
            queue_.pop();
        }

        return batch;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_producer_.notify_all();
    }

    int pending_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(queue_.size());
    }

private:
    std::queue<EvalRequest> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_producer_;
    int min_batch_size_;
    int max_batch_size_;
    std::chrono::microseconds timeout_;
    bool shutdown_;
};

} // namespace alphago
