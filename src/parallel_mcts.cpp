/**
 * Multi-threaded MCTS with virtual loss and GPU batch queue.
 *
 * Uses a PersistentEvaluator thread to avoid per-call thread creation
 * (which leaks PyTorch per-thread CUDA state).
 */

#include <thread>
#include <atomic>
#include <random>
#include <cstring>

#include "parallel_mcts.h"
#include "mcts.h"
#include "mcts_node.h"
#include "node_pool.h"
#include "batch_queue.h"
#include "eval_thread.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace alphago {

MCTSResult MCTSSearch::search_parallel(
    const Board& board,
    uint8_t color_to_play,
    const NetBatchEvalFn& batch_eval_fn,
    double temperature
) {
    // Reset and reuse the persistent node pool
    node_pool_.reset();

    MCTSNode* root = node_pool_.alloc();

    // 1. Expand root (single-threaded, single GPU call)
    {
        py::gil_scoped_acquire acquire;
        std::vector<const Board*> boards = {&board};
        std::vector<uint8_t> colors = {color_to_play};
        auto outputs = batch_eval_fn(boards, colors);

        expand_node_pooled(root, board, color_to_play, outputs[0].policy, outputs[0].value, node_pool_);
    }

    add_dirichlet_noise(root);

    if (!root->is_expanded()) {
        return extract_result(root, board.size(), temperature);
    }

    // 2. Create batch queue
    BatchQueue queue(
        config_.min_batch_size,
        config_.max_batch_size,
        config_.batch_timeout_us
    );

    // 3. Lazy-create the persistent evaluator thread (created once, reused)
    if (!evaluator_) {
        evaluator_ = std::make_unique<PersistentEvaluator>();
    }

    // 4. Start evaluation session
    evaluator_->start(&queue, &batch_eval_fn);

    // 5. Simulation counter
    std::atomic<int> sims_remaining(config_.num_simulations);

    // Stats tracking
    std::vector<std::atomic<int>> thread_sim_counts(config_.num_threads);
    for (auto& c : thread_sim_counts) c.store(0);

    // 6. Launch worker threads
    std::vector<std::thread> workers;
    workers.reserve(config_.num_threads);

    for (int t = 0; t < config_.num_threads; t++) {
        workers.emplace_back([&, t]() {
            while (true) {
                int remaining = sims_remaining.fetch_sub(1, std::memory_order_acq_rel);
                if (remaining <= 0) break;

                MCTSNode* leaf = select_with_virtual_loss(root);

                uint8_t sim_color;
                Board sim_board = reconstruct_board(
                    leaf, root, board, color_to_play, sim_color
                );

                bool game_over = is_game_over(leaf, root);

                double value;

                if (game_over) {
                    auto s = sim_board.score();
                    value = (s.winner == BLACK) ? 1.0 : -1.0;
                } else if (!leaf->is_expanded()) {
                    NetOutput out = queue.submit(sim_board, sim_color);
                    expand_node_pooled_threadsafe(leaf, sim_board, sim_color, out.policy, out.value, node_pool_);
                    value = out.value;
                    if (sim_color == WHITE) value = -value;
                } else {
                    value = 0.0;
                }

                backup_with_virtual_loss(leaf, value);
                thread_sim_counts[t].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // 7. Join workers
    for (auto& w : workers) {
        w.join();
    }

    // 8. Stop evaluator session (blocks until draining is complete)
    queue.shutdown();
    evaluator_->stop();

    // 9. Extract result
    auto result = extract_result(root, board.size(), temperature);

    // Populate parallel stats
    result.parallel_stats.num_batches = evaluator_->num_batches();
    result.parallel_stats.total_batch_items = evaluator_->total_items();
    for (int t = 0; t < config_.num_threads; t++) {
        result.parallel_stats.sims_per_thread.push_back(thread_sim_counts[t].load());
    }

    return result;
}

} // namespace alphago
