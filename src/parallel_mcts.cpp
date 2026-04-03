/**
 * Multi-threaded MCTS with virtual loss and GPU batch queue.
 *
 * Architecture:
 *   N worker threads explore the same MCTS tree concurrently.
 *   Each worker: select (with virtual loss) → reconstruct board →
 *                submit to BatchQueue → wait → expand → backup.
 *   1 evaluator thread: collect batch → acquire GIL →
 *                       call Python batch_eval_fn → release GIL → dispatch results.
 */

// Standard headers BEFORE pybind11 to avoid cmath/random ambiguity on GCC 11
#include <thread>
#include <atomic>
#include <random>
#include <cstring>

#include "parallel_mcts.h"
#include "mcts.h"
#include "mcts_node.h"
#include "batch_queue.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace alphago {

MCTSResult MCTSSearch::search_parallel(
    const Board& board,
    uint8_t color_to_play,
    const NetBatchEvalFn& batch_eval_fn,
    double temperature
) {
    auto root = std::make_unique<MCTSNode>();

    // 1. Expand root (single-threaded, single GPU call)
    //    GIL is already released by the bindings wrapper,
    //    but batch_eval_fn handles GIL acquisition internally for this initial call.
    {
        py::gil_scoped_acquire acquire;
        std::vector<const Board*> boards = {&board};
        std::vector<uint8_t> colors = {color_to_play};
        auto outputs = batch_eval_fn(boards, colors);

        float root_value = outputs[0].value;
        if (color_to_play == WHITE) root_value = -root_value;

        expand_node(root.get(), board, color_to_play, outputs[0].policy, outputs[0].value);
    }

    add_dirichlet_noise(root.get());

    if (root->children.empty()) {
        return extract_result(root.get(), board.size(), temperature);
    }

    // 2. Create batch queue
    BatchQueue queue(
        config_.min_batch_size,
        config_.max_batch_size,
        config_.batch_timeout_us
    );

    // 3. Simulation counter (workers decrement atomically)
    std::atomic<int> sims_remaining(config_.num_simulations);
    std::atomic<bool> workers_done(false);

    // Stats tracking
    std::vector<std::atomic<int>> thread_sim_counts(config_.num_threads);
    for (auto& c : thread_sim_counts) c.store(0);
    std::atomic<int> num_batches(0);
    std::atomic<int> total_batch_items(0);

    // 4. Launch evaluator thread
    //    This is the ONLY thread that calls Python (acquires GIL).
    std::thread evaluator([&]() {
        while (true) {
            auto batch = queue.collect_batch();

            if (batch.empty()) {
                // Empty batch = shutdown or timeout with nothing pending
                if (workers_done.load(std::memory_order_acquire)) {
                    // Final drain: process any remaining items
                    batch = queue.collect_batch();
                    if (batch.empty()) break;
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
                outputs = batch_eval_fn(boards, colors);
            }

            // Track batch stats
            num_batches.fetch_add(1, std::memory_order_relaxed);
            total_batch_items.fetch_add(static_cast<int>(batch.size()), std::memory_order_relaxed);

            // Dispatch results to waiting worker threads
            for (size_t i = 0; i < batch.size(); i++) {
                batch[i].promise.set_value(std::move(outputs[i]));
            }
        }
    });

    // 5. Launch worker threads
    std::vector<std::thread> workers;
    workers.reserve(config_.num_threads);

    for (int t = 0; t < config_.num_threads; t++) {
        workers.emplace_back([&, t]() {
            while (true) {
                int remaining = sims_remaining.fetch_sub(1, std::memory_order_acq_rel);
                if (remaining <= 0) break;

                // SELECT with virtual loss
                MCTSNode* leaf = select_with_virtual_loss(root.get());

                // RECONSTRUCT board at leaf (thread-local copy)
                uint8_t sim_color;
                Board sim_board = reconstruct_board(
                    leaf, root.get(), board, color_to_play, sim_color
                );

                // Check game over
                bool game_over = is_game_over(leaf, root.get());

                double value;

                if (game_over) {
                    auto s = sim_board.score();
                    value = (s.winner == BLACK) ? 1.0 : -1.0;
                } else if (!leaf->is_expanded()) {
                    // Submit to batch queue and wait for GPU result
                    NetOutput out = queue.submit(sim_board, sim_color);

                    // Expand (thread-safe: only first thread expands)
                    expand_node_threadsafe(leaf, sim_board, sim_color, out.policy, out.value);

                    // Value from black's perspective
                    value = out.value;
                    if (sim_color == WHITE) value = -value;
                } else {
                    // Already expanded by another thread — use a neutral value
                    value = 0.0;
                }

                // BACKUP with virtual loss removal
                backup_with_virtual_loss(leaf, value);
                thread_sim_counts[t].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // 6. Join workers
    for (auto& w : workers) {
        w.join();
    }

    // 7. Signal evaluator to stop and join
    workers_done.store(true, std::memory_order_release);
    queue.shutdown();
    evaluator.join();

    // 8. Extract result
    auto result = extract_result(root.get(), board.size(), temperature);

    // Populate parallel stats
    result.parallel_stats.num_batches = num_batches.load();
    result.parallel_stats.total_batch_items = total_batch_items.load();
    for (int t = 0; t < config_.num_threads; t++) {
        result.parallel_stats.sims_per_thread.push_back(thread_sim_counts[t].load());
    }

    return result;
}

} // namespace alphago
