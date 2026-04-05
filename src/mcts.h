#pragma once

#include "board.h"
#include "mcts_node.h"
#include "node_pool.h"
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

// Forward declare to avoid include cycle
namespace alphago { class PersistentEvaluator; class BatchQueue; }

namespace alphago {

// Callback type for neural net evaluation.
struct NetOutput {
    std::vector<float> policy; // size: board_size^2 + 1
    float value;               // from current player's perspective
};

using NetEvalFn = std::function<NetOutput(const Board& board, uint8_t color)>;

// Batch version: evaluate multiple positions at once
using NetBatchEvalFn = std::function<std::vector<NetOutput>(
    const std::vector<const Board*>& boards,
    const std::vector<uint8_t>& colors
)>;

struct MCTSConfig {
    int num_simulations = 200;
    double c_puct = 1.5;
    double dirichlet_alpha = 0.1;
    double dirichlet_weight = 0.25;

    // Parallel search settings
    int num_threads = 4;
    int min_batch_size = 4;
    int max_batch_size = 16;
    int batch_timeout_us = 100;
    double virtual_loss_value = 1.0;
};

struct ParallelStats {
    std::vector<int> sims_per_thread;  // simulations each worker completed
    int num_batches = 0;               // total batches sent to evaluator
    int total_batch_items = 0;         // total items evaluated across batches
};

struct MCTSResult {
    Move best_move;
    std::vector<float> policy_vec;
    int total_visits;
    ParallelStats parallel_stats;      // populated only by search_parallel
};

class MCTSSearch {
public:
    MCTSSearch(MCTSConfig config = {});

    // Single-threaded search (existing)
    MCTSResult search(
        const Board& board,
        uint8_t color_to_play,
        const NetEvalFn& eval_fn,
        double temperature = 1.0
    );

    // Batched search for multiple games (existing)
    std::vector<MCTSResult> search_batch(
        const std::vector<Board>& boards,
        const std::vector<uint8_t>& colors,
        const NetBatchEvalFn& batch_eval_fn,
        double temperature = 1.0
    );

    // Multi-threaded search with virtual loss and GPU batch queue
    MCTSResult search_parallel(
        const Board& board,
        uint8_t color_to_play,
        const NetBatchEvalFn& batch_eval_fn,
        double temperature = 1.0
    );

    // Public helpers used by parallel_mcts.cpp
    void expand_node(
        MCTSNode* node,
        const Board& board,
        uint8_t color,
        const std::vector<float>& policy,
        float value
    );

    // Pool-based expand: allocates children from the pool instead of malloc
    void expand_node_pooled(
        MCTSNode* node,
        const Board& board,
        uint8_t color,
        const std::vector<float>& policy,
        float value,
        NodePool& pool
    );

    void add_dirichlet_noise(MCTSNode* root);

    Board reconstruct_board(
        MCTSNode* leaf,
        MCTSNode* root,
        const Board& root_board,
        uint8_t root_color,
        uint8_t& out_color
    );

    bool is_game_over(MCTSNode* leaf, MCTSNode* root);

    MCTSResult extract_result(MCTSNode* root, int board_size, double temperature);

    // Thread-safe variants for parallel search
    MCTSNode* select_with_virtual_loss(MCTSNode* root);
    void backup_with_virtual_loss(MCTSNode* node, double value);
    void expand_node_threadsafe(
        MCTSNode* node,
        const Board& board,
        uint8_t color,
        const std::vector<float>& policy,
        float value
    );
    void expand_node_pooled_threadsafe(
        MCTSNode* node,
        const Board& board,
        uint8_t color,
        const std::vector<float>& policy,
        float value,
        NodePool& pool
    );

    // Shared-queue variant: caller manages the evaluator and queue.
    // All concurrent games submit to the same queue → larger GPU batches.
    MCTSResult search_parallel_shared(
        const Board& board,
        uint8_t color_to_play,
        BatchQueue& shared_queue,
        double temperature = 1.0
    );

private:
    MCTSConfig config_;
    NodePool node_pool_;  // Reusable pool for search_parallel
    std::unique_ptr<PersistentEvaluator> evaluator_;  // Reusable evaluator thread
    std::mutex tree_mutex_;  // Per-instance mutex (not global!) for parallel tree ops

    MCTSNode* select(MCTSNode* root);
    void backup(MCTSNode* node, double value);
};

} // namespace alphago
