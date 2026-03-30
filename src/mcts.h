#pragma once

#include "board.h"
#include "mcts_node.h"
#include <functional>
#include <vector>

namespace alphago {

// Callback type for neural net evaluation.
// Python passes a function that takes (board_features, color) and returns (policy, value).
// This keeps the neural net in Python/PyTorch while MCTS runs in C++.
struct NetOutput {
    std::vector<float> policy; // size: board_size^2 + 1
    float value;               // from black's perspective
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
};

struct MCTSResult {
    Move best_move;
    std::vector<float> policy_vec; // visit-count based policy
    int total_visits;
};

class MCTSSearch {
public:
    MCTSSearch(MCTSConfig config = {});

    // Single-game search
    MCTSResult search(
        const Board& board,
        uint8_t color_to_play,
        const NetEvalFn& eval_fn,
        double temperature = 1.0
    );

    // Batched search for multiple games
    std::vector<MCTSResult> search_batch(
        const std::vector<Board>& boards,
        const std::vector<uint8_t>& colors,
        const NetBatchEvalFn& batch_eval_fn,
        double temperature = 1.0
    );

private:
    MCTSConfig config_;

    void expand_node(
        MCTSNode* node,
        const Board& board,
        uint8_t color,
        const std::vector<float>& policy,
        float value
    );

    void add_dirichlet_noise(MCTSNode* root);

    MCTSNode* select(MCTSNode* root);

    void backup(MCTSNode* node, double value);

    // Reconstruct board state at a node by replaying moves from root
    Board reconstruct_board(
        MCTSNode* leaf,
        MCTSNode* root,
        const Board& root_board,
        uint8_t root_color,
        uint8_t& out_color
    );

    bool is_game_over(MCTSNode* leaf, MCTSNode* root);

    MCTSResult extract_result(MCTSNode* root, int board_size, double temperature);
};

} // namespace alphago
