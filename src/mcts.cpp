#include "mcts.h"
#include "eval_thread.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>

namespace alphago {

static thread_local std::mt19937 rng(42 + std::hash<std::thread::id>{}(std::this_thread::get_id()));

MCTSSearch::MCTSSearch(MCTSConfig config) : config_(config), node_pool_(8192) {
    // Lazy-create evaluator on first search_parallel call
}

void MCTSSearch::expand_node(
    MCTSNode* node,
    const Board& board,
    uint8_t color,
    const std::vector<float>& policy,
    float /*value*/
) {
    auto legal = board.legal_moves(color);
    int size = board.size();
    int num_moves = size * size + 1;

    // Build legal mask and renormalize policy
    std::vector<float> masked(num_moves, 0.0f);
    for (auto& m : legal) {
        int idx = m.is_pass() ? num_moves - 1 : m.row * size + m.col;
        if (idx < static_cast<int>(policy.size())) {
            masked[idx] = policy[idx];
        }
    }

    float sum = 0.0f;
    for (float v : masked) sum += v;
    if (sum > 1e-8f) {
        for (float& v : masked) v /= sum;
    } else {
        // Uniform over legal moves
        float uniform = 1.0f / static_cast<float>(legal.size());
        for (auto& m : legal) {
            int idx = m.is_pass() ? num_moves - 1 : m.row * size + m.col;
            masked[idx] = uniform;
        }
    }

    // Create children
    for (auto& m : legal) {
        int idx = m.is_pass() ? num_moves - 1 : m.row * size + m.col;
        auto child = std::make_unique<MCTSNode>();
        child->move = m;
        child->parent = node;
        child->color = color;
        child->prior = masked[idx];
        node->children.push_back(std::move(child));
    }
}

void MCTSSearch::expand_node_pooled(
    MCTSNode* node,
    const Board& board,
    uint8_t color,
    const std::vector<float>& policy,
    float /*value*/,
    NodePool& pool
) {
    auto legal = board.legal_moves(color);
    int size = board.size();
    int num_moves = size * size + 1;

    std::vector<float> masked(num_moves, 0.0f);
    for (auto& m : legal) {
        int idx = m.is_pass() ? num_moves - 1 : m.row * size + m.col;
        if (idx < static_cast<int>(policy.size())) {
            masked[idx] = policy[idx];
        }
    }

    float sum = 0.0f;
    for (float v : masked) sum += v;
    if (sum > 1e-8f) {
        for (float& v : masked) v /= sum;
    } else {
        float uniform = 1.0f / static_cast<float>(legal.size());
        for (auto& m : legal) {
            int idx = m.is_pass() ? num_moves - 1 : m.row * size + m.col;
            masked[idx] = uniform;
        }
    }

    for (auto& m : legal) {
        int idx = m.is_pass() ? num_moves - 1 : m.row * size + m.col;
        MCTSNode* child = pool.alloc();
        child->move = m;
        child->parent = node;
        child->color = color;
        child->prior = masked[idx];
        node->children_raw.push_back(child);
    }
}

void MCTSSearch::expand_node_pooled_threadsafe(
    MCTSNode* node,
    const Board& board,
    uint8_t color,
    const std::vector<float>& policy,
    float value,
    NodePool& pool
) {
    std::lock_guard<std::mutex> lock(tree_mutex_);
    if (node->is_expanded()) {
        return;
    }
    expand_node_pooled(node, board, color, policy, value, pool);
}

void MCTSSearch::add_dirichlet_noise(MCTSNode* root) {
    int n = root->num_children();
    if (n == 0 || config_.dirichlet_alpha <= 0) return;

    std::gamma_distribution<double> gamma(config_.dirichlet_alpha, 1.0);

    std::vector<double> noise(n);
    double noise_sum = 0.0;
    for (int i = 0; i < n; i++) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }
    for (int i = 0; i < n; i++) noise[i] /= noise_sum;

    double w = config_.dirichlet_weight;
    for (int i = 0; i < n; i++) {
        MCTSNode* child = root->child_at(i);
        child->prior = (1.0 - w) * child->prior + w * noise[i];
    }
}

MCTSNode* MCTSSearch::select(MCTSNode* root) {
    MCTSNode* current = root;
    while (current->is_expanded()) {
        current = current->best_child_puct(config_.c_puct);
    }
    return current;
}

void MCTSSearch::backup(MCTSNode* node, double value) {
    MCTSNode* current = node;
    while (current != nullptr) {
        current->visit_count++;
        if (current->color == BLACK) {
            current->total_value += value;
        } else if (current->color == WHITE) {
            current->total_value -= value;
        } else {
            current->total_value += value;
        }
        current = current->parent;
    }
}

// ── Virtual Loss Methods (for parallel search) ────────────────────

MCTSNode* MCTSSearch::select_with_virtual_loss(MCTSNode* root) {
    std::lock_guard<std::mutex> lock(tree_mutex_);
    MCTSNode* current = root;
    while (current->is_expanded()) {
        current->virtual_loss_count++;
        current = current->best_child_puct_vl(config_.c_puct, config_.virtual_loss_value);
    }
    current->virtual_loss_count++;
    return current;
}

void MCTSSearch::backup_with_virtual_loss(MCTSNode* node, double value) {
    std::lock_guard<std::mutex> lock(tree_mutex_);
    MCTSNode* current = node;
    while (current != nullptr) {
        current->virtual_loss_count--;
        current->visit_count++;
        if (current->color == BLACK) {
            current->total_value += value;
        } else if (current->color == WHITE) {
            current->total_value -= value;
        } else {
            current->total_value += value;
        }
        current = current->parent;
    }
}

void MCTSSearch::expand_node_threadsafe(
    MCTSNode* node,
    const Board& board,
    uint8_t color,
    const std::vector<float>& policy,
    float value
) {
    std::lock_guard<std::mutex> lock(tree_mutex_);
    if (node->is_expanded()) {
        return;
    }
    expand_node(node, board, color, policy, value);
}

Board MCTSSearch::reconstruct_board(
    MCTSNode* leaf,
    MCTSNode* root,
    const Board& root_board,
    uint8_t root_color,
    uint8_t& out_color
) {
    // Collect path from leaf to root
    std::vector<MCTSNode*> path;
    MCTSNode* cur = leaf;
    while (cur != root) {
        path.push_back(cur);
        cur = cur->parent;
    }

    // Use copy_light: skips position_history_ to avoid heap allocation.
    // MCTS simulation boards are short-lived and don't need full superko.
    Board board = root_board.copy_light();
    uint8_t color = root_color;
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; i--) {
        board.play(color, path[i]->move);
        color = opponent(color);
    }
    out_color = color;
    return board;
}

bool MCTSSearch::is_game_over(MCTSNode* leaf, MCTSNode* root) {
    // Two consecutive passes = game over
    if (leaf == root) return false;
    if (!leaf->move.is_pass()) return false;
    MCTSNode* prev = leaf->parent;
    if (prev == root) return false;
    // Find the node before leaf in the path
    // Actually, leaf->parent's move is what we need
    // leaf's move is pass, check if parent also played pass
    // But parent is the node whose child is leaf. The parent's move is the previous move.
    // Actually the parent's move was played by the opponent. Check if parent's move is pass.
    if (leaf->parent && leaf->parent != root && leaf->parent->move.is_pass()) return true;
    return false;
}

MCTSResult MCTSSearch::extract_result(MCTSNode* root, int board_size, double temperature) {
    int num_moves = board_size * board_size + 1;

    std::vector<float> visits_vec(num_moves, 0.0f);
    int nc = root->num_children();
    for (int i = 0; i < nc; i++) {
        MCTSNode* child = root->child_at(i);
        int idx = child->move.is_pass() ? num_moves - 1 : child->move.row * board_size + child->move.col;
        visits_vec[idx] = static_cast<float>(child->visit_count);
    }

    std::vector<float> policy_vec(num_moves, 0.0f);
    if (temperature < 1e-8) {
        // Deterministic
        int best = static_cast<int>(std::max_element(visits_vec.begin(), visits_vec.end()) - visits_vec.begin());
        policy_vec[best] = 1.0f;
    } else {
        float inv_temp = 1.0f / static_cast<float>(temperature);
        float sum = 0.0f;
        for (int i = 0; i < num_moves; i++) {
            policy_vec[i] = std::pow(visits_vec[i], inv_temp);
            sum += policy_vec[i];
        }
        if (sum > 0) {
            for (float& v : policy_vec) v /= sum;
        }
    }

    // Select move
    Move best_move = pass_move();
    if (temperature < 1e-8) {
        int best_idx = static_cast<int>(std::max_element(visits_vec.begin(), visits_vec.end()) - visits_vec.begin());
        if (best_idx == num_moves - 1) {
            best_move = pass_move();
        } else {
            best_move = board_move(best_idx / board_size, best_idx % board_size);
        }
    } else {
        // Sample from policy
        std::discrete_distribution<int> dist(policy_vec.begin(), policy_vec.end());
        int idx = dist(rng);
        if (idx == num_moves - 1) {
            best_move = pass_move();
        } else {
            best_move = board_move(idx / board_size, idx % board_size);
        }
    }

    return {best_move, policy_vec, root->visit_count};
}

MCTSResult MCTSSearch::search(
    const Board& board,
    uint8_t color_to_play,
    const NetEvalFn& eval_fn,
    double temperature
) {
    auto root = std::make_unique<MCTSNode>();

    // Expand root
    auto net_out = eval_fn(board, color_to_play);
    expand_node(root.get(), board, color_to_play, net_out.policy, net_out.value);
    add_dirichlet_noise(root.get());

    for (int sim = 0; sim < config_.num_simulations; sim++) {
        // Select
        MCTSNode* leaf = select(root.get());

        // Reconstruct board
        uint8_t sim_color;
        Board sim_board = reconstruct_board(leaf, root.get(), board, color_to_play, sim_color);

        // Check game over
        bool game_over = is_game_over(leaf, root.get());

        double value;
        if (game_over) {
            auto s = sim_board.score();
            value = (s.winner == BLACK) ? 1.0 : -1.0;
        } else if (!leaf->is_expanded()) {
            auto out = eval_fn(sim_board, sim_color);
            // Convert to black's perspective
            float v = out.value;
            if (sim_color == WHITE) v = -v;
            expand_node(leaf, sim_board, sim_color, out.policy, out.value);
            value = v;
        } else {
            value = 0.0;
        }

        backup(leaf, value);
    }

    return extract_result(root.get(), board.size(), temperature);
}

std::vector<MCTSResult> MCTSSearch::search_batch(
    const std::vector<Board>& boards,
    const std::vector<uint8_t>& colors,
    const NetBatchEvalFn& batch_eval_fn,
    double temperature
) {
    int K = static_cast<int>(boards.size());
    std::vector<std::unique_ptr<MCTSNode>> roots(K);

    // Batch-expand all roots
    std::vector<const Board*> root_boards(K);
    std::vector<uint8_t> root_colors(K);
    for (int i = 0; i < K; i++) {
        roots[i] = std::make_unique<MCTSNode>();
        root_boards[i] = &boards[i];
        root_colors[i] = colors[i];
    }

    auto root_outputs = batch_eval_fn(root_boards, root_colors);
    for (int i = 0; i < K; i++) {
        expand_node(roots[i].get(), boards[i], colors[i],
                    root_outputs[i].policy, root_outputs[i].value);
        add_dirichlet_noise(roots[i].get());
    }

    // Run simulations with batched evaluation
    for (int sim = 0; sim < config_.num_simulations; sim++) {
        // Select leaves for all games
        struct LeafInfo {
            MCTSNode* leaf;
            Board board;
            uint8_t color;
            bool game_over;
            int game_idx;
        };
        std::vector<LeafInfo> leaf_infos;
        std::vector<int> needs_eval; // indices into leaf_infos that need net eval

        for (int i = 0; i < K; i++) {
            MCTSNode* leaf = select(roots[i].get());
            uint8_t sim_color;
            Board sim_board = reconstruct_board(leaf, roots[i].get(), boards[i], colors[i], sim_color);
            bool game_over = is_game_over(leaf, roots[i].get());

            leaf_infos.push_back({leaf, std::move(sim_board), sim_color, game_over, i});

            if (!game_over && !leaf->is_expanded()) {
                needs_eval.push_back(static_cast<int>(leaf_infos.size()) - 1);
            }
        }

        // Batch evaluate leaves that need it
        std::vector<NetOutput> outputs;
        if (!needs_eval.empty()) {
            std::vector<const Board*> eval_boards;
            std::vector<uint8_t> eval_colors;
            for (int idx : needs_eval) {
                eval_boards.push_back(&leaf_infos[idx].board);
                eval_colors.push_back(leaf_infos[idx].color);
            }

            outputs = batch_eval_fn(eval_boards, eval_colors);

            for (int j = 0; j < static_cast<int>(needs_eval.size()); j++) {
                int idx = needs_eval[j];
                expand_node(leaf_infos[idx].leaf, leaf_infos[idx].board,
                           leaf_infos[idx].color, outputs[j].policy, outputs[j].value);
            }
        }

        // Compute backup values
        std::vector<double> leaf_values(K, 0.0);

        for (int i = 0; i < K; i++) {
            if (leaf_infos[i].game_over) {
                auto s = leaf_infos[i].board.score();
                leaf_values[i] = (s.winner == BLACK) ? 1.0 : -1.0;
            }
        }

        for (int j = 0; j < static_cast<int>(needs_eval.size()); j++) {
            int idx = needs_eval[j];
            float v = outputs[j].value;
            if (leaf_infos[idx].color == WHITE) v = -v;
            leaf_values[idx] = v;
        }

        // Backup all
        for (int i = 0; i < K; i++) {
            backup(leaf_infos[i].leaf, leaf_values[i]);
        }
    }

    // Extract results
    std::vector<MCTSResult> results;
    for (int i = 0; i < K; i++) {
        results.push_back(extract_result(roots[i].get(), boards[i].size(), temperature));
    }
    return results;
}

} // namespace alphago
