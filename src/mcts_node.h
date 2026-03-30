#pragma once

#include "board.h"
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <atomic>
#include <mutex>

namespace alphago {

constexpr double DEFAULT_VIRTUAL_LOSS_VALUE = 1.0;

struct MCTSNode {
    Move move;
    MCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<MCTSNode>> children;
    uint8_t color = 0;
    double prior = 0.0;

    // Use plain types — thread safety is handled externally via mutexes
    // in the parallel search path. Single-threaded search uses these directly.
    int visit_count = 0;
    double total_value = 0.0;
    int virtual_loss_count = 0;

    // Standard Q value (single-threaded, no virtual loss)
    double q_value() const {
        if (visit_count == 0) return 0.0;
        return total_value / visit_count;
    }

    // Q value accounting for virtual loss (parallel search)
    double q_value_vl(double vloss_value = DEFAULT_VIRTUAL_LOSS_VALUE) const {
        int total = visit_count + virtual_loss_count;
        if (total == 0) return 0.0;
        return (total_value - virtual_loss_count * vloss_value) / total;
    }

    // Standard PUCT (single-threaded)
    double puct_score(double c_puct = 1.5) const {
        int parent_visits = parent ? parent->visit_count : 1;
        double exploration = c_puct * prior * std::sqrt(static_cast<double>(parent_visits))
                           / (1.0 + visit_count);
        return q_value() + exploration;
    }

    // PUCT with virtual loss (parallel search)
    double puct_score_vl(double c_puct = 1.5, double vloss_value = DEFAULT_VIRTUAL_LOSS_VALUE) const {
        int parent_visits = parent ? (parent->visit_count + parent->virtual_loss_count) : 1;
        if (parent_visits < 1) parent_visits = 1;
        int total = visit_count + virtual_loss_count;
        double exploration = c_puct * prior * std::sqrt(static_cast<double>(parent_visits))
                           / (1.0 + total);
        return q_value_vl(vloss_value) + exploration;
    }

    bool is_expanded() const { return !children.empty(); }
    bool is_root() const { return parent == nullptr; }

    MCTSNode* best_child_puct(double c_puct = 1.5) const {
        MCTSNode* best = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        for (auto& child : children) {
            double s = child->puct_score(c_puct);
            if (s > best_score) {
                best_score = s;
                best = child.get();
            }
        }
        return best;
    }

    MCTSNode* best_child_puct_vl(double c_puct = 1.5, double vloss_value = DEFAULT_VIRTUAL_LOSS_VALUE) const {
        MCTSNode* best = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        for (auto& child : children) {
            double s = child->puct_score_vl(c_puct, vloss_value);
            if (s > best_score) {
                best_score = s;
                best = child.get();
            }
        }
        return best;
    }

    MCTSNode* most_visited_child() const {
        MCTSNode* best = nullptr;
        int best_visits = -1;
        for (auto& child : children) {
            if (child->visit_count > best_visits) {
                best_visits = child->visit_count;
                best = child.get();
            }
        }
        return best;
    }
};

} // namespace alphago
