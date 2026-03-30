#pragma once

#include "board.h"
#include <vector>
#include <cmath>
#include <memory>
#include <limits>

namespace alphago {

struct MCTSNode {
    Move move;
    MCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<MCTSNode>> children;
    uint8_t color = 0; // who played this move
    int visit_count = 0;
    double total_value = 0.0;
    double prior = 0.0;

    double q_value() const {
        return visit_count == 0 ? 0.0 : total_value / visit_count;
    }

    double puct_score(double c_puct = 1.5) const {
        int parent_visits = parent ? parent->visit_count : 1;
        double exploration = c_puct * prior * std::sqrt(static_cast<double>(parent_visits))
                           / (1.0 + visit_count);
        return q_value() + exploration;
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
