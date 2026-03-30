#pragma once

#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include <utility>

namespace alphago {

constexpr int MAX_BOARD_SIZE = 19;
constexpr uint8_t EMPTY = 0;
constexpr uint8_t BLACK = 1;
constexpr uint8_t WHITE = 2;

inline uint8_t opponent(uint8_t color) { return color == BLACK ? WHITE : BLACK; }

// Move: row, col packed. -1,-1 = pass
struct Move {
    int row = -1;
    int col = -1;

    bool is_pass() const { return row < 0; }
    bool operator==(const Move& o) const { return row == o.row && col == o.col; }
};

inline Move pass_move() { return {-1, -1}; }
inline Move board_move(int r, int c) { return {r, c}; }

class Board {
public:
    explicit Board(int size = 13);

    Board(const Board& other);
    Board& operator=(const Board& other);

    int size() const { return size_; }
    uint8_t at(int r, int c) const { return grid_[r][c]; }
    uint64_t hash() const { return hash_; }

    // Core operations
    bool is_legal(uint8_t color, Move move) const;
    void play(uint8_t color, Move move);
    std::vector<Move> legal_moves(uint8_t color) const;

    // Scoring (Chinese rules)
    struct Score {
        double black;
        double white;
        int black_territory;
        int white_territory;
        int black_stones;
        int white_stones;
        int winner; // BLACK, WHITE, or 0 for draw
        double margin;
    };
    Score score() const;

    // State
    int captured_by(uint8_t color) const { return captured_[color]; }
    int move_count() const { return move_count_; }
    const uint8_t* grid_data() const { return &grid_[0][0]; }

    // Group finding (public for feature extraction)
    struct GroupInfo {
        std::vector<std::pair<int,int>> stones;
        std::vector<std::pair<int,int>> liberties;
    };
    GroupInfo find_group(int r, int c) const;
    int count_liberties(int r, int c) const;

    double komi = 6.5;

private:
    int size_;
    uint8_t grid_[MAX_BOARD_SIZE][MAX_BOARD_SIZE];
    uint64_t zobrist_table_[MAX_BOARD_SIZE][MAX_BOARD_SIZE][3];
    uint64_t hash_;
    std::unordered_set<uint64_t> position_history_;
    int captured_[3]; // indexed by color
    int move_count_;

    // Neighbor iteration (no allocation)
    struct Neighbors {
        std::pair<int,int> data[4];
        int count;
    };
    Neighbors neighbors(int r, int c) const;

    // (find_group and count_liberties are public, declared above)

    // Stone placement
    int place_stone(uint8_t color, int r, int c);
    bool is_suicide(uint8_t color, int r, int c) const;

    void toggle_hash(int r, int c, uint8_t color);
    void init_zobrist();
};

} // namespace alphago
