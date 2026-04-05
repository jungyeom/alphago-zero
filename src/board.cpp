#include "board.h"
#include <cstring>
#include <random>
#include <queue>
#include <algorithm>

namespace alphago {

Board::Board(int size, bool with_history) : size_(size), hash_(0), move_count_(0) {
    std::memset(grid_, 0, sizeof(grid_));
    captured_[0] = 0;
    captured_[1] = 0;
    captured_[2] = 0;
    track_history = with_history;
    init_zobrist();
    if (with_history) {
        position_history_.insert(hash_);
    }
}

Board::Board(const Board& other)
    : size_(other.size_)
    , hash_(other.hash_)
    , position_history_(other.position_history_)
    , move_count_(other.move_count_)
    , komi(other.komi)
{
    std::memcpy(grid_, other.grid_, sizeof(grid_));
    std::memcpy(zobrist_table_, other.zobrist_table_, sizeof(zobrist_table_));
    captured_[0] = other.captured_[0];
    captured_[1] = other.captured_[1];
    captured_[2] = other.captured_[2];
}

Board& Board::operator=(const Board& other) {
    if (this != &other) {
        size_ = other.size_;
        hash_ = other.hash_;
        position_history_ = other.position_history_;
        move_count_ = other.move_count_;
        komi = other.komi;
        std::memcpy(grid_, other.grid_, sizeof(grid_));
        std::memcpy(zobrist_table_, other.zobrist_table_, sizeof(zobrist_table_));
        captured_[0] = other.captured_[0];
        captured_[1] = other.captured_[1];
        captured_[2] = other.captured_[2];
    }
    return *this;
}

Board Board::copy_light() const {
    // Create with history disabled — position_history_ stays empty (zero heap alloc).
    Board b(size_, /*with_history=*/false);
    std::memcpy(b.grid_, grid_, sizeof(grid_));
    std::memcpy(b.zobrist_table_, zobrist_table_, sizeof(zobrist_table_));
    b.hash_ = hash_;
    b.captured_[0] = captured_[0];
    b.captured_[1] = captured_[1];
    b.captured_[2] = captured_[2];
    b.move_count_ = move_count_;
    b.komi = komi;
    return b;
}

void Board::init_zobrist() {
    // Fixed seed for reproducibility (matches Python's RandomState(42))
    std::mt19937_64 rng(42);
    for (int r = 0; r < MAX_BOARD_SIZE; r++) {
        for (int c = 0; c < MAX_BOARD_SIZE; c++) {
            zobrist_table_[r][c][0] = 0;
            zobrist_table_[r][c][1] = rng();
            zobrist_table_[r][c][2] = rng();
        }
    }
}

void Board::toggle_hash(int r, int c, uint8_t color) {
    hash_ ^= zobrist_table_[r][c][color];
}

Board::Neighbors Board::neighbors(int r, int c) const {
    Neighbors n;
    n.count = 0;
    if (r > 0)          n.data[n.count++] = {r - 1, c};
    if (r < size_ - 1)  n.data[n.count++] = {r + 1, c};
    if (c > 0)          n.data[n.count++] = {r, c - 1};
    if (c < size_ - 1)  n.data[n.count++] = {r, c + 1};
    return n;
}

Board::GroupInfo Board::find_group(int r, int c) const {
    GroupInfo info;
    uint8_t color = grid_[r][c];
    if (color == EMPTY) return info;

    bool visited[MAX_BOARD_SIZE][MAX_BOARD_SIZE] = {};
    bool lib_visited[MAX_BOARD_SIZE][MAX_BOARD_SIZE] = {};

    std::queue<std::pair<int,int>> queue;
    queue.push({r, c});
    visited[r][c] = true;
    info.stones.push_back({r, c});

    while (!queue.empty()) {
        auto [cr, cc] = queue.front();
        queue.pop();

        auto nbrs = neighbors(cr, cc);
        for (int i = 0; i < nbrs.count; i++) {
            auto [nr, nc] = nbrs.data[i];
            if (grid_[nr][nc] == EMPTY && !lib_visited[nr][nc]) {
                lib_visited[nr][nc] = true;
                info.liberties.push_back({nr, nc});
            } else if (grid_[nr][nc] == color && !visited[nr][nc]) {
                visited[nr][nc] = true;
                info.stones.push_back({nr, nc});
                queue.push({nr, nc});
            }
        }
    }

    return info;
}

int Board::count_liberties(int r, int c) const {
    return static_cast<int>(find_group(r, c).liberties.size());
}

bool Board::has_liberties(int r, int c, int min_libs) const {
    // BFS using stack-allocated arrays — zero heap allocation.
    uint8_t color = grid_[r][c];
    if (color == EMPTY) return true;

    bool visited[MAX_BOARD_SIZE][MAX_BOARD_SIZE] = {};
    bool lib_visited[MAX_BOARD_SIZE][MAX_BOARD_SIZE] = {};

    // Stack-allocated queue (max group size = board area)
    std::pair<int,int> queue_buf[MAX_BOARD_SIZE * MAX_BOARD_SIZE];
    int queue_head = 0, queue_tail = 0;

    visited[r][c] = true;
    queue_buf[queue_tail++] = {r, c};
    int lib_count = 0;

    while (queue_head < queue_tail) {
        auto [cr, cc] = queue_buf[queue_head++];
        auto nbrs = neighbors(cr, cc);
        for (int i = 0; i < nbrs.count; i++) {
            auto [nr, nc] = nbrs.data[i];
            if (grid_[nr][nc] == EMPTY && !lib_visited[nr][nc]) {
                lib_visited[nr][nc] = true;
                lib_count++;
                if (lib_count >= min_libs) return true;
            } else if (grid_[nr][nc] == color && !visited[nr][nc]) {
                visited[nr][nc] = true;
                queue_buf[queue_tail++] = {nr, nc};
            }
        }
    }
    return lib_count >= min_libs;
}

int Board::place_stone(uint8_t color, int r, int c) {
    grid_[r][c] = color;
    toggle_hash(r, c, color);

    int captured_count = 0;
    uint8_t opp = opponent(color);

    auto nbrs = neighbors(r, c);
    for (int i = 0; i < nbrs.count; i++) {
        auto [nr, nc] = nbrs.data[i];
        if (grid_[nr][nc] == opp) {
            // First check if the group has any liberties (stack-allocated, fast)
            if (!has_liberties(nr, nc, 1)) {
                // Group is captured — find stones to remove (need full find_group)
                auto group = find_group(nr, nc);
                captured_count += static_cast<int>(group.stones.size());
                for (auto [sr, sc] : group.stones) {
                    grid_[sr][sc] = EMPTY;
                    toggle_hash(sr, sc, opp);
                }
            }
        }
    }

    return captured_count;
}

bool Board::is_suicide(uint8_t color, int r, int c) const {
    // Temporarily place stone (const_cast for the check, then undo)
    auto* self = const_cast<Board*>(this);
    self->grid_[r][c] = color;

    uint8_t opp = opponent(color);
    bool captures = false;
    auto nbrs = neighbors(r, c);
    for (int i = 0; i < nbrs.count; i++) {
        auto [nr, nc] = nbrs.data[i];
        if (grid_[nr][nc] == opp) {
            // Use has_liberties (stack-allocated BFS, no heap alloc)
            if (!has_liberties(nr, nc, 1)) {
                captures = true;
                break;
            }
        }
    }

    // Check own liberties (stack-allocated BFS)
    bool own_has_libs = has_liberties(r, c, 1);

    self->grid_[r][c] = EMPTY;

    return !captures && !own_has_libs;
}

bool Board::is_legal(uint8_t color, Move move) const {
    if (move.is_pass()) return true;

    int r = move.row, c = move.col;
    if (r < 0 || r >= size_ || c < 0 || c >= size_) return false;
    if (grid_[r][c] != EMPTY) return false;

    // Fast path: if any adjacent cell is empty, it's not suicide
    // and we only need the superko check
    bool has_adjacent_liberty = false;
    auto nbrs = neighbors(r, c);
    for (int i = 0; i < nbrs.count; i++) {
        if (grid_[nbrs.data[i].first][nbrs.data[i].second] == EMPTY) {
            has_adjacent_liberty = true;
            break;
        }
    }

    if (!has_adjacent_liberty) {
        // Need full suicide check
        if (is_suicide(color, r, c)) return false;
    }

    // Skip superko check for simulation boards (no history tracking).
    // This avoids the expensive simulate-check-undo cycle AND
    // the vector<pair> heap allocation for captured_stones.
    if (!track_history) {
        return true;  // passed bounds, occupied, and suicide checks
    }

    // Superko check: simulate in-place, check hash, undo
    auto* self = const_cast<Board*>(this);
    self->grid_[r][c] = color;
    self->toggle_hash(r, c, color);

    uint8_t opp = opponent(color);
    std::vector<std::pair<int,int>> captured_stones;

    for (int i = 0; i < nbrs.count; i++) {
        auto [nr, nc] = nbrs.data[i];
        if (self->grid_[nr][nc] == opp) {
            auto group = find_group(nr, nc);
            if (group.liberties.empty()) {
                for (auto [sr, sc] : group.stones) {
                    self->grid_[sr][sc] = EMPTY;
                    self->toggle_hash(sr, sc, opp);
                    captured_stones.push_back({sr, sc});
                }
            }
        }
    }

    uint64_t result_hash = self->hash_;
    bool is_superko = position_history_.count(result_hash) > 0;

    // Undo
    for (auto [sr, sc] : captured_stones) {
        self->grid_[sr][sc] = opp;
        self->toggle_hash(sr, sc, opp);
    }
    self->grid_[r][c] = EMPTY;
    self->toggle_hash(r, c, color);

    return !is_superko;
}

void Board::play(uint8_t color, Move move) {
    if (move.is_pass()) {
        move_count_++;
        return;
    }

    int captured = place_stone(color, move.row, move.col);
    captured_[color] += captured;
    if (track_history) {
        position_history_.insert(hash_);
    }
    move_count_++;
}

std::vector<Move> Board::legal_moves(uint8_t color) const {
    std::vector<Move> moves;
    moves.reserve(size_ * size_ + 1);

    for (int r = 0; r < size_; r++) {
        for (int c = 0; c < size_; c++) {
            if (is_legal(color, board_move(r, c))) {
                moves.push_back(board_move(r, c));
            }
        }
    }
    moves.push_back(pass_move()); // pass always legal
    return moves;
}

Board::Score Board::score() const {
    bool visited[MAX_BOARD_SIZE][MAX_BOARD_SIZE] = {};
    int territory[3] = {};

    for (int r = 0; r < size_; r++) {
        for (int c = 0; c < size_; c++) {
            if (visited[r][c] || grid_[r][c] != EMPTY) continue;

            // BFS to find empty region
            std::vector<std::pair<int,int>> region;
            std::unordered_set<uint8_t> borders;
            std::queue<std::pair<int,int>> queue;
            queue.push({r, c});
            visited[r][c] = true;

            while (!queue.empty()) {
                auto [cr, cc] = queue.front();
                queue.pop();
                region.push_back({cr, cc});

                auto nbrs = neighbors(cr, cc);
                for (int i = 0; i < nbrs.count; i++) {
                    auto [nr, nc] = nbrs.data[i];
                    uint8_t cell = grid_[nr][nc];
                    if (cell != EMPTY) {
                        borders.insert(cell);
                    } else if (!visited[nr][nc]) {
                        visited[nr][nc] = true;
                        queue.push({nr, nc});
                    }
                }
            }

            if (borders.size() == 1) {
                uint8_t owner = *borders.begin();
                territory[owner] += static_cast<int>(region.size());
            }
        }
    }

    int black_stones = 0, white_stones = 0;
    for (int r = 0; r < size_; r++)
        for (int c = 0; c < size_; c++) {
            if (grid_[r][c] == BLACK) black_stones++;
            else if (grid_[r][c] == WHITE) white_stones++;
        }

    Score s;
    s.black_stones = black_stones;
    s.white_stones = white_stones;
    s.black_territory = territory[BLACK];
    s.white_territory = territory[WHITE];
    s.black = black_stones + territory[BLACK];
    s.white = white_stones + territory[WHITE] + komi;
    s.margin = std::abs(s.black - s.white);
    if (s.black > s.white) s.winner = BLACK;
    else if (s.white > s.black) s.winner = WHITE;
    else s.winner = 0;

    return s;
}

} // namespace alphago
