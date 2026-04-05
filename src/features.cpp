#include "go_features.h"
#include <cstring>

namespace alphago {

BoardFeatures board_to_features(
    const Board& board,
    uint8_t color_to_play,
    Move last_move
) {
    BoardFeatures f;
    f.board_size = board.size();
    std::memset(f.data, 0, sizeof(f.data));

    int size = board.size();
    uint8_t opp = opponent(color_to_play);

    // Planes 0-2: stones and empty
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            uint8_t cell = board.at(r, c);
            if (cell == color_to_play) f.data[0][r][c] = 1.0f;
            else if (cell == opp)      f.data[1][r][c] = 1.0f;
            else                       f.data[2][r][c] = 1.0f;
        }
    }

    // Plane 3: last move
    if (!last_move.is_pass()) {
        f.data[3][last_move.row][last_move.col] = 1.0f;
    }

    // Plane 4: color to play
    if (color_to_play == BLACK) {
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++)
                f.data[4][r][c] = 1.0f;
    }

    // Planes 5-7: liberty features
    // Use the board's public interface to count liberties
    bool visited[MAX_BOARD_SIZE][MAX_BOARD_SIZE] = {};
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            if (board.at(r, c) == EMPTY || visited[r][c]) continue;

            int lib_count = board.count_liberties(r, c);

            // Mark all stones in this group via simple flood fill
            int plane = (lib_count == 1) ? 5 : (lib_count == 2) ? 6 : 7;

            // BFS to find connected stones of same color
            uint8_t color = board.at(r, c);
            std::vector<std::pair<int,int>> stack;
            stack.push_back({r, c});
            visited[r][c] = true;

            while (!stack.empty()) {
                auto [cr, cc] = stack.back();
                stack.pop_back();
                f.data[plane][cr][cc] = 1.0f;

                // Check 4 neighbors
                int dr[] = {-1, 1, 0, 0};
                int dc[] = {0, 0, -1, 1};
                for (int d = 0; d < 4; d++) {
                    int nr = cr + dr[d], nc = cc + dc[d];
                    if (nr >= 0 && nr < size && nc >= 0 && nc < size
                        && !visited[nr][nc] && board.at(nr, nc) == color) {
                        visited[nr][nc] = true;
                        stack.push_back({nr, nc});
                    }
                }
            }
        }
    }

    return f;
}

void apply_symmetry(
    const float* features_in,
    const float* policy_in,
    float* features_out,
    float* policy_out,
    int channels,
    int board_size,
    int sym_index
) {
    int N = board_size;
    int board_cells = N * N;

    auto transform_plane = [&](const float* in, float* out) {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                int sr = r, sc = c;

                if (sym_index >= 4) {
                    sc = N - 1 - sc;
                }

                int rotations = sym_index % 4;
                for (int rot = 0; rot < rotations; rot++) {
                    int tmp = sr;
                    sr = sc;
                    sc = N - 1 - tmp;
                }

                out[sr * N + sc] = in[r * N + c];
            }
        }
    };

    for (int ch = 0; ch < channels; ch++) {
        transform_plane(
            features_in + ch * board_cells,
            features_out + ch * board_cells
        );
    }

    float board_policy_in[MAX_BOARD_SIZE * MAX_BOARD_SIZE];
    float board_policy_out[MAX_BOARD_SIZE * MAX_BOARD_SIZE];

    std::memcpy(board_policy_in, policy_in, board_cells * sizeof(float));
    transform_plane(board_policy_in, board_policy_out);

    std::memcpy(policy_out, board_policy_out, board_cells * sizeof(float));
    policy_out[board_cells] = policy_in[board_cells];
}

} // namespace alphago
