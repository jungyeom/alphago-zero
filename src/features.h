#pragma once

#include "board.h"
#include <array>
#include <vector>

namespace alphago {

constexpr int NUM_FEATURES = 8;

// Feature planes (same as Python):
//   0: current player's stones
//   1: opponent's stones
//   2: empty points
//   3: last move (one-hot)
//   4: color to play (all 1s if black)
//   5: liberties == 1 (atari)
//   6: liberties == 2
//   7: liberties >= 3

struct BoardFeatures {
    // shape: [NUM_FEATURES][MAX_BOARD_SIZE][MAX_BOARD_SIZE]
    float data[NUM_FEATURES][MAX_BOARD_SIZE][MAX_BOARD_SIZE];
    int board_size;
};

// Extract features from a board position.
// last_move: the most recent move played (pass_move() if none or pass)
BoardFeatures board_to_features(
    const Board& board,
    uint8_t color_to_play,
    Move last_move = pass_move()
);

// Apply one of 8 symmetries to features and policy.
// sym_index: 0=identity, 1-3=rotations, 4-7=reflected+rotations
void apply_symmetry(
    const float* features_in,   // [C][H][W]
    const float* policy_in,     // [H*W + 1]
    float* features_out,        // [C][H][W]
    float* policy_out,          // [H*W + 1]
    int channels,
    int board_size,
    int sym_index
);

} // namespace alphago
