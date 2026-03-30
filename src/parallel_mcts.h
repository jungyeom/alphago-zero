#pragma once

#include "mcts.h"
#include "batch_queue.h"

// This file is separate from mcts.h because it includes pybind11
// for GIL handling in the evaluator thread.

namespace alphago {

// The search_parallel() implementation lives in parallel_mcts.cpp
// and is declared in mcts.h as part of MCTSSearch.
// This header provides the internal implementation helpers.

} // namespace alphago
