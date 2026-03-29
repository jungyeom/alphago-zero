# AlphaZero-Style Go Engine — Executive Summary

## Goal
Build an AlphaZero-style AI that plays 13x13 Go at a strong level, implemented from scratch as a learning project.

## Constraints
- ~$50 cloud compute budget for training
- Implemented from scratch in Python/PyTorch (learning-first, not performance-first)
- Package management via uv

## Architecture

**Core algorithm**: Neural network + Monte Carlo Tree Search (MCTS) trained via self-play.

1. A neural network takes a board position and outputs:
   - **Policy**: probability distribution over moves (where to play)
   - **Value**: who is winning (-1 to +1)

2. MCTS uses the network to search ahead, building a tree of likely moves and counter-moves. The search produces a better policy than the raw network output.

3. The network is trained on self-play games where the MCTS-improved policy is the training target. This creates a virtuous cycle: better network → better search → better training data → better network.

## Tech Stack
- **Go engine + MCTS + training**: Python, NumPy, PyTorch
- **Web frontend**: React + TypeScript (Vite), FastAPI backend
- **Package management**: uv
- **Target board size**: 13x13

## Network Architecture
- 6 residual blocks, 128 filters (~1-2M parameters)
- Policy head: softmax over 170 moves (169 board points + pass)
- Value head: tanh output in [-1, 1]

## Compute Strategy
- Data augmentation via 8-fold board symmetry (free 8x training data)
- 200-400 MCTS simulations per move during training
- Mixed precision (fp16) for ~2x GPU throughput
- A10G or L4 GPU on RunPod/Lambda (~$1-2/hr)
- Estimated 20-40 hours of training → $20-50

## Implementation Phases
1. Go rules engine (board, captures, ko, scoring) ← DONE
2. Random self-play (verify rules engine) ← DONE (1,100 games, 0 errors)
3. MCTS with random rollouts (no neural net) ← DONE (70% vs random on 5x5)
4. Neural network (ResNet with policy + value heads) ← DONE (1.2M params, overfits single position)
5. MCTS + neural net integration ← DONE (PUCT + net eval, 2-5x faster than rollouts)
6. Self-play training loop
7. Evaluation arena (model vs model)
8. Web frontend (play against the AI)

## Key Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-28 | 13x13 board size | Balance between strategic depth and trainability on $50 budget |
| 2026-03-28 | All Python (no Rust) | Focus learning energy on AI/MCTS, not language mechanics. Can rewrite hot paths later. |
| 2026-03-28 | Web UI for play interface | Interactive Go board, easy to use, uses existing board rendering libraries |
| 2026-03-28 | Chinese rules (area scoring) | Simpler to implement than Japanese rules, no need for complex dead-stone agreement |
| 2026-03-28 | 6 res blocks / 128 filters | Small enough for budget, large enough to learn 13x13 patterns |
| 2026-03-28 | uv for package management | User's standard toolchain preference |
| 2026-03-28 | Optimized is_legal (in-place) | Board copy per intersection was too slow for MCTS; switched to in-place simulate+undo |
| 2026-03-28 | Random rollouts are weak evaluator | MCTS+rollouts tops ~70% vs random on 5x5. Neural net (Step 5) is critical for real strength |
