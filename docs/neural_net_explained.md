# How the Neural Network Works (Steps 4-5)

This document covers two tightly connected pieces: the neural network itself (Step 4) and how it integrates with MCTS (Step 5). Together, they transform MCTS from a slow, random search into the fast, pattern-aware search that makes AlphaZero work.

---

## 1. What the Network Does

The network takes a board position and outputs two things:

- **Policy**: a probability distribution over all legal moves — "where should I play?"
- **Value**: a single number between -1 and +1 — "who is winning?"

These replace the two weaknesses of the random-rollout MCTS:
- The **uniform prior** (all moves treated equally) is replaced by the **policy** (focus search on promising moves)
- The **random rollout** (play random moves to the end) is replaced by the **value** (instant position evaluation)

---

## 2. Input Features (`model/features.py`)

The network can't look at raw board state — it needs structured information as a multi-channel tensor. Think of it like an image with 8 "color channels" instead of RGB's 3.

| Channel | What it represents | Why it helps |
|---------|-------------------|--------------|
| 0 | Current player's stones | Where my stones are |
| 1 | Opponent's stones | Where their stones are |
| 2 | Empty points | Where I can potentially play |
| 3 | Last move (one-hot) | Important for ko and context |
| 4 | Color to play (all 1s or 0s) | Black and white play differently |
| 5 | Stones with 1 liberty (atari) | Urgent tactical information |
| 6 | Stones with 2 liberties | Near-atari groups |
| 7 | Stones with 3+ liberties | Safe groups |

The input tensor shape is `(8, board_size, board_size)` — for 13x13, that's `(8, 13, 13)`.

### Why "current player's perspective"?

We always encode the board so that channel 0 = the player whose turn it is and channel 1 = the opponent. This means the same network can play as both black and white — it always sees "my stones" and "their stones" regardless of color. The color-to-play channel (4) tells it which side it's playing.

### Data Augmentation via Symmetries

A Go board has 8 symmetries (4 rotations x 2 reflections). If position P has value V and policy move M, then rotating P by 90 degrees gives a position with the same value V and a rotated move M.

The `apply_symmetry` function applies any of the 8 transformations to both the feature tensor and the policy vector. This gives us **8x training data for free** — critical for our $50 budget.

---

## 3. Network Architecture (`model/network.py`)

### Overview

```
Input (8, 13, 13)
  → Conv3x3 (8 → 128 channels) → BatchNorm → ReLU
  → 6x Residual Block
  → Policy Head → 170 probabilities (169 board points + pass)
  → Value Head  → 1 scalar in [-1, +1]
```

### Residual Blocks

Each of the 6 residual blocks does:

```
input x
  → Conv3x3 → BatchNorm → ReLU
  → Conv3x3 → BatchNorm
  → Add x back (skip connection)
  → ReLU
```

The **skip connection** (`Add x back`) is the key innovation from the ResNet paper. It lets gradients flow directly through the network during training, making it possible to train deeper networks without them "forgetting" what the input was.

Each conv layer has 128 filters with 3x3 kernels and padding=1 (so the spatial dimensions stay the same throughout).

### Policy Head

```
trunk output (128, 13, 13)
  → Conv1x1 (128 → 2 channels) → BatchNorm → ReLU
  → Flatten to vector of length 2 × 13 × 13 = 338
  → Linear layer → 170 outputs
  → Log-softmax → log-probabilities
```

The output has 170 values: one for each of the 169 board intersections plus one for "pass." During MCTS, we mask out illegal moves and renormalize.

### Value Head

```
trunk output (128, 13, 13)
  → Conv1x1 (128 → 1 channel) → BatchNorm → ReLU
  → Flatten to vector of length 169
  → Linear layer → 256 hidden units → ReLU
  → Linear layer → 1 output
  → Tanh → scalar in [-1, +1]
```

The tanh ensures the output is bounded: +1 means "current player is winning completely" and -1 means "current player is losing completely."

### Parameter Count

With 6 blocks and 128 filters: approximately **1.2 million parameters**. This is small enough to train on a single GPU within our budget, but large enough to learn meaningful 13x13 Go patterns.

---

## 4. How the Network Integrates with MCTS (`search_with_net`)

### What Changed from Random-Rollout MCTS

| Component | Before (Step 3) | After (Step 5) |
|-----------|-----------------|----------------|
| **Selection formula** | UCB1: `Q + c·√(ln N_parent / N_child)` | PUCT: `Q + c·P·√N_parent / (1 + N_child)` |
| **Expansion priors** | Uniform (1/num_moves) | Network policy (learned) |
| **Leaf evaluation** | Random rollout (play to end) | Network value (instant) |
| **Root exploration** | None | Dirichlet noise on priors |

### PUCT Formula

```
score(move) = Q(move) + c_puct × P(move) × √N_parent / (1 + N_move)
```

The key difference from UCB1: the exploration term includes **P(move)**, the network's prior probability. Moves the network thinks are promising get explored more. This dramatically reduces the branching factor — instead of trying all 170 moves equally, the search focuses on the 10-20 moves the network thinks are best.

### Dirichlet Noise

At the root node, we add random noise to the priors:

```
P_root = 0.75 × P_network + 0.25 × Dirichlet(α=0.1)
```

This ensures the search doesn't get stuck in the network's blind spots. Even if the network assigns zero probability to a move, the noise gives it a small chance of being explored. The alpha parameter (0.1 for 13x13) controls how "spiky" the noise is — lower values mean the noise concentrates on fewer moves.

### The Search Loop

```
1. Expand root with network → get priors and value
2. Add Dirichlet noise to root priors
3. For each simulation:
   a. SELECT: walk down tree using PUCT
   b. Reconstruct board at leaf (replay moves)
   c. If game over → score directly
   d. Else → EXPAND leaf with network (get priors + value)
   e. BACKUP value through the tree
4. Return root with all statistics
```

### Speed Comparison

With a small test network (2 blocks, 32 filters) on 9x9:
- **Random rollout MCTS**: each simulation plays ~80 random moves and checks legality for each
- **Neural net MCTS**: each simulation does one forward pass through the network

The neural net version is **2-5x faster** even on CPU with a small network. On GPU with the full-size network, the speedup is much larger because forward passes can be batched.

---

## 5. How Training Will Work (Preview of Step 6)

The network and MCTS form a self-improving loop:

```
┌─────────────────────────────────────────────┐
│                                             │
│  Network predicts (policy, value)           │
│       ↓                                     │
│  MCTS uses network to search                │
│       ↓                                     │
│  MCTS produces improved policy              │
│  (visit counts → better than raw network)   │
│       ↓                                     │
│  Play self-play games using MCTS            │
│  Record (state, mcts_policy, game_result)   │
│       ↓                                     │
│  Train network on self-play data            │
│  policy_loss = cross_entropy(net, mcts)     │
│  value_loss = mse(net_value, game_result)   │
│       ↓                                     │
│  Network gets better → MCTS gets better     │
│       ↓                                     │
│  (repeat)                                   │
│                                             │
└─────────────────────────────────────────────┘
```

The critical insight: **MCTS always produces a better policy than the raw network** because search finds improvements. By training the network to match the MCTS policy, the network absorbs these improvements. Then the improved network makes MCTS even better, and the cycle continues.

---

## 6. Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Log-softmax output | Yes | More numerically stable than softmax + log for cross-entropy loss |
| Shared trunk | Yes | Policy and value share features — saves compute, improves both |
| He initialization | Yes | Standard for ReLU networks, prevents gradient issues |
| 8 input features | Minimal but sufficient | More features (move history, ladder detection) could help but add complexity |
| Mask illegal moves | After network output | Network outputs for all positions, we zero illegals and renormalize |
