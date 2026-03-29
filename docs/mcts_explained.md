# How Monte Carlo Tree Search Works

MCTS is the search algorithm at the heart of AlphaGo and AlphaZero. It's how the AI "thinks ahead" — exploring possible futures to decide what move to play. This document explains the version we've built so far, which uses **random rollouts** (no neural network yet).

---

## 1. The Problem MCTS Solves

In Go, you can't just try every possible sequence of moves — there are too many. A 13x13 board has roughly 170 legal moves per position, and games last ~200 moves. That's 170^200 possible games, a number larger than atoms in the universe.

MCTS handles this by **sampling**. Instead of exploring everything, it plays out many random games from the current position and uses the results to estimate which moves are good. The key insight: you don't need to explore every possibility — you just need to explore the promising ones more deeply.

---

## 2. The Four Steps

Each MCTS "simulation" (iteration) follows four steps:

### SELECT

Start at the root (current position) and walk down the tree, picking the "best" child at each level, until you reach a **leaf node** — a position that hasn't been explored yet.

"Best" is defined by the **UCB1 formula**:

```
score = Q + c * sqrt(ln(N_parent) / N_child)
```

- **Q** = average win rate through this node (exploitation — pick what's worked before)
- **c * sqrt(...)** = exploration bonus (exploration — try things we haven't tried much)
- **c** = constant that balances the two (we use 1.4)

Unvisited nodes get a score of infinity, so they're always tried at least once. After that, the formula naturally balances:
- Nodes with high win rates get visited more (exploitation)
- Nodes with few visits get a bonus (exploration)

### EXPAND

When we reach a leaf node, we create child nodes for all legal moves from that position. Each child represents "what if we played this move?"

### EVALUATE (Rollout)

From the leaf position, play **completely random moves** for both sides until the game ends. Score the result: +1 if black wins, -1 if white wins.

This is the "Monte Carlo" part — we're using random sampling to estimate the value of a position. One random game is noisy and unreliable, but the average over many random games gives a surprisingly useful signal.

> **Important**: In Step 5, we'll replace this random rollout with a neural network evaluation. The network will look at the position and directly output a value estimate, which is much faster and more accurate than playing random moves to the end.

### BACKUP

Take the result from the rollout and propagate it back up the path we walked during SELECT. At each node, increment the visit count and add the result to the total value.

The values are stored from each player's perspective — a black win is +1 at black's nodes and -1 at white's nodes. This way, each node's Q value (average value) represents "how good is this move for the player who made it."

---

## 3. The Tree Node (`node.py`)

Each node in the search tree stores:

| Field | What it is | Why it matters |
|-------|-----------|----------------|
| `move` | The move that led to this node | So we know what move to play |
| `parent` | Link to parent node | For backup (propagating values up) |
| `children` | List of child nodes | The moves we can explore from here |
| `color` | Who played the move | To flip value perspective during backup |
| `visit_count` (N) | How many times this node was visited | Used in UCB1 formula and for final move selection |
| `total_value` (W) | Sum of all backed-up values | Divided by N to get Q (average value) |
| `prior` (P) | Prior probability | Unused now, will come from neural net in Step 5 |

### Two selection formulas

The node supports two formulas:

1. **UCB1** (what we use now): `Q + c * sqrt(ln(N_parent) / N_child)` — the classic MCTS formula. The exploration bonus depends only on visit counts.

2. **PUCT** (what AlphaZero uses): `Q + c * P * sqrt(N_parent) / (1 + N_child)` — uses the neural network's prior probability P to guide exploration. Moves the network thinks are promising get explored more. We'll switch to this in Step 5.

---

## 4. The Search Process (`search.py`)

### How a search runs

```
1. Create root node
2. Expand root (create children for all legal moves)
3. For each simulation:
   a. SELECT: walk down tree using UCB1 to find a leaf
   b. Reconstruct the board state at the leaf (replay moves from root)
   c. EXPAND: create children of the leaf
   d. EVALUATE: random rollout from the leaf position
   e. BACKUP: propagate the value up to the root
4. Return the root (with all accumulated statistics)
```

### Reconstructing board state

The tree only stores moves, not full board states (that would use too much memory). To know the actual board position at a leaf node, we:

1. Walk from the leaf back up to the root, collecting the moves along the way
2. Starting from the root's board position, replay those moves one by one

This is a tradeoff: it costs some computation per simulation, but saves a lot of memory.

### Choosing the final move

After all simulations are done, we pick the move with the **most visits**, not the highest win rate. Why?

- Visit count is more stable (less affected by noise)
- UCB1 naturally directs more visits to better moves
- The most-visited child is the one the search is most "confident" about

### Move probabilities and temperature

For training the neural network later, we don't just want the best move — we want a **probability distribution** over all moves. This comes from the visit counts:

```
probability(move) = visits(move)^(1/temperature) / sum(all visits^(1/temperature))
```

- **Temperature = 1.0**: proportional to visit counts (more exploratory)
- **Temperature → 0**: all probability on the most-visited move (deterministic)

During training, we'll use temperature=1.0 for the first ~15 moves (to generate diverse training data) and then switch to low temperature (to play stronger moves).

---

## 5. Why Random Rollouts Work (and Their Limitations)

### Why they work at all

Random play might seem useless, but averaging over hundreds of random games from a position gives a real signal. A position where black has surrounded a large territory will lead to more black wins even with random play. A position where a black group is about to be captured will lead to more black losses.

### Their limitations

1. **Noisy**: One random game tells you almost nothing. You need many simulations, which is slow.
2. **Tactically blind**: Random play doesn't see captures, atari, or life/death. It might randomly save a dying group or randomly kill a living one.
3. **Strategically oblivious**: Random play doesn't understand influence, thickness, or territory. It just plays legal moves at random.

This is exactly why we'll add a neural network in Step 5. The network replaces the rollout with a direct position evaluation — it looks at the board and says "black is winning by about 0.3." This is faster (one forward pass vs playing 200 random moves) and far more accurate (it learns patterns from millions of positions).

---

## 6. How MCTS Connects to AlphaZero

Here's the full picture of how MCTS fits into the training loop (Steps 5-6):

```
Current MCTS (Step 3):          AlphaZero MCTS (Step 5):
  SELECT  → UCB1                  SELECT  → PUCT (uses neural net prior P)
  EXPAND  → uniform prior         EXPAND  → prior P from neural net
  EVALUATE → random rollout       EVALUATE → neural net value V (no rollout!)
  BACKUP  → same                  BACKUP  → same
```

The neural network gives MCTS two things:
1. **Better priors** → the search focuses on promising moves instead of trying everything uniformly
2. **Better evaluation** → direct value estimate instead of noisy random play

MCTS then gives the neural network better training data:
- The MCTS visit distribution is a **better policy** than the raw network output (because search found improvements)
- This improved policy becomes the training target
- The network learns to approximate it, and the cycle continues

---

## 7. Key Numbers

| Parameter | Current value | Why |
|-----------|--------------|-----|
| Simulations per move | 50-200 | Enough to beat random, but slow for 13x13 |
| UCB1 exploration constant (c) | 1.4 | Standard value, balances exploration vs exploitation |
| Max rollout length | 200 moves | Safety cap on random game length |
| Board state reconstruction | Replay from root | Memory-efficient, costs O(depth) per simulation |

For reference, AlphaGo Zero used **1600 simulations per move**. We'll scale up during training.

---

## 8. Benchmark Results: MCTS vs Random

We tested MCTS (random rollouts) against a purely random agent:

| Board | Sims | MCTS Win Rate | Notes |
|-------|------|---------------|-------|
| 5x5 | 50 | 60% | Barely better — 50 sims ≈ 2x the legal moves |
| 5x5 | 200 | 70% | Clear improvement with more search |
| 9x9 | 30 | 20% | Worse than random — 30 sims < 82 legal moves, no useful signal |

### Key Takeaways

1. **MCTS needs simulations > legal moves to work.** With fewer sims than legal moves, most children are visited at most once. UCB1 can't distinguish good from bad.

2. **Random rollouts are a weak evaluator.** Even with sufficient simulations, the win rate tops out around 70% on 5x5. Random play is terrible at Go — it can't see captures, life/death, or territory. The signal is very noisy.

3. **This is exactly why the neural net matters.** In Step 5, the network replaces rollouts with a direct value estimate. This gives:
   - **Better evaluation**: trained on millions of positions, not random play
   - **Better priors**: focuses search on promising moves via the policy head
   - **Much faster**: one forward pass vs playing 100+ random moves

4. **Performance optimization matters.** We had to optimize `is_legal()` (in-place simulation instead of board copy) and the rollout (sample random empty points instead of generating all legal moves) to make the benchmark tractable.

### What "working" looks like at each stage

| Stage | Expected strength |
|-------|-------------------|
| Random rollouts (now) | Slightly better than random on small boards |
| Neural net + MCTS (Step 5) | Much stronger than random, should play recognizable Go |
| Trained via self-play (Step 6) | Strong enough to beat beginners |
