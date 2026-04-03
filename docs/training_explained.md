# How the Self-Play Training Loop Works

This is where everything we've built comes together. The network learns to play Go by playing against itself, using MCTS to generate training data that's better than its own raw output.

---

## 1. The Virtuous Cycle

The core insight of AlphaZero is a feedback loop:

```
Network predicts -> MCTS improves the prediction -> Network learns the improvement -> Repeat
```

At each iteration:
1. The current network plays games against itself using MCTS
2. MCTS produces a better policy than the raw network (because search explores ahead)
3. We train the network to match the MCTS-improved policy
4. The network gets better, so MCTS gets better, so the training target gets better

This is "bootstrapping" — the network lifts itself up by its own bootstraps, because MCTS search amplifies whatever the network has learned so far.

---

## 2. The Training Pipeline

### `config.py` — All the knobs

Every hyperparameter lives here in a single dataclass. Key parameters:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `num_simulations` | 200 | MCTS sims per move during self-play |
| `games_per_iteration` | 100 | Self-play games before each training phase |
| `batch_size` | 256 | Training batch size |
| `training_epochs` | 3 | Passes over the replay buffer per iteration |
| `learning_rate` | 0.01 | Starting LR (decays at iteration 100 and 200) |
| `replay_buffer_size` | 50,000 | Max stored positions |
| `temp_threshold` | 15 | First 15 moves use temperature=1.0, then 0.1 |
| `use_symmetry_augmentation` | True | 8x data via board rotations/reflections |
| `use_cpp` | True | Use C++ board engine + MCTS |
| `num_search_threads` | 1 | C++ MCTS worker threads (set 4-8 on GPU) |
| `min_batch_size` | 4 | Min positions per GPU batch in parallel MCTS |
| `max_batch_size` | 16 | Max positions per GPU batch in parallel MCTS |
| `num_parallel_games` | 16 | Simultaneous self-play games |

### `replay_buffer.py` — Position storage

A fixed-size queue that stores training examples. Each example is a tuple of:
- **Features**: the board state as an 8-channel tensor
- **Policy**: the MCTS visit distribution (training target for the policy head)
- **Value**: who eventually won the game (+1 or -1, from the current player's perspective)

When the buffer is full, oldest positions are dropped. This keeps training on recent games, which matters because self-play quality improves over time.

**Symmetry augmentation**: Go is symmetric under 4 rotations and 2 reflections (8 total). Each stored position is augmented with all 7 transformations, giving 8x training data for free.

### `self_play.py` — Generating games (sequential)

For each game:
1. Start with an empty board
2. At each position, run MCTS with the current network to get a policy
3. Record (features, MCTS_policy, color_to_play)
4. Select a move from the policy and play it
5. When the game ends, label every position with the game outcome

This file supports three MCTS backends:
- **C++ multi-threaded** (`search_parallel_cpp`): when `use_cpp=True` and `num_search_threads > 1`
- **C++ single-threaded** (`search_cpp`): when `use_cpp=True` and `num_search_threads == 1`
- **Python** (`get_move_probabilities_with_net`): fallback when C++ is unavailable

### `parallel_self_play.py` — Generating games (parallel)

Runs `num_parallel_games` games simultaneously. When a game finishes, a new one starts in its slot. This keeps the pipeline moving without waiting for long games to end.

When C++ multi-threaded MCTS is available (`use_cpp=True`, `num_search_threads > 1`), each game's MCTS runs with N C++ worker threads + virtual loss + GPU batch queue. Otherwise, it falls back to the Python `ParallelMCTS` which batches leaf evaluations across all games' MCTS trees in Python.

The trainer routes here by default (when `use_parallel_self_play=True` and `num_parallel_games > 1`).

### `trainer.py` — The main loop

Each iteration:

```
1. SELF-PLAY: Generate 100 games (parallel, with C++ multi-threaded MCTS)
2. STORE: Push positions into the replay buffer (with 8x symmetry augmentation)
3. TRAIN: Sample batches from buffer, run 3 epochs of gradient descent
4. CHECKPOINT: Save the model every 5 iterations
5. REPEAT for 200 iterations
```

**Loss function**:
```
L = policy_loss + value_loss

policy_loss = -mean(target_policy * log_predicted_policy)   # cross-entropy
value_loss  = mean((predicted_value - actual_outcome)^2)     # MSE
```

**Optimizer**: SGD with momentum (0.9) and weight decay (1e-4 for L2 regularization). Learning rate starts at 0.01 and decays by 10x at iterations 100 and 200.

---

## 3. Self-Play Acceleration

Self-play is the bottleneck — each iteration generates 100 games, each ~200 moves, each move requiring 200 MCTS simulations. Three levels of acceleration stack together:

### Level 1: C++ Board Engine

The Go board (legal moves, captures, ko, scoring) runs in C++ via pybind11. ~15x faster than Python for board operations.

### Level 2: Multi-Threaded MCTS

Each move's MCTS runs N C++ worker threads concurrently (default: 4). Worker threads explore different branches using virtual loss, and their leaf evaluations are batched through a GPU queue. See `docs/mcts_explained.md` for the full architecture.

Speedup: ~2.4x on CPU with 4 threads, higher on GPU.

### Level 3: Parallel Games

Multiple games run simultaneously. The trainer manages N game slots (default: 16), running each game's MCTS and advancing all active games each round. When a game finishes, a new one starts in its slot.

When C++ multi-threaded MCTS is active, each game uses its own pool of worker threads. When using the Python fallback, leaf evaluations from all games are batched into a single GPU forward pass for better throughput.

### How They Combine

```
                  Parallel Games (Level 3)
                  ┌──────────────────────┐
                  │  Game 1   Game 2 ... │
                  │    │         │       │
                  │    v         v       │
                  │  MCTS      MCTS     │  <-- Each game uses multi-threaded MCTS (Level 2)
                  │  ┌─┐       ┌─┐      │      with C++ board engine (Level 1)
                  │  │W│W│W│E  │W│W│W│E │
                  │  └─┘       └─┘      │
                  └──────────────────────┘
                  W = C++ worker thread
                  E = evaluator thread (calls GPU)
```

---

## 4. The Temperature Schedule

Without temperature, every game would start the same way (the network always plays its top move). The network would only see a handful of openings and overfit.

With temperature=1.0, the first 15 moves are sampled from the full MCTS distribution. Move A with 30% of visits and Move B with 25% both get played frequently, generating diverse openings.

After move 15, temperature drops to 0.1 (nearly argmax). The rest of the game is played at full strength — we want strong data for the value head, which learns from game outcomes.

---

## 5. Why a Replay Buffer?

We don't just train on the most recent game. We store many games in a buffer and sample randomly. This helps because:

1. **Decorrelation**: consecutive positions in a game are highly correlated. Random sampling breaks this.
2. **Efficiency**: each position gets used for training multiple times.
3. **Stability**: the network doesn't forget old positions while learning new ones.

The buffer is capped at 50,000 positions. This keeps training data recent — very old games were played by a much weaker version of the network.

---

## 6. Running Training

### Local test (CPU, ~15 minutes)

```bash
# Uses 5x5 board with C++ multi-threaded MCTS (4 threads)
uv run python train_local.py
```

### Local with custom settings

```bash
uv run python -m training.trainer \
  --board-size 9 \
  --num-iterations 30 \
  --games-per-iter 20 \
  --simulations 100 \
  --search-threads 4 \
  --use-cpp \
  --device cpu
```

### GPU training (real 13x13)

```bash
uv run python -m training.trainer \
  --board-size 13 \
  --num-iterations 200 \
  --games-per-iter 100 \
  --simulations 200 \
  --search-threads 4 \
  --parallel-games 128 \
  --use-cpp \
  --device cuda
```

### Resuming from a checkpoint

```bash
uv run python -m training.trainer \
  --resume checkpoints/model_iter_0050.pt \
  --search-threads 4 \
  --parallel-games 128 \
  --use-cpp \
  --device cuda
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--board-size` | 13 | Board size |
| `--num-iterations` | 200 | Total training iterations |
| `--games-per-iter` | 100 | Self-play games per iteration |
| `--simulations` | 200 | MCTS sims per move |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 0.01 | Learning rate |
| `--search-threads` | 1 | C++ MCTS worker threads per game (try 4-8 on GPU) |
| `--parallel-games` | 16 | Simultaneous self-play games (try 64-128 on GPU) |
| `--use-cpp` | off | Enable C++ board engine + MCTS |
| `--device` | auto | cpu, cuda, or mps |
| `--resume` | none | Checkpoint path to resume from |

---

## 7. What to Watch For

During training, you should see:

- **Policy loss decreasing**: the network is learning to match MCTS's improved policy
- **Value loss decreasing**: the network is learning to predict game outcomes
- **Game length changing**: early games are short and random; as the network improves, games get longer and more structured
- **Black/white balance**: win rates should be roughly balanced (komi compensates for first-move advantage)

If policy loss plateaus, try:
- More MCTS simulations per move
- Lower learning rate
- More games per iteration

See `docs/monitoring_training.md` for TensorBoard setup and detailed troubleshooting.

---

## 8. Compute Budget Breakdown

With C++ multi-threaded MCTS (4 threads) and parallel games (128):

| Board | Config | Est. time (RTX 4090) | Est. cost (spot) |
|-------|--------|---------------------|-----------------|
| 5x5 | 30 iters, 10 games, 50 sims | ~15 min | free (CPU) |
| 9x9 | 150 iters, 80 games, 150 sims | ~4-8 hrs | $1-2 |
| 13x13 | 100 iters, 50 games, 150 sims | ~15-25 hrs | $3-5 |
| 13x13 | 200 iters, 100 games, 200 sims | ~30-50 hrs | $6-10 |

Self-play is the bottleneck (~90% of wall time). The multi-threaded C++ MCTS roughly halves self-play time compared to the Python implementation. Mixed precision (fp16) additionally halves GPU training time.
