# How the Self-Play Training Loop Works

This is where everything we've built comes together. The network learns to play Go by playing against itself, using MCTS to generate training data that's better than its own raw output.

---

## 1. The Virtuous Cycle

The core insight of AlphaZero is a feedback loop:

```
Network predicts → MCTS improves the prediction → Network learns the improvement → Repeat
```

At each iteration:
1. The current network plays games against itself using MCTS
2. MCTS produces a better policy than the raw network (because search explores ahead)
3. We train the network to match the MCTS-improved policy
4. The network gets better, so MCTS gets better, so the training target gets better

This is "bootstrapping" — the network lifts itself up by its own bootstraps, because MCTS search amplifies whatever the network has learned so far.

---

## 2. The Four Files

### `config.py` — All the knobs

Every hyperparameter lives here in a single dataclass. Key parameters:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `num_simulations` | 200 | MCTS sims per move during self-play |
| `games_per_iteration` | 100 | How many self-play games before each training phase |
| `batch_size` | 256 | Training batch size |
| `training_epochs` | 3 | Passes over the replay buffer per iteration |
| `learning_rate` | 0.01 | Starting LR (decays at iteration 100 and 200) |
| `replay_buffer_size` | 50,000 | Max stored positions |
| `temp_threshold` | 15 | First 15 moves use temperature=1.0, then 0.1 |
| `use_symmetry_augmentation` | True | 8x data via board rotations/reflections |

### `replay_buffer.py` — Position storage

A fixed-size queue that stores training examples. Each example is a tuple of:
- **Features**: the board state as an 8-channel tensor
- **Policy**: the MCTS visit distribution (the training target for the policy head)
- **Value**: who eventually won the game (+1 or -1, from the current player's perspective)

When the buffer is full, oldest positions are dropped. This means the network always trains on recent games, which matters because the self-play quality improves over time.

**Symmetry augmentation**: Go is symmetric under 4 rotations and 2 reflections (8 total). When we store a position, we also store all 7 transformed versions. This gives us 8x the training data for free, which is critical on a limited compute budget.

### `self_play.py` — Generating games

For each game:
1. Start with an empty board
2. At each position, run MCTS with the current network to get a policy
3. Record (features, MCTS_policy, color_to_play)
4. Select a move from the policy and play it
5. When the game ends, go back and label every position with the game outcome

**Temperature schedule**: The first 15 moves use temperature=1.0 (sample moves proportionally to MCTS visit counts). This creates diversity in the opening, producing more varied training data. After move 15, temperature drops to 0.1 (nearly deterministic), so the rest of the game is played at full strength.

### `trainer.py` — The main loop

Each iteration:

```
1. SELF-PLAY: Generate 100 games → ~20,000 positions (x8 with symmetry = 160,000)
2. STORE: Push positions into the replay buffer
3. TRAIN: Sample batches from buffer, run 3 epochs of gradient descent
4. CHECKPOINT: Save the model every 5 iterations
5. REPEAT for 200 iterations
```

**Loss function**: The network has two heads, so the loss is:
```
L = policy_loss + value_loss

policy_loss = -mean(target_policy * log_predicted_policy)   # cross-entropy
value_loss  = mean((predicted_value - actual_outcome)^2)     # MSE
```

The policy loss pushes the network to match what MCTS found. The value loss pushes it to correctly predict who wins from any position.

**Optimizer**: SGD with momentum (0.9) and weight decay (1e-4 for L2 regularization). Learning rate starts at 0.01 and decays by 10x at iterations 100 and 200.

---

## 3. The Temperature Schedule — Why It Matters

Without temperature, every game would start the same way (the network always plays its top move). The network would only see a handful of openings and overfit to them.

With temperature=1.0, the first 15 moves are sampled from the full MCTS distribution. Move A with 30% of visits and Move B with 25% of visits both get played frequently. This generates diverse openings.

After move 15, temperature drops to 0.1 (nearly argmax). This means the rest of the game is played at full strength — we want strong training data for the value head, which learns from game outcomes.

---

## 4. Why a Replay Buffer?

We don't just train on the most recent game. We store many games in a buffer and sample randomly from it. This helps because:

1. **Decorrelation**: consecutive positions in a game are highly correlated. Random sampling breaks this correlation.
2. **Efficiency**: each position gets used for training multiple times.
3. **Stability**: the network doesn't forget old positions while learning new ones.

The buffer is capped at 50,000 positions. This keeps the training data recent — very old games were played by a much weaker version of the network and are less useful.

---

## 5. Running Training

### On your local machine (testing)
```bash
uv run python -m training.trainer \
  --board-size 5 \
  --num-iterations 10 \
  --games-per-iter 5 \
  --simulations 20 \
  --device cpu
```

### On a GPU (real training)
```bash
uv run python -m training.trainer \
  --board-size 13 \
  --num-iterations 200 \
  --games-per-iter 100 \
  --simulations 200 \
  --device cuda
```

### Resuming from a checkpoint
```bash
uv run python -m training.trainer \
  --resume checkpoints/model_iter_0050.pt \
  --device cuda
```

---

## 6. What to Watch For

During training, you should see:

- **Policy loss decreasing**: the network is learning to match MCTS's improved policy
- **Value loss decreasing**: the network is learning to predict game outcomes
- **Game length changing**: early games are short and random; as the network improves, games get longer and more structured
- **Black/white balance**: win rates should be roughly balanced (komi compensates for first-move advantage)

If policy loss plateaus, try:
- More MCTS simulations per move
- Lower learning rate
- More games per iteration

---

## 7. Compute Budget Breakdown

For 13x13 with our default config:

| Phase | Per iteration | Total (200 iters) |
|-------|-------------|-------------------|
| Self-play (100 games x 200 sims) | ~10-30 min on GPU | ~30-100 hrs |
| Training (3 epochs, 256 batch) | ~1-2 min | ~3-6 hrs |
| **Total** | | **~35-106 hrs** |

On an A10G (~$1/hr), this is **$35-106**. To fit within $50:
- Reduce to 100 iterations
- Use 100 simulations instead of 200
- Start with 9x9 and transfer to 13x13

Mixed precision (fp16) roughly halves GPU time for training, but self-play is the bottleneck since MCTS is mostly CPU-bound. Batching multiple games' neural net evaluations together would help but adds complexity.
