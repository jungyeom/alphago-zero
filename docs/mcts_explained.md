# How Monte Carlo Tree Search Works

MCTS is the search algorithm at the heart of AlphaGo and AlphaZero. It's how the AI "thinks ahead" — exploring possible futures to decide what move to play.

Our implementation has three layers:
1. **Python MCTS** — readable reference implementation with random rollouts and neural net evaluation
2. **C++ single-threaded MCTS** — same algorithm, ~15x faster board operations via pybind11
3. **C++ multi-threaded MCTS** — N worker threads with virtual loss + GPU batch queue for ~2-4x additional speedup

---

## 1. The Problem MCTS Solves

In Go, you can't just try every possible sequence of moves — there are too many. A 13x13 board has roughly 170 legal moves per position, and games last ~200 moves. That's 170^200 possible games, a number larger than atoms in the universe.

MCTS handles this by **sampling**. Instead of exploring everything, it plays out many random games from the current position and uses the results to estimate which moves are good. The key insight: you don't need to explore every possibility — you just need to explore the promising ones more deeply.

---

## 2. The Four Steps

Each MCTS "simulation" (iteration) follows four steps:

### SELECT

Start at the root (current position) and walk down the tree, picking the "best" child at each level, until you reach a **leaf node** — a position that hasn't been explored yet.

"Best" is defined by the **PUCT formula** (what AlphaZero uses):

```
score = Q + c * P * sqrt(N_parent) / (1 + N_child)
```

- **Q** = average win rate through this node (exploitation — pick what's worked before)
- **P** = prior probability from the neural network (focus on moves the net thinks are good)
- **c** = exploration constant (we use 1.5)
- The exploration bonus decreases as a child is visited more

We also support the classic **UCB1 formula** (`Q + c * sqrt(ln(N_parent) / N_child)`) for comparison, but PUCT is what's used in practice.

### EXPAND

When we reach a leaf node, create child nodes for all legal moves. Each child gets a prior probability from the neural network's policy head.

### EVALUATE

The neural network looks at the leaf position and outputs a value estimate in [-1, +1] — how much it thinks black is winning. This replaces random rollouts, which are much slower and noisier.

### BACKUP

Propagate the value back up the path from the leaf to the root. At each node, increment the visit count and add the value (flipped for the opponent's perspective).

---

## 3. Implementation Layers

### Python MCTS (`mcts/node.py`, `mcts/search.py`)

The readable reference implementation. Every node, selection, expansion, and backup step is in Python. This is where to look to understand the algorithm.

Key files:
- `mcts/node.py` — tree node with UCB1 and PUCT scoring
- `mcts/search.py` — single-threaded search with rollouts and neural net evaluation
- `mcts/batch_search.py` — batched search across N games (batches GPU evaluations for better throughput)

### C++ Single-Threaded MCTS (`src/mcts.cpp`)

Same algorithm as the Python version, but the board engine, tree traversal, and node operations run in C++. Only neural net evaluation crosses back to Python.

This gives ~15x speedup over pure Python for board operations (legal move generation, capture detection, scoring).

Key files:
- `src/board.h/cpp` — Go board with Zobrist hashing, superko detection
- `src/mcts.h/cpp` — MCTS search loop, PUCT selection, backup
- `src/mcts_node.h` — tree node with PUCT scoring
- `src/features.h/cpp` — board-to-tensor feature extraction
- `src/bindings.cpp` — pybind11 Python bindings
- `mcts/cpp_search.py` — Python wrappers (`search_cpp()`, `search_parallel_cpp()`)

### C++ Multi-Threaded MCTS (`src/parallel_mcts.cpp`)

The key optimization. Multiple C++ threads explore the same MCTS tree simultaneously:

```
Worker Thread 1: select -> reconstruct -> [wait for GPU] -> expand -> backup
Worker Thread 2: select -> reconstruct -> [wait for GPU] -> expand -> backup
Worker Thread 3: select -> reconstruct -> [wait for GPU] -> expand -> backup
Worker Thread 4: select -> reconstruct -> [wait for GPU] -> expand -> backup
                                |
                                v
Evaluator Thread: collect batch -> GPU forward pass -> dispatch results
```

This overlaps CPU tree work with GPU inference, keeping both busy.

---

## 4. Multi-Threaded MCTS Deep Dive

### The Problem with Single-Threaded MCTS

In single-threaded MCTS, the GPU sits idle while the CPU does tree traversal, and the CPU sits idle during the GPU forward pass. With 200 simulations per move, most of the time is wasted waiting.

### Virtual Loss

When multiple threads select paths through the tree, they'd all follow the same "best" path and redundantly explore it. **Virtual loss** solves this:

1. When a thread selects a node, it adds a temporary penalty (virtual loss) to that node
2. This makes the node look worse to other threads
3. Other threads explore different branches instead
4. When the thread finishes (backup), it removes the virtual loss

The formula with virtual loss:

```
Q_vl = (total_value - virtual_loss_count * vloss_value) / (visit_count + virtual_loss_count)
```

This naturally causes threads to diversify across the tree — each thread explores a different promising branch.

### GPU Batch Queue

Worker threads don't call the neural network directly. Instead:

1. Worker reaches a leaf and submits its board position to a **batch queue**
2. Worker blocks (waits for result via a promise/future)
3. The **evaluator thread** collects multiple positions from the queue
4. Evaluator runs one batched GPU forward pass for all collected positions
5. Evaluator dispatches results back to the waiting workers

This batches GPU calls automatically. With 4 worker threads, the evaluator typically processes 2-4 positions per batch. On GPU, this is much more efficient than 4 separate forward passes.

Key files:
- `src/batch_queue.h` — promise/future based queue connecting workers to evaluator
- `src/parallel_mcts.cpp` — worker thread loop and evaluator thread loop

### GIL Management

Python's Global Interpreter Lock (GIL) prevents true parallelism in Python threads. Our design avoids this bottleneck:

- The pybind11 binding **releases the GIL** before entering C++ (`py::gil_scoped_release`)
- Worker threads run entirely in C++ — no GIL needed
- Only the **evaluator thread** acquires the GIL, and only for the duration of the neural net forward pass
- The GIL is released immediately after the forward pass completes

This means N worker threads run at full speed in C++ while the evaluator handles all Python/GPU interaction.

### Thread Safety

All tree modifications are protected by a global mutex:
- `select_with_virtual_loss()` — reads tree + adds virtual loss (under lock)
- `expand_node_threadsafe()` — creates children (under lock, checks if already expanded)
- `backup_with_virtual_loss()` — updates values + removes virtual loss (under lock)

Board reconstruction and batch queue operations are thread-safe without the tree mutex (each worker has its own board copy; the queue has its own internal mutex).

The global tree mutex is simple and correct. For 4-8 threads it performs well. Scaling to 16+ threads would benefit from per-node atomic operations (future optimization).

---

## 5. Performance

### Measured Speedup (9x9 board, 200 simulations, CPU)

| Configuration | Time per search | Speedup |
|---------------|----------------|---------|
| Python MCTS | ~0.5s | baseline |
| C++ single-threaded | ~0.06s | ~8x |
| C++ 2 threads | ~0.05s | ~10x |
| C++ 4 threads | ~0.026s | ~19x |

On GPU, the speedup is even larger because the batch queue keeps the GPU fed with larger batches.

### ParallelStats

The C++ multi-threaded MCTS tracks statistics to verify threads are doing real work:

```python
result = mcts.search_parallel(board, color, batch_eval_fn, temperature)
stats = result.parallel_stats

stats.sims_per_thread  # e.g. [50, 50, 50, 50] for 4 threads, 200 sims
stats.num_batches      # how many GPU batches the evaluator processed
stats.total_batch_items  # total positions evaluated (should equal num_simulations)
```

---

## 6. Choosing the Final Move

After all simulations are done, we pick the move based on visit counts:

```
probability(move) = visits(move)^(1/temperature) / sum(all visits^(1/temperature))
```

- **Temperature = 1.0**: proportional to visit counts (more exploratory)
- **Temperature -> 0**: all probability on the most-visited move (deterministic)

During training, temperature=1.0 is used for the first ~15 moves (diverse training data) then drops to 0.1 (full strength).

### Dirichlet Noise

At the root node, we mix Dirichlet noise into the priors:

```
prior = (1 - weight) * network_prior + weight * dirichlet_noise
```

With alpha=0.1 and weight=0.25, this ensures the search explores some moves that the network wouldn't normally consider. This is essential for discovering new strategies during self-play.

---

## 7. Key Parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `num_simulations` | 200 | MCTS iterations per move |
| `c_puct` | 1.5 | Exploration constant in PUCT formula |
| `dirichlet_alpha` | 0.1 | Noise parameter (~10/board_size^2) |
| `dirichlet_weight` | 0.25 | Mix ratio: 75% net prior + 25% noise |
| `num_threads` | 4 | Worker threads for parallel search |
| `min_batch_size` | 4 | Minimum positions per GPU batch |
| `max_batch_size` | 16 | Maximum positions per GPU batch |
| `virtual_loss_value` | 1.0 | Penalty per virtual loss unit |

For reference, AlphaGo Zero used 1600 simulations per move with much larger networks.

---

## 8. How MCTS Connects to Training

MCTS both uses and improves the neural network:

```
Network provides:
  - Policy head -> prior probabilities P for PUCT selection
  - Value head  -> leaf evaluation V (replaces random rollouts)

MCTS produces:
  - Visit distribution -> better policy (training target for policy head)
  - Game outcomes     -> value labels (training target for value head)
```

The MCTS visit distribution is a better policy than the raw network output because search found improvements. The network learns to approximate it, and the cycle continues — each generation is stronger than the last.
