# How the Go Engine Works

This document walks through `board.py` and `game.py` — the two files that implement all Go rules. If you understand these, you understand the foundation that MCTS and the neural network will build on top of.

---

## 1. How the Board is Represented

The board is stored as a 2D grid of numbers using a numpy array:

- `0` = empty intersection
- `1` = black stone
- `2` = white stone

Coordinates use `(row, col)` where `(0, 0)` is the top-left corner of the board. On a 13x13 board, `(12, 12)` is the bottom-right.

When we create a board, we also set up:
- A **move history** (list of every move played, in order)
- A **capture counter** (how many stones each side has captured)
- **Komi** (6.5 points given to white as compensation for going second)

> **Why this matters for the AI**: Later, the neural network will receive the board grid as input. The simpler and more consistent the representation, the easier it is for the network to learn patterns.

---

## 2. Neighbors and Connected Groups

### Neighbors

Every intersection on a Go board has up to 4 orthogonal neighbors (up, down, left, right — no diagonals). Corner points have 2 neighbors, edge points have 3, and center points have 4.

The `_neighbors` method simply returns whichever of those 4 directions are within the board boundaries.

### Groups (Connected Stones)

A **group** is a set of same-colored stones that are connected to each other through orthogonal adjacency. Think of it like a chain — if you can walk from one stone to another by stepping through stones of the same color (up/down/left/right only), they're in the same group.

The `_group` method finds all stones in a group using **breadth-first search (BFS)**:

1. Start at a stone
2. Look at all its neighbors
3. If a neighbor is the same color, add it to the group and explore *its* neighbors too
4. If a neighbor is empty, record it as a **liberty** (a breathing point)
5. Keep going until there are no more connected same-color stones to explore

It returns two things:
- The set of all stones in the group
- The set of all empty points adjacent to the group (its liberties)

### Liberties

A **liberty** is an empty point directly adjacent to a group. Liberties are how the group "breathes." If a group has zero liberties — meaning every adjacent point is occupied — it is captured and removed from the board.

> **Why this matters for the AI**: Liberty counting is the most fundamental concept in Go tactics. The neural network will later receive liberty information as part of its input features.

---

## 3. Zobrist Hashing — A Fingerprint for Board Positions

We need a way to quickly answer: "Have we seen this exact board position before?" This matters for the **ko rule** (explained below). Comparing two entire 13x13 grids every time would be slow, so we use a clever trick called **Zobrist hashing**.

### How it works

At startup, we generate a big table of random 64-bit numbers — one for every possible combination of `(row, col, color)`. So for a 13x13 board, that's `13 x 13 x 2 = 338` random numbers (one set for black, one for white, at each intersection).

The board's **hash** is a single 64-bit number that uniquely(ish) represents the current position. We maintain it incrementally:

- When we **place** a stone at (row, col) with a given color, we XOR the hash with the corresponding random number
- When we **remove** a stone (capture), we XOR with the same number again, which undoes the first XOR

The magic of XOR:
- `hash ^ value ^ value = hash` (XORing twice cancels out)
- Order doesn't matter — placing stones A then B gives the same hash as placing B then A, as long as the final position is the same

This means we never have to recompute the hash from scratch. Every time the board changes, we just do one fast XOR operation per stone added or removed.

We store every hash we've ever seen in a set called `_position_history`. To check for repeated positions, we just ask: "Is this hash already in the set?" That's an O(1) operation.

> **Why this matters for the AI**: During MCTS search, we'll be copying and modifying board states thousands of times per move. Fast position comparison via hashing makes this practical.

---

## 4. Captures — How Stones Get Removed

When you place a stone, it might take away the last liberty of an adjacent opponent group. If so, that entire group is captured (removed from the board).

The `_place_stone` method does this:

1. Place the stone on the grid and update the hash
2. Look at each neighbor of the placed stone
3. If a neighbor is an opponent stone, find its entire group and count its liberties
4. If that group has **zero liberties**, remove every stone in it (set to empty, update hash)
5. Return the total number of stones captured

Important: we check *opponent* groups first. This is why a move that fills your own last liberty can still be legal — if it captures opponent stones, those captures happen first, freeing up liberties for your group.

---

## 5. The Suicide Rule

**Suicide** is playing a stone that would result in your own group having zero liberties, without capturing any opponent stones. Our engine forbids suicide (as most rulesets do).

The `_is_suicide` method checks this by temporarily placing the stone and asking two questions:

1. Does this move capture any opponent stones? (If yes, it's not suicide — captures free up liberties)
2. Does the placed stone's group have any liberties? (If yes, it's fine)

If the answer to *both* is "no" — no captures and no liberties — the move is suicide and is illegal.

The temporary stone is removed after checking, so the board state is not modified.

---

## 6. Ko and Superko — No Repeating Positions

### The problem

In Go, it's possible to create an infinite loop. Imagine black captures one stone, then white recaptures the stone that just captured, then black recaptures again, forever. This is called **ko**.

### Simple ko example

```
. X O .          . X O .         . X O .
X . X O    →     X X . O    →    X . X O    →  (would repeat forever)
. X O .          . X O .         . X O .
```

Black captures at the empty point, then white wants to capture right back. The ko rule says: **you may not make a move that returns the board to any previous position.**

### How we enforce it (Positional Superko)

We use the Zobrist hash history. Every time a stone is placed, the resulting board hash is added to `_position_history` (a set).

When checking if a move is legal, we:
1. Make a copy of the board
2. Simulate the move (place stone + do captures) on the copy
3. Check if the resulting hash already exists in the history
4. If it does, the move is illegal (it would repeat a position)

This is **positional superko**, which is stricter than simple ko — it prevents *any* repeated position, not just the immediately previous one. This is slightly more expensive to check but eliminates all possible infinite loops.

---

## 7. Legal Move Checking — The Full Pipeline

The `is_legal` method runs through these checks in order:

1. **Pass?** Always legal. Return `True`.
2. **In bounds?** Row and column must be within 0 to size-1.
3. **Empty?** The intersection must not already have a stone.
4. **Not suicide?** The move must not leave your own group with zero liberties (unless it captures).
5. **Not superko?** The resulting board position must not have occurred before in this game.

If all checks pass, the move is legal.

The `legal_moves` method simply loops over every intersection, runs `is_legal` on each one, and collects the legal ones. It always includes `None` (pass) at the end.

> **Why this matters for the AI**: The neural network will output a probability for every possible move (169 board points + pass = 170 values). We need to mask out illegal moves before using these probabilities in MCTS.

---

## 8. Scoring — Who Won?

We use **Chinese rules** (area scoring) because they're simpler to implement than Japanese rules.

### How scoring works

1. **Count stones on the board** for each color
2. **Find territory**: BFS through each connected region of empty points. If an empty region is bordered by *only one color*, it's that color's territory. If it touches both colors, it belongs to neither (contested).
3. **Score** = stones on board + territory
4. **Komi**: White gets 6.5 extra points (compensation for going second). The 0.5 ensures no ties.

The `score` method returns a dictionary with black's score, white's score, the winner, and the margin.

### Why Chinese rules?

Under Japanese rules, you'd need to identify "dead stones" (stones that *could* be captured but haven't been yet) — this requires agreement between players or complex life/death analysis. Chinese rules avoid this: you just count what's on the board plus surrounded empty space. The result is almost always the same, but the implementation is far simpler.

---

## 9. The Game Manager (`game.py`)

The `Board` class handles rules. The `Game` class handles *game flow*:

### What it tracks
- **Whose turn it is** (`current_player`): starts with black, alternates after each move
- **Consecutive passes** (`_consecutive_passes`): counts passes in a row
- **Game over flag** (`is_over`): set when the game ends
- **Resignation** (`_resigned_by`): records which player resigned, if any

### How a move is played

`game.play(move)` does:
1. Check the game isn't already over
2. Pass the move to the board (which checks legality and handles captures)
3. If it was a pass, increment the consecutive pass counter; otherwise reset it to 0
4. If there have been 2 consecutive passes, the game is over
5. Switch to the other player's turn

### How the game ends

Two ways:
- **Two consecutive passes**: both players agree the game is done. Score the board.
- **Resignation**: one player gives up. The other wins immediately (no scoring needed).

### Getting the result

`game.result()` returns a dictionary with the winner and scores. If the game ended by resignation, the winner is the opponent of whoever resigned. If it ended by passing, the board is scored using the Chinese rules described above.

---

## How These Pieces Connect to the AI

Here's the data flow when the AI eventually plays:

```
Game.play(move)
  → Board.is_legal(color, move)     # validates the move
  → Board._place_stone(...)         # updates the grid
  → Board._toggle_hash(...)         # updates the position fingerprint
  → Board._group(...)               # checks for captures via BFS

Game.legal_moves()
  → used by MCTS to know which moves to consider

Board state (the grid)
  → converted to tensor features
  → fed into the neural network
  → network outputs policy (where to play) + value (who's winning)
```

The board engine is called thousands of times during each AI move decision (once per MCTS simulation), so correctness here is non-negotiable — a single rules bug would corrupt the entire training process.
