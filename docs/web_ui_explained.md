# How the Web UI Works

The web frontend lets you play Go against the trained AI in your browser.

---

## 1. Architecture

```
Browser (React)  ←→  FastAPI server  ←→  Go engine + MCTS + Neural net
     port 5173           port 8000
     (Vite dev)          (uvicorn)
```

The frontend is a React + TypeScript app that communicates with a Python backend via REST API. During development, Vite proxies `/api` requests to the backend.

For production, the frontend is built into static files (`web/dist/`) and served directly by FastAPI.

---

## 2. Running It

### Development (two terminals)

**Terminal 1 — Backend:**
```bash
cd alphago-zero
uv run python -m evaluation.server
```
Starts FastAPI on `http://localhost:8000`.

**Terminal 2 — Frontend:**
```bash
cd alphago-zero/web
npm run dev
```
Opens Vite dev server on `http://localhost:5173` with hot reload.

### Production (single server)
```bash
cd alphago-zero/web && npm run build
cd .. && uv run python -m evaluation.server
```
Visit `http://localhost:8000` — FastAPI serves both the API and the built frontend.

### Loading a trained model
The server creates a small untrained network by default. To load a trained checkpoint, pass it in the "New Game" API request or modify the server startup.

---

## 3. API Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/api/new` | `{board_size, human_color, simulations}` | New game state |
| `GET` | `/api/game` | — | Current game state |
| `POST` | `/api/move` | `{row, col}` | AI response + game state |
| `POST` | `/api/pass` | — | AI response + game state |
| `POST` | `/api/resign` | — | Final game state |
| `GET` | `/api/ai-thinking` | — | Top moves, win rate |

### Game state format
```json
{
  "board": [[0,0,1,...], ...],     // 0=empty, 1=black, 2=white
  "current_player": "black",
  "is_over": false,
  "human_color": "black",
  "captures_black": 3,
  "captures_white": 1,
  "last_move": [4, 5],            // or null
  "move_count": 42,
  "result": null                   // or {winner, margin, reason} when game ends
}
```

---

## 4. The Go Board Component (`Board.tsx`)

The board is rendered as an SVG with:
- **Grid lines** with proper edge thickness
- **Star points** (hoshi) for 9x9, 13x13, and 19x19
- **Stones** as circles (black #111, white #f5f5f5)
- **Last move marker** (red circle on the most recent stone)
- **Column labels** (A-T, skipping I per Go convention)
- **Row labels** (numbered from bottom)
- **Click targets** (invisible rectangles over empty intersections)

The board auto-scales based on size — a 5x5 board uses larger cells than a 13x13.

---

## 5. Game Flow

1. **Setup screen**: Choose board size (5/9/13) and color (black/white)
2. Click "New Game" — sends `POST /api/new`, receives initial board
3. If you're white, the AI plays first automatically
4. **Your turn**: click an intersection to place a stone
   - Sends `POST /api/move {row, col}`
   - Server validates the move, plays it, then runs MCTS for the AI's response
   - Returns both the AI's move and the updated board
5. **Pass/Resign**: buttons in the sidebar
6. **Game over**: shows winner, score, and reason (scoring or resignation)
7. Click "New Game" to return to setup

---

## 6. What the AI "Thinks"

The `/api/ai-thinking` endpoint runs MCTS on the current position and returns:
- Top 5 moves by visit count
- Win rate for each move
- Overall position evaluation

This data is available for displaying AI analysis in the UI (e.g., showing top moves overlaid on the board).
