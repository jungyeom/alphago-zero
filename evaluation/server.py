"""
FastAPI backend for playing against the AI.

Endpoints:
  POST /api/new          — start a new game
  GET  /api/game         — get current game state
  POST /api/move         — play a human move, get AI response
  POST /api/pass         — human passes
  POST /api/resign       — human resigns
  GET  /api/ai-thinking  — get AI's move analysis (top moves, win %)
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from go_engine.board import BLACK, WHITE, EMPTY
from go_engine.game import Game
from model.network import create_network
from mcts.search import search_with_net
from training.trainer import load_checkpoint

app = FastAPI(title="AlphaZero Go")

# ── Global state ────────────────────────────────────────────────────

game: Game | None = None
net: torch.nn.Module | None = None
device: torch.device = torch.device("cpu")
board_size: int = 13
num_simulations: int = 200
human_color: int = BLACK


# ── Request/Response models ─────────────────────────────────────────

class NewGameRequest(BaseModel):
    board_size: int = 13
    human_color: str = "black"  # "black" or "white"
    simulations: int = 200
    model_path: str | None = None


class MoveRequest(BaseModel):
    row: int
    col: int


class GameState(BaseModel):
    board: list[list[int]]
    current_player: str
    is_over: bool
    human_color: str
    captures_black: int
    captures_white: int
    last_move: list[int] | None
    move_count: int
    result: dict | None = None


class AIAnalysis(BaseModel):
    top_moves: list[dict]
    win_rate: float
    total_simulations: int


# ── Helpers ─────────────────────────────────────────────────────────

def get_game_state() -> GameState:
    """Build the current game state response."""
    if game is None:
        raise HTTPException(status_code=400, detail="No game in progress. POST /api/new first.")

    board = game.board.grid.tolist()
    current = "black" if game.current_player == BLACK else "white"
    h_color = "black" if human_color == BLACK else "white"

    last_move = None
    if game.board.move_history:
        _, lm = game.board.move_history[-1]
        if lm is not None:
            last_move = list(lm)

    result = None
    if game.is_over:
        result = game.result()

    return GameState(
        board=board,
        current_player=current,
        is_over=game.is_over,
        human_color=h_color,
        captures_black=game.board.captured[BLACK],
        captures_white=game.board.captured[WHITE],
        last_move=last_move,
        move_count=len(game.board.move_history),
        result=result,
    )


def ai_play() -> dict | None:
    """Have the AI play a move. Returns the move played or None."""
    if game is None or game.is_over:
        return None
    if game.current_player == human_color:
        return None  # not AI's turn

    if net is None:
        # No model loaded — play random
        import random
        legal = game.legal_moves()
        move = random.choice([m for m in legal if m is not None] or [None])
        game.play(move)
        return {"row": move[0], "col": move[1]} if move else None

    root = search_with_net(
        game.board, game.current_player, net,
        num_simulations=num_simulations, device=device,
    )
    best = root.most_visited_child()
    move = best.move
    game.play(move)
    return {"row": move[0], "col": move[1]} if move else None


# ── Endpoints ───────────────────────────────────────────────────────

@app.post("/api/new")
def new_game(req: NewGameRequest):
    """Start a new game."""
    global game, net, device, board_size, num_simulations, human_color

    board_size = req.board_size
    num_simulations = req.simulations
    human_color = BLACK if req.human_color == "black" else WHITE

    game = Game(size=board_size)

    # Load model: explicit path > auto-detect latest checkpoint > fresh network
    if req.model_path and os.path.exists(req.model_path):
        net, _ = load_checkpoint(req.model_path, device)
        net.eval()
        print(f"Loaded model: {req.model_path}")
    else:
        # Auto-detect: look for model_final.pt or latest checkpoint
        ckpt_dir = "checkpoints"
        auto_path = None
        if os.path.exists(os.path.join(ckpt_dir, "model_final.pt")):
            auto_path = os.path.join(ckpt_dir, "model_final.pt")
        elif os.path.isdir(ckpt_dir):
            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
            if ckpts:
                auto_path = os.path.join(ckpt_dir, ckpts[-1])

        if auto_path:
            try:
                net, ckpt = load_checkpoint(auto_path, device)
                # Verify board size matches
                if ckpt["config"]["board_size"] == board_size:
                    net.eval()
                    print(f"Auto-loaded model: {auto_path}")
                else:
                    print(f"Checkpoint board size mismatch ({ckpt['config']['board_size']} vs {board_size}), using fresh network")
                    net = create_network(board_size=board_size, num_blocks=2, num_filters=32)
                    net.to(device)
                    net.eval()
            except Exception as e:
                print(f"Failed to load {auto_path}: {e}, using fresh network")
                net = create_network(board_size=board_size, num_blocks=2, num_filters=32)
                net.to(device)
                net.eval()
        else:
            net = create_network(board_size=board_size, num_blocks=2, num_filters=32)
            net.to(device)
            net.eval()
            print("No checkpoint found, using fresh (untrained) network")

    # If AI plays first (human is white), make the AI move
    ai_move = None
    if human_color == WHITE:
        ai_move = ai_play()

    return {"status": "ok", "ai_move": ai_move, "game": get_game_state()}


@app.get("/api/game")
def get_game():
    """Get current game state."""
    return get_game_state()


@app.post("/api/move")
def play_move(req: MoveRequest):
    """Human plays a move, then AI responds."""
    if game is None:
        raise HTTPException(400, "No game in progress")
    if game.is_over:
        raise HTTPException(400, "Game is over")
    if game.current_player != human_color:
        raise HTTPException(400, "Not your turn")

    move = (req.row, req.col)
    if not game.board.is_legal(human_color, move):
        raise HTTPException(400, f"Illegal move: ({req.row}, {req.col})")

    game.play(move)

    # AI responds
    ai_move = None
    if not game.is_over:
        ai_move = ai_play()

    return {"ai_move": ai_move, "game": get_game_state()}


@app.post("/api/pass")
def play_pass():
    """Human passes."""
    if game is None:
        raise HTTPException(400, "No game in progress")
    if game.is_over:
        raise HTTPException(400, "Game is over")
    if game.current_player != human_color:
        raise HTTPException(400, "Not your turn")

    game.play(None)

    ai_move = None
    if not game.is_over:
        ai_move = ai_play()

    return {"ai_move": ai_move, "game": get_game_state()}


@app.post("/api/resign")
def resign():
    """Human resigns."""
    if game is None:
        raise HTTPException(400, "No game in progress")
    if game.is_over:
        raise HTTPException(400, "Game is over")

    game.resign()
    return {"game": get_game_state()}


@app.get("/api/ai-thinking")
def ai_thinking():
    """Get AI's analysis of the current position."""
    if game is None or net is None:
        raise HTTPException(400, "No game or model")
    if game.is_over:
        raise HTTPException(400, "Game is over")

    color = game.current_player
    root = search_with_net(
        game.board, color, net,
        num_simulations=num_simulations, device=device,
    )

    # Top 5 moves by visit count
    ranked = sorted(root.children, key=lambda n: n.visit_count, reverse=True)[:5]
    top_moves = []
    for node in ranked:
        move_str = f"({node.move[0]},{node.move[1]})" if node.move else "pass"
        top_moves.append({
            "move": move_str,
            "row": node.move[0] if node.move else None,
            "col": node.move[1] if node.move else None,
            "visits": node.visit_count,
            "win_rate": round((node.q_value + 1) / 2 * 100, 1),  # convert to 0-100%
        })

    # Overall win rate from AI's perspective
    if root.visit_count > 0:
        raw_q = root.q_value
        # Convert from black-perspective to current-player-perspective
        if color == WHITE:
            raw_q = -raw_q
        win_pct = round((raw_q + 1) / 2 * 100, 1)
    else:
        win_pct = 50.0

    return AIAnalysis(
        top_moves=top_moves,
        win_rate=win_pct,
        total_simulations=root.visit_count,
    )


# ── Serve frontend static files ────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "web", "dist")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("evaluation.server:app", host="0.0.0.0", port=8000, reload=True)
