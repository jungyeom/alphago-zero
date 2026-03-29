import { useState, useCallback } from "react";
import Board from "./Board";
import type { GameState } from "./api";
import { newGame, playMove, playPass, resign } from "./api";
import "./App.css";

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [boardSize, setBoardSize] = useState(9);
  const [humanColor, setHumanColor] = useState("black");

  const handleNewGame = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await newGame(boardSize, humanColor, 100);
      setGameState(res.game);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start game");
    }
    setLoading(false);
  }, [boardSize, humanColor]);

  const handlePlay = useCallback(
    async (row: number, col: number) => {
      if (!gameState || loading) return;
      setLoading(true);
      setError(null);
      try {
        const res = await playMove(row, col);
        setGameState(res.game);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Illegal move");
      }
      setLoading(false);
    },
    [gameState, loading]
  );

  const handlePass = useCallback(async () => {
    if (!gameState || loading) return;
    setLoading(true);
    try {
      const res = await playPass();
      setGameState(res.game);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error");
    }
    setLoading(false);
  }, [gameState, loading]);

  const handleResign = useCallback(async () => {
    if (!gameState || loading) return;
    setLoading(true);
    try {
      const res = await resign();
      setGameState(res.game);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error");
    }
    setLoading(false);
  }, [gameState, loading]);

  return (
    <div className="app">
      <h1>AlphaZero Go</h1>

      {!gameState && (
        <div className="setup">
          <div className="option">
            <label>Board Size</label>
            <select value={boardSize} onChange={(e) => setBoardSize(Number(e.target.value))}>
              <option value={5}>5x5</option>
              <option value={9}>9x9</option>
              <option value={13}>13x13</option>
            </select>
          </div>
          <div className="option">
            <label>Play as</label>
            <select value={humanColor} onChange={(e) => setHumanColor(e.target.value)}>
              <option value="black">Black (first)</option>
              <option value="white">White (second)</option>
            </select>
          </div>
          <button className="primary" onClick={handleNewGame} disabled={loading}>
            {loading ? "Starting..." : "New Game"}
          </button>
        </div>
      )}

      {gameState && (
        <div className="game">
          <div className="board-container">
            <Board
              board={gameState.board}
              lastMove={gameState.last_move}
              humanColor={gameState.human_color}
              currentPlayer={gameState.current_player}
              isOver={gameState.is_over}
              onPlay={handlePlay}
            />
          </div>

          <div className="sidebar">
            <div className="info">
              <div className="status">
                {gameState.is_over ? (
                  <span className="game-over">
                    Game Over: {gameState.result?.winner} wins
                    {gameState.result?.reason === "resign"
                      ? " by resignation"
                      : ` by ${gameState.result?.margin.toFixed(1)} pts`}
                  </span>
                ) : loading ? (
                  <span className="thinking">AI thinking...</span>
                ) : gameState.current_player === gameState.human_color ? (
                  <span>Your turn ({gameState.human_color})</span>
                ) : (
                  <span>AI's turn</span>
                )}
              </div>

              <div className="captures">
                <span>Black captures: {gameState.captures_black}</span>
                <span>White captures: {gameState.captures_white}</span>
              </div>

              <div className="move-count">Move {gameState.move_count}</div>
            </div>

            <div className="actions">
              <button
                onClick={handlePass}
                disabled={loading || gameState.is_over || gameState.current_player !== gameState.human_color}
              >
                Pass
              </button>
              <button
                onClick={handleResign}
                disabled={loading || gameState.is_over}
              >
                Resign
              </button>
              <button className="primary" onClick={() => { setGameState(null); setError(null); }}>
                New Game
              </button>
            </div>
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
}

export default App;
