const BASE = "/api";

export interface GameState {
  board: number[][];
  current_player: string;
  is_over: boolean;
  human_color: string;
  captures_black: number;
  captures_white: number;
  last_move: [number, number] | null;
  move_count: number;
  result: {
    winner: string;
    black: number;
    white: number;
    margin: number;
    reason: string;
  } | null;
}

export interface MoveResponse {
  ai_move: { row: number; col: number } | null;
  game: GameState;
}

export interface NewGameResponse {
  status: string;
  ai_move: { row: number; col: number } | null;
  game: GameState;
}

export interface AIThinking {
  top_moves: {
    move: string;
    row: number | null;
    col: number | null;
    visits: number;
    win_rate: number;
  }[];
  win_rate: number;
  total_simulations: number;
}

export async function newGame(
  boardSize: number = 13,
  humanColor: string = "black",
  simulations: number = 200,
  modelPath?: string
): Promise<NewGameResponse> {
  const res = await fetch(`${BASE}/new`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      board_size: boardSize,
      human_color: humanColor,
      simulations,
      model_path: modelPath || null,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getGame(): Promise<GameState> {
  const res = await fetch(`${BASE}/game`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function playMove(
  row: number,
  col: number
): Promise<MoveResponse> {
  const res = await fetch(`${BASE}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function playPass(): Promise<MoveResponse> {
  const res = await fetch(`${BASE}/pass`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resign(): Promise<MoveResponse> {
  const res = await fetch(`${BASE}/resign`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getAIThinking(): Promise<AIThinking> {
  const res = await fetch(`${BASE}/ai-thinking`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
