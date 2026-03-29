import { useCallback } from "react";

interface BoardProps {
  board: number[][];
  lastMove: [number, number] | null;
  humanColor: string;
  currentPlayer: string;
  isOver: boolean;
  onPlay: (row: number, col: number) => void;
}

const EMPTY = 0;
const BLACK = 1;

export default function Board({
  board,
  lastMove,
  humanColor,
  currentPlayer,
  isOver,
  onPlay,
}: BoardProps) {
  const size = board.length;
  const cellSize = Math.min(36, Math.floor(560 / size));
  const padding = cellSize;
  const boardPixels = (size - 1) * cellSize;
  const svgSize = boardPixels + padding * 2;
  const stoneRadius = cellSize * 0.44;

  const isMyTurn = currentPlayer === humanColor && !isOver;

  // Star points for common board sizes
  const starPoints: [number, number][] = [];
  if (size === 9) {
    for (const r of [2, 4, 6]) for (const c of [2, 4, 6]) starPoints.push([r, c]);
  } else if (size === 13) {
    for (const r of [3, 6, 9]) for (const c of [3, 6, 9]) starPoints.push([r, c]);
  } else if (size === 19) {
    for (const r of [3, 9, 15]) for (const c of [3, 9, 15]) starPoints.push([r, c]);
  }

  const toX = (col: number) => padding + col * cellSize;
  const toY = (row: number) => padding + row * cellSize;

  const handleClick = useCallback(
    (row: number, col: number) => {
      if (!isMyTurn) return;
      if (board[row][col] !== EMPTY) return;
      onPlay(row, col);
    },
    [board, isMyTurn, onPlay]
  );

  return (
    <svg
      width={svgSize}
      height={svgSize}
      style={{ background: "#DCB35C", borderRadius: 4, cursor: isMyTurn ? "pointer" : "default" }}
    >
      {/* Grid lines */}
      {Array.from({ length: size }, (_, i) => (
        <g key={`lines-${i}`}>
          <line
            x1={toX(0)} y1={toY(i)} x2={toX(size - 1)} y2={toY(i)}
            stroke="#333" strokeWidth={i === 0 || i === size - 1 ? 1.5 : 0.8}
          />
          <line
            x1={toX(i)} y1={toY(0)} x2={toX(i)} y2={toY(size - 1)}
            stroke="#333" strokeWidth={i === 0 || i === size - 1 ? 1.5 : 0.8}
          />
        </g>
      ))}

      {/* Star points */}
      {starPoints.map(([r, c]) => (
        <circle
          key={`star-${r}-${c}`}
          cx={toX(c)} cy={toY(r)} r={3}
          fill="#333"
        />
      ))}

      {/* Click targets (invisible, full cells) */}
      {isMyTurn &&
        Array.from({ length: size }, (_, r) =>
          Array.from({ length: size }, (_, c) => {
            if (board[r][c] !== EMPTY) return null;
            return (
              <rect
                key={`click-${r}-${c}`}
                x={toX(c) - cellSize / 2}
                y={toY(r) - cellSize / 2}
                width={cellSize}
                height={cellSize}
                fill="transparent"
                onClick={() => handleClick(r, c)}
              />
            );
          })
        )}

      {/* Stones */}
      {board.map((row, r) =>
        row.map((cell, c) => {
          if (cell === EMPTY) return null;
          const isBlack = cell === BLACK;
          const isLast = lastMove && lastMove[0] === r && lastMove[1] === c;
          return (
            <g key={`stone-${r}-${c}`}>
              <circle
                cx={toX(c)} cy={toY(r)} r={stoneRadius}
                fill={isBlack ? "#111" : "#f5f5f5"}
                stroke={isBlack ? "#000" : "#888"}
                strokeWidth={1}
              />
              {/* Last move marker */}
              {isLast && (
                <circle
                  cx={toX(c)} cy={toY(r)} r={stoneRadius * 0.3}
                  fill="none"
                  stroke={isBlack ? "#e44" : "#e44"}
                  strokeWidth={2}
                />
              )}
            </g>
          );
        })
      )}

      {/* Column labels */}
      {Array.from({ length: size }, (_, i) => {
        const label = String.fromCharCode(65 + i + (i >= 8 ? 1 : 0));
        return (
          <g key={`label-${i}`}>
            <text
              x={toX(i)} y={padding - cellSize * 0.4}
              textAnchor="middle" fontSize={11} fill="#555"
            >
              {label}
            </text>
            <text
              x={toX(i)} y={svgSize - padding + cellSize * 0.6}
              textAnchor="middle" fontSize={11} fill="#555"
            >
              {label}
            </text>
          </g>
        );
      })}

      {/* Row labels */}
      {Array.from({ length: size }, (_, i) => {
        const label = String(size - i);
        return (
          <g key={`rowlabel-${i}`}>
            <text
              x={padding - cellSize * 0.55} y={toY(i) + 4}
              textAnchor="middle" fontSize={11} fill="#555"
            >
              {label}
            </text>
            <text
              x={svgSize - padding + cellSize * 0.55} y={toY(i) + 4}
              textAnchor="middle" fontSize={11} fill="#555"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
