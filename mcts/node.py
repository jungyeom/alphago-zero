"""
MCTS tree node.

Each node represents a board state reached by a specific move.
It stores visit statistics used by the selection formula (UCB1 / PUCT).
"""

import math


class Node:
    """
    A single node in the MCTS search tree.

    Attributes:
        move: The move that led to this node (None for root, or (row, col), or "pass")
        parent: Parent node (None for root)
        children: List of child nodes (empty until expanded)
        color: The color that PLAYED the move to reach this node
        visit_count: Number of times this node has been visited (N)
        total_value: Sum of all backpropagated values through this node (W)
        prior: Prior probability from neural net or uniform (P)
    """

    __slots__ = [
        "move",
        "parent",
        "children",
        "color",
        "visit_count",
        "total_value",
        "prior",
    ]

    def __init__(
        self,
        move: tuple[int, int] | None = None,
        parent: "Node | None" = None,
        color: int = 0,
        prior: float = 0.0,
    ):
        self.move = move
        self.parent = parent
        self.children: list[Node] = []
        self.color = color
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    @property
    def q_value(self) -> float:
        """Average value (W/N). Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1_score(self, c: float = 1.4) -> float:
        """
        UCB1 selection score (for MCTS without neural net).

        score = Q + c * sqrt(ln(N_parent) / N_child)

        Unvisited nodes get infinite score (always explored first).
        """
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        return self.q_value + exploration

    def puct_score(self, c_puct: float = 1.5) -> float:
        """
        PUCT selection score (for MCTS with neural net).

        score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

        This uses the prior probability P from the neural network to guide
        exploration toward moves the net thinks are promising.
        """
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    @property
    def is_expanded(self) -> bool:
        """A node is expanded if it has children."""
        return len(self.children) > 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def best_child(self, use_puct: bool = False, c: float = 1.4) -> "Node":
        """Select the child with the highest UCB1 or PUCT score."""
        if not self.children:
            raise ValueError("Cannot select best child from unexpanded node")
        if use_puct:
            return max(self.children, key=lambda n: n.puct_score(c))
        return max(self.children, key=lambda n: n.ucb1_score(c))

    def most_visited_child(self) -> "Node":
        """Select the child with the most visits (used for final move selection)."""
        if not self.children:
            raise ValueError("No children")
        return max(self.children, key=lambda n: n.visit_count)

    def __repr__(self) -> str:
        return (
            f"Node(move={self.move}, N={self.visit_count}, "
            f"Q={self.q_value:.3f}, P={self.prior:.3f})"
        )
