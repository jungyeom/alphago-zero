"""
AlphaZero-style neural network: ResNet with policy and value heads.

Architecture:
  Input (8 channels) → Conv3x3 → BN → ReLU
    → N Residual Blocks (each: Conv3x3→BN→ReLU→Conv3x3→BN + skip → ReLU)
    → Policy Head: Conv1x1→BN→ReLU→Conv1x1→Flatten→Log-softmax over (size²+1) moves
    → Value Head: Conv1x1→BN→ReLU→Flatten→Dense(256)→ReLU→Dense(1)→Tanh

The network takes a board position and outputs:
  - policy: log-probability distribution over all moves (including pass)
  - value: scalar in [-1, +1] estimating who is winning (+1 = current player winning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.features import NUM_FEATURES


class ResidualBlock(nn.Module):
    """
    A single residual block:
      x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+x) → ReLU
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    Full AlphaZero network.

    Args:
        board_size: Size of the Go board (e.g. 13)
        num_blocks: Number of residual blocks (default: 6)
        num_filters: Number of convolutional filters (default: 128)
    """

    def __init__(
        self,
        board_size: int = 13,
        num_blocks: int = 6,
        num_filters: int = 128,
    ):
        super().__init__()
        self.board_size = board_size
        self.num_moves = board_size * board_size + 1  # board points + pass

        # Initial convolution: input features → num_filters channels
        self.conv_input = nn.Conv2d(NUM_FEATURES, num_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head
        # Conv1x1 to reduce channels, then flatten and project to move space
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_moves)

        # Value head
        # Conv1x1 to reduce channels, then flatten, dense layers, tanh
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, NUM_FEATURES, board_size, board_size)

        Returns:
            policy: log-probabilities of shape (batch, num_moves)
            value: scalar in [-1, 1] of shape (batch, 1)
        """
        # Shared trunk
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)  # flatten
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)  # flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference (no gradient tracking).

        Returns:
            policy: probabilities (not log) of shape (batch, num_moves)
            value: scalar in [-1, 1] of shape (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            log_policy, value = self.forward(x)
            policy = torch.exp(log_policy)
        return policy, value

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_network(
    board_size: int = 13,
    num_blocks: int = 6,
    num_filters: int = 128,
) -> AlphaZeroNet:
    """Create and initialize an AlphaZero network."""
    net = AlphaZeroNet(
        board_size=board_size,
        num_blocks=num_blocks,
        num_filters=num_filters,
    )
    # Initialize weights (Kaiming/He initialization for conv layers)
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return net
