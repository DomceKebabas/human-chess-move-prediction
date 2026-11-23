"""
Move-prediction models. All take the same inputs (see move_batches.py) and
return one logit per move class (NUM_MOVE_CLASSES = 4168):

    logits = model(boards, castling, rating)        # (B, 4168)

  * "linear": softmax regression = multinomial logistic regression on the
              flattened board (12x8x8 = 768) + 4 castling flags + rating
              (773 numbers). Baseline 3.
  * "mlp":    the same inputs with one hidden layer (optional baseline).
  * "cnn":    the main model. Castling flags and the rating are fed as extra
              constant 8x8 planes (12 + 4 + 1 = 17 input planes), followed by
              residual conv blocks and a small policy head.

Illegal moves are removed with masked_logits() before the softmax, so they
always get probability exactly 0.

build_model(config) rebuilds a model from the config stored in a checkpoint.
"""
import torch
from torch import nn
from torch.nn import functional as F

from encoding import NUM_MOVE_CLASSES

NUM_FLAT_INPUTS = 12 * 8 * 8 + 4 + 1


def masked_logits(logits, legal):
    """Illegal moves get -inf, so softmax gives them probability 0."""
    return logits.masked_fill(~legal, float("-inf"))


def flat_inputs(boards, castling, rating):
    return torch.cat([boards.flatten(1), castling, rating[:, None]], dim=1)


class FlatModel(nn.Module):
    """hidden=0: logistic regression. hidden>0: MLP with one hidden layer."""

    def __init__(self, hidden=0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(NUM_FLAT_INPUTS, hidden), nn.ReLU(),
                                     nn.Linear(hidden, NUM_MOVE_CLASSES))
        else:
            self.net = nn.Linear(NUM_FLAT_INPUTS, NUM_MOVE_CLASSES)

    def forward(self, boards, castling, rating):
        return self.net(flat_inputs(boards, castling, rating))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class MoveCNN(nn.Module):
    """
    17 input planes -> stem conv -> `blocks` residual blocks (2 convs each)
    -> 1x1 conv to `head_channels` -> linear to 4168 move logits.
    Default: 3 blocks = 6 conv layers in the body, 64 channels.
    """

    INPUT_PLANES = 12 + 4 + 1

    def __init__(self, channels=64, blocks=3, head_channels=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(self.INPUT_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(channels, head_channels, 1, bias=False),
                                  nn.BatchNorm2d(head_channels), nn.ReLU())
        self.policy = nn.Linear(head_channels * 64, NUM_MOVE_CLASSES)

    def forward(self, boards, castling, rating):
        batch = boards.shape[0]
        castling_planes = castling[:, :, None, None].expand(batch, 4, 8, 8)
        rating_plane = rating[:, None, None, None].expand(batch, 1, 8, 8)
        x = torch.cat([boards, castling_planes, rating_plane], dim=1)
        x = self.head(self.body(self.stem(x)))
        return self.policy(x.flatten(1))


def build_model(config):
    """config: {"model": "linear" | "mlp" | "cnn", plus that model's sizes}."""
    kind = config["model"]
    if kind == "linear":
        return FlatModel(hidden=0)
    if kind == "mlp":
        return FlatModel(hidden=config.get("hidden", 512))
    if kind == "cnn":
        return MoveCNN(config.get("channels", 64), config.get("blocks", 3), config.get("head_channels", 32))
    raise ValueError(f"unknown model {kind!r}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
