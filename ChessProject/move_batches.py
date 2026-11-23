"""
Turn dataset rows into PyTorch batches.

The whole dataset is held in memory as numpy arrays (boards ~640 MB as uint8
for 831k rows); only the rows of the current batch are converted to float
tensors. This is simpler and faster than a per-item torch Dataset.

A batch is a dict of tensors:
    boards    (B, 12, 8, 8) float   mover's view, see encoding.encode_board
    castling  (B, 4) float
    rating    (B,) float            normalised: (rating - 1500) / 1000,
                                    or 0 for every row when use_rating=False
    legal     (B, 4168) bool        legal-move mask (from legal_cache)
    target    (B,) long             the move the human played

"Without rating" (use_rating=False) feeds the SAME model a constant rating of
0, so models with and without rating have exactly the same architecture and
differ only in what they are told.
"""
import numpy as np
import torch

import legal_cache

RATING_CENTER = 1500.0
RATING_SCALE = 1000.0


def normalise_rating(ratings):
    return (np.asarray(ratings, dtype=np.float32) - RATING_CENTER) / RATING_SCALE


class MoveBatches:
    def __init__(self, data, legal_flat, legal_offsets, use_rating=True):
        self.boards = data["boards"]
        self.castling = data["castling"]
        self.ratings = normalise_rating(data["ratings"])
        self.moves = data["moves"].astype(np.int64)
        self.legal_flat = legal_flat
        self.legal_offsets = legal_offsets
        self.use_rating = use_rating

    def batch(self, rows, device="cpu"):
        rows = np.asarray(rows)
        rating = self.ratings[rows] if self.use_rating else np.zeros(len(rows), dtype=np.float32)
        tensors = {
            "boards": torch.from_numpy(self.boards[rows].astype(np.float32)),
            "castling": torch.from_numpy(self.castling[rows].astype(np.float32)),
            "rating": torch.from_numpy(rating),
            "legal": torch.from_numpy(legal_cache.masks(self.legal_flat, self.legal_offsets, rows)),
            "target": torch.from_numpy(self.moves[rows]),
        }
        return {key: value.to(device) for key, value in tensors.items()}

    def iterate(self, rows, batch_size, shuffle=False, rng=None, device="cpu"):
        """Yield batches over `rows` (shuffled each call if shuffle=True)."""
        rows = np.asarray(rows)
        if shuffle:
            rows = (rng or np.random.default_rng()).permutation(rows)
        for start in range(0, len(rows), batch_size):
            yield self.batch(rows[start:start + batch_size], device)
