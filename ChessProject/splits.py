"""
Train/validation/test split and a report on how much the splits overlap.

The split is decided per GAME from a hash of the game id, so all positions of
one game always land together. That alone does NOT make the splits independent:
the same position or the same player can still appear on both sides. The
overlap report measures that, so you can state the numbers in your write-up.

Usage on an existing dataset:
    python splits.py dataset_all.npz
"""
import argparse
import logging
import zlib

import numpy as np

from common import add_logging_args, setup_logging

log = logging.getLogger(__name__)

TRAIN, VAL, TEST = 0, 1, 2


def split_of(game_id):
    """Deterministic 80/10/10 split by game id."""
    bucket = zlib.crc32(game_id.encode("utf-8")) % 10
    if bucket < 8:
        return TRAIN
    return VAL if bucket == 8 else TEST


def position_key(fen):
    """
    FEN without the move counters. Two identical positions reached at different
    move numbers have different full FENs, so comparing full FENs would
    undercount repeated positions.
    """
    return " ".join(fen.split(" ")[:4])


def _percent(part, whole):
    return 100.0 * part / whole if whole else 0.0


def overlap_report(data):
    """
    data: dict of arrays with keys split, fens, game_ids, plies and
    (optionally) players, opponents. Returns the numbers and logs them.
    """
    split = data["split"].tolist()
    report = {}

    # 1. The same (game, ply) twice would mean duplicated data.
    row_keys = list(zip(data["game_ids"].tolist(), data["plies"].tolist()))
    report["duplicate_rows"] = len(row_keys) - len(set(row_keys))

    # 2. Positions that occur in both train and val/test.
    position_keys = [position_key(fen) for fen in data["fens"].tolist()]
    train_positions = {key for key, s in zip(position_keys, split) if s == TRAIN}
    for name, code in (("val", VAL), ("test", TEST)):
        rows = [key for key, s in zip(position_keys, split) if s == code]
        shared = sum(key in train_positions for key in rows)
        report[f"{name}_positions_also_in_train_pct"] = _percent(shared, len(rows))

    # 3. Players who appear in train games AND in val/test games.
    if "players" in data and "opponents" in data:
        players = data["players"].tolist()
        opponents = data["opponents"].tolist()
        train_players = set()
        for player, opponent, s in zip(players, opponents, split):
            if s == TRAIN:
                train_players.update((player, opponent))
        for name, code in (("val", VAL), ("test", TEST)):
            movers = [p for p, s in zip(players, split) if s == code]
            shared = sum(p in train_players for p in movers)
            report[f"{name}_movers_also_in_train_games_pct"] = _percent(shared, len(movers))

    log.info("Split overlap report")
    log.info("  duplicate (game, ply) rows:                 %d", report["duplicate_rows"])
    for name in ("val", "test"):
        log.info("  %-4s positions also seen in train:          %.1f%%",
                 name, report[f"{name}_positions_also_in_train_pct"])
        key = f"{name}_movers_also_in_train_games_pct"
        if key in report:
            log.info("  %-4s players who also play in train games:  %.1f%%", name, report[key])
    return report


def main():
    parser = argparse.ArgumentParser(description="Overlap report for a built dataset (.npz)")
    parser.add_argument("dataset")
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)
    loaded = np.load(args.dataset)
    overlap_report({key: loaded[key] for key in loaded.files})


if __name__ == "__main__":
    main()