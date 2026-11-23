"""
Baseline 1: "predict the move humans most often played in this position".

How it works
  1. Learn from the TRAIN split only: for every position (FEN without move
     counters) count which moves were played there.
  2. For a validation/test position, rank the moves by those counts.
  3. If the position was never seen in train (or has fewer than k different
     moves), fill the remaining guesses with the legal moves that are most
     common overall in train (e.g. "castle king-side" is common everywhere).

Two variants are reported side by side:
  * all-ratings:  counts from all players together (ignores rating).
  * by-band:      counts from players in the SAME rating band first, then
                  falls back to the all-ratings counts, then to the overall
                  move frequencies.
If "by-band" beats "all-ratings", knowing the rating already helps this very
simple model, which is a first small answer to the research question.

Usage (Terminal tab, in the project folder):
    python baseline_most_common.py dataset_all.npz
    python baseline_most_common.py dataset_all.npz --split test
    python baseline_most_common.py dataset_all.npz --csv results_most_common_val.csv
"""
import argparse
import logging
from collections import Counter, defaultdict

import chess
import numpy as np

from bands import BAND_WIDTH, MIN_RATING, TOP_BAND_START
from common import add_logging_args, load_dataset, setup_logging
from encoding import legal_move_indices
from evaluation import bands_of, log_table, report, summarize, train_players
from splits import TEST, TRAIN, VAL, position_key

log = logging.getLogger(__name__)

SPLIT_CODES = {"val": VAL, "test": TEST}


def count_moves(keys, moves):
    """{key: Counter({move: how often it was played})}."""
    counts = defaultdict(Counter)
    for key, move in zip(keys, moves):
        counts[key][move] += 1
    return dict(counts)


def rank_moves(fen, counters, overall_counts, k):
    """
    Return up to k move classes, best guess first.

    counters: list of Counters (or None), tried in order, most specific first.
    overall_counts: Counter of all train moves, used to fill the remaining slots
    with LEGAL moves only. Ties are broken by the smaller class index so the
    result is always the same.
    """
    ranked = []
    for counter in counters:
        if not counter:
            continue
        for move, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if move not in ranked:
                ranked.append(move)
            if len(ranked) == k:
                return ranked

    legal = legal_move_indices(chess.Board(fen))
    for move in sorted(legal, key=lambda m: (-overall_counts[m], m)):
        if move not in ranked:
            ranked.append(move)
        if len(ranked) == k:
            break
    return ranked


def evaluate(data, split_code, k=3, min_rating=MIN_RATING, band_width=BAND_WIDTH,
             top_band_start=TOP_BAND_START):
    """
    Fit on TRAIN, predict every row of `split_code`. Returns one dict per
    predicted row with the band, whether position / player were seen in train,
    and whether each variant got the move in its top-1 / top-k.
    """
    split = data["split"]
    fens = data["fens"].tolist()
    moves = data["moves"].tolist()
    players = data["players"].tolist()
    keys = [position_key(fen) for fen in fens]
    bands = bands_of(data["ratings"], min_rating, band_width, top_band_start)

    train_rows = np.flatnonzero(split == TRAIN)
    position_counts = count_moves([keys[i] for i in train_rows], [moves[i] for i in train_rows])
    band_position_counts = count_moves([(bands[i], keys[i]) for i in train_rows],
                                       [moves[i] for i in train_rows])
    overall_counts = Counter(moves[i] for i in train_rows)
    seen_players = train_players(data)
    log.info("Fitted on %d train positions (%d distinct positions)", len(train_rows), len(position_counts))

    results = []
    for i in np.flatnonzero(split == split_code):
        same_position = position_counts.get(keys[i])
        same_band_position = band_position_counts.get((bands[i], keys[i]))
        guesses_all = rank_moves(fens[i], [same_position], overall_counts, k)
        guesses_band = rank_moves(fens[i], [same_band_position, same_position], overall_counts, k)
        results.append({
            "band": bands[i],
            "seen_position": same_position is not None,
            "seen_player": players[i] in seen_players,
            "all_top1": moves[i] == guesses_all[0],
            "all_topk": moves[i] in guesses_all,
            "band_top1": moves[i] == guesses_band[0],
            "band_topk": moves[i] in guesses_band,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="merged .npz, e.g. dataset_all.npz")
    parser.add_argument("--split", choices=sorted(SPLIT_CODES), default="val",
                        help="which split to evaluate on (default: val; use test only at the very end)")
    parser.add_argument("--k", type=int, default=3, help="also report top-k accuracy (default 3)")
    parser.add_argument("--min-rating", type=int, default=MIN_RATING)
    parser.add_argument("--band-width", type=int, default=BAND_WIDTH)
    parser.add_argument("--top-band-start", type=int, default=TOP_BAND_START,
                        help="ratings from here up form one top band (default 2500 -> '2500+')")
    parser.add_argument("--csv", default=None, help="also save the per-band table to this CSV file")
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)

    data = load_dataset(args.dataset)
    results = evaluate(data, SPLIT_CODES[args.split], args.k, args.min_rating, args.band_width,
                       args.top_band_start)

    # Columns: all_* = counts from all ratings, band_* = same rating band first.
    report(results, f"Most-common-move baseline on {args.split} (k={args.k})", args.csv,
           args.band_width, args.top_band_start)
    log_table("By position (seen = this exact position occurs in train)",
              summarize(results, lambda r: "seen position" if r["seen_position"] else "new position"))


if __name__ == "__main__":
    main()
