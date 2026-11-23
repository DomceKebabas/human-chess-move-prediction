"""
Baseline 2: "how often does the human play Stockfish's move?"

For a fixed random sample of positions per rating band (default 2,000 val
positions per band) Stockfish is asked for its 3 best moves (MultiPV). We
report how often the human move is Stockfish's top move (top-1) and among its
top 3 (top-3). Expect this to rise with rating: strong players play more
engine-like moves.

The stored FEN is the position seen from the mover's side (mirrored when Black
is to move). A mirrored position is the same chess problem with the colours
swapped, so Stockfish analyses the stored FEN directly and its moves are
encoded with encode_move(move, mirrored=False), the same class space as the
dataset's "moves" column.

Engine output cache (--cache, CSV)
  One line per analysed position: row index, Stockfish's top-k move classes,
  the evaluation of the best move and of the HUMAN move (centipawns, from the
  mover's point of view; mate = +-10000). Phase 6 reuses the evaluations for
  the blunder analysis. Re-running with the same --cache skips the positions
  that were already analysed, so a long run can be stopped and resumed.

Installing Stockfish on Windows
  Download the Windows build from https://stockfishchess.org/download/, unzip
  it, and pass the .exe:  --engine C:\\tools\\stockfish\\stockfish-windows-x86-64-avx2.exe
  (or set the STOCKFISH_PATH environment variable once).

Usage:
    python baseline_stockfish.py dataset_all.npz --engine PATH --csv results_stockfish_val.csv
    python baseline_stockfish.py dataset_all.npz --per-band 200 --depth 8     # quick test run
"""
import argparse
import csv
import logging
import os
import shutil
import time
from pathlib import Path

import chess
import chess.engine

from common import add_logging_args, load_dataset, setup_logging
from encoding import encode_move
from evaluation import bands_of, report, sample_rows_per_band, train_players
from splits import TEST, VAL

log = logging.getLogger(__name__)

SPLIT_CODES = {"val": VAL, "test": TEST}
MATE_SCORE = 10_000
CACHE_FIELDS = ["row", "engine_moves", "best_cp", "human_cp"]


def score_cp(info, color):
    """Centipawns from `color`'s point of view; mates become +-MATE_SCORE."""
    return info["score"].pov(color).score(mate_score=MATE_SCORE)


def analyse_position(engine, fen, human_move_class, limit, k=3):
    """
    Ask the engine for its k best moves in the (mover's view) position `fen`.

    Returns a cache record: engine move classes (best first), the evaluation of
    the best move, and the evaluation of the human's move (analysed separately
    with root_moves, so it is comparable to best_cp).
    """
    board = chess.Board(fen)
    infos = engine.analyse(board, limit, multipv=k)
    if isinstance(infos, dict):          # python-chess returns a dict when multipv=1
        infos = [infos]
    engine_moves = [encode_move(info["pv"][0], mirrored=False) for info in infos if info.get("pv")]
    best_cp = score_cp(infos[0], board.turn)

    human_move = next(m for m in board.legal_moves if encode_move(m, mirrored=False) == human_move_class)
    if engine_moves and engine_moves[0] == human_move_class:
        human_cp = best_cp
    else:
        human_info = engine.analyse(board, limit, root_moves=[human_move])
        human_cp = score_cp(human_info, board.turn)
    return {"engine_moves": engine_moves, "best_cp": best_cp, "human_cp": human_cp}


def read_cache(path):
    """{row: record} from an existing cache CSV (empty if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return {}
    cache = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for line in csv.DictReader(handle):
            cache[int(line["row"])] = {
                "engine_moves": [int(m) for m in line["engine_moves"].split()],
                "best_cp": int(line["best_cp"]),
                "human_cp": int(line["human_cp"]),
            }
    return cache


def open_cache_for_append(path):
    path = Path(path)
    is_new = not path.exists()
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
    if is_new:
        writer.writeheader()
    return handle, writer


def analyse_rows(engine, data, rows, limit, k, cache_path):
    """Analyse every row not yet in the cache; returns {row: record} for all rows."""
    cache = read_cache(cache_path)
    todo = [row for row in rows if row not in cache]
    log.info("%d positions to analyse (%d already cached)", len(todo), len(rows) - len(todo))
    if not todo:
        return cache

    handle, writer = open_cache_for_append(cache_path)
    started = time.time()
    try:
        for done, row in enumerate(todo, 1):
            record = analyse_position(engine, str(data["fens"][row]), int(data["moves"][row]), limit, k)
            cache[row] = record
            writer.writerow({"row": row, "engine_moves": " ".join(map(str, record["engine_moves"])),
                             "best_cp": record["best_cp"], "human_cp": record["human_cp"]})
            if done % 200 == 0:
                handle.flush()
                rate = done / (time.time() - started)
                log.info("analysed %d / %d  (%.1f positions/s, about %.0f min left)",
                         done, len(todo), rate, (len(todo) - done) / rate / 60)
    finally:
        handle.close()
    return cache


def result_rows(data, rows, cache, bands, seen_players):
    """Result rows for evaluation.report: top1 / topk agreement with the engine."""
    results = []
    for row in rows:
        human = int(data["moves"][row])
        engine_moves = cache[row]["engine_moves"]
        results.append({
            "band": bands[row],
            "seen_player": str(data["players"][row]) in seen_players,
            "top1": bool(engine_moves) and engine_moves[0] == human,
            "topk": human in engine_moves,
        })
    return results


def find_engine(path):
    path = path or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not path:
        raise SystemExit("Stockfish not found. Pass --engine PATH or set STOCKFISH_PATH "
                         "(see the top of this file for the Windows download).")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="merged .npz, e.g. dataset_all.npz")
    parser.add_argument("--engine", default=None, help="path to the Stockfish executable")
    parser.add_argument("--split", choices=sorted(SPLIT_CODES), default="val")
    parser.add_argument("--per-band", type=int, default=2000, help="positions per rating band")
    parser.add_argument("--depth", type=int, default=12, help="search depth per position")
    parser.add_argument("--k", type=int, default=3, help="number of engine moves (MultiPV)")
    parser.add_argument("--threads", type=int, default=1, help="1 keeps results repeatable")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing the positions")
    parser.add_argument("--cache", default=None,
                        help="engine output CSV (default: stockfish_cache_<split>_d<depth>.csv)")
    parser.add_argument("--csv", default=None, help="save the per-band table to this CSV file")
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)

    data = load_dataset(args.dataset)
    bands = bands_of(data["ratings"])
    split_rows = [int(i) for i in (data["split"] == SPLIT_CODES[args.split]).nonzero()[0]]
    rows = sample_rows_per_band(split_rows, bands, args.per_band, args.seed)
    cache_path = args.cache or f"stockfish_cache_{args.split}_d{args.depth}.csv"

    engine = chess.engine.SimpleEngine.popen_uci(find_engine(args.engine))
    try:
        engine.configure({"Threads": args.threads, "Hash": 64})
        cache = analyse_rows(engine, data, rows, chess.engine.Limit(depth=args.depth), args.k, cache_path)
    finally:
        engine.quit()

    results = result_rows(data, rows, cache, bands, train_players(data))
    report(results, f"Stockfish agreement (depth {args.depth}, top-{args.k}) on {args.split}", args.csv)


if __name__ == "__main__":
    main()
