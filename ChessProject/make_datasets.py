"""
make_datasets.py - build one dataset per rating band, plus a combined one.

  1. Streams the .pgn.zst ONCE and takes a uniform random sample of
     --games-per-band qualifying games per band (reservoir sampling), so the
     sample covers the whole month instead of its first few hours.
     Writes filtered_<band>.pgn for each band that has games.
  2. Runs build_dataset.py on each band  ->  dataset_<band>.npz
  3. Merges all bands into dataset_all.npz and prints the split overlap report.

Bands
  By default the range --min-rating .. --max-rating is cut into bands of
  --band-width rating points (100-299, 300-499, ...). Both players of a game
  must be inside the same band and within --max-diff of each other. Bands with
  no games are simply skipped, so a generous --max-rating is harmless.
  You can still give explicit bands:  --bands 1000-1199,1400-1599

Memory: the sample is held in RAM, and so is the merged dataset. Roughly
  positions ~= bands * games-per-band * 15,  at about 1 KB per position.
Keep --games-per-band modest when using many bands (the default is 5000).

Because it must read the whole file, a real run takes a while. Time a partial
scan first:
  python make_datasets.py FILE.pgn.zst --max-scan-games 2000000
(With --max-scan-games the sample only covers the part of the file that was
scanned, so use it for timing and tests, not for the final dataset.)

Final run:
  python make_datasets.py lichess_db_standard_rated_2026-09.pgn.zst
"""
import argparse
import bisect
import logging
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common import add_logging_args, setup_logging
from filter_games import parse_headers, reject_reason_with_headers, stream_games
from splits import overlap_report

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def parse_bands(text):
    """'1000-1199,1400-1599' -> [(1000, 1199), (1400, 1599)]"""
    bands = []
    for part in text.split(","):
        low, high = part.strip().split("-")
        bands.append((int(low), int(high)))
    return bands


def generate_bands(min_rating, max_rating, width):
    """Consecutive, non-overlapping bands: (100, 299), (300, 499), ... up to max_rating."""
    bands = []
    low = min_rating
    while low <= max_rating:
        bands.append((low, min(low + width - 1, max_rating)))
        low += width
    return bands


def check_bands(bands):
    """Sort the bands and make sure none of them overlap."""
    bands = sorted(bands)
    for (low_a, high_a), (low_b, high_b) in zip(bands, bands[1:]):
        if low_b <= high_a:
            raise SystemExit(f"Bands {low_a}-{high_a} and {low_b}-{high_b} overlap.")
    return bands


def _band_index(headers, lows):
    """Index of the band containing the white player's rating, or None."""
    try:
        white_elo = int(headers["WhiteElo"])
    except (KeyError, ValueError):
        return None
    index = bisect.bisect_right(lows, white_elo) - 1
    return index if index >= 0 else None


def sample_into_bands(path, bands, args, output_dir=HERE):
    """One pass over the file; reservoir-sample qualifying games for every band."""
    names = [f"{low}-{high}" for low, high in bands]
    lows = [low for low, _ in bands]
    band_filters = [
        SimpleNamespace(min_rating=low, max_rating=high, max_diff=args.max_diff, min_plies=args.min_plies)
        for low, high in bands
    ]
    reservoirs = [[] for _ in bands]
    qualifying_seen = [0] * len(bands)
    rng = random.Random(args.seed)
    stats = {}
    next_report = 200_000

    for game_lines in stream_games(path, stats):
        headers = parse_headers(game_lines)
        band_index = _band_index(headers, lows)
        # The filter checks that BOTH players are inside this band.
        if band_index is not None and \
                reject_reason_with_headers(headers, game_lines, band_filters[band_index]) is None:
            qualifying_seen[band_index] += 1
            reservoir = reservoirs[band_index]
            if len(reservoir) < args.games_per_band:
                reservoir.append("".join(game_lines))
            else:
                slot = rng.randrange(qualifying_seen[band_index])
                if slot < args.games_per_band:
                    reservoir[slot] = "".join(game_lines)

        if stats["scanned"] >= next_report:
            log.info("scanned %s games | qualifying so far: %s in %d bands", f"{stats['scanned']:,}",
                     f"{sum(qualifying_seen):,}", sum(1 for s in qualifying_seen if s))
            next_report += 200_000
        if args.max_scan_games and stats["scanned"] >= args.max_scan_games:
            log.info("Stopped at --max-scan-games. The sample covers only the scanned part of the file.")
            break

    log.info("Sampling done. Scanned %s games.", f"{stats['scanned']:,}")
    empty = []
    for name, reservoir, seen in zip(names, reservoirs, qualifying_seen):
        if not reservoir:
            empty.append(name)
            continue
        warning = "   <-- fewer than requested" if len(reservoir) < args.games_per_band else ""
        log.info("  band %-10s %9s qualifying games found, %s sampled%s",
                 name, f"{seen:,}", f"{len(reservoir):,}", warning)
        with open(Path(output_dir) / f"filtered_{name}.pgn", "w", encoding="utf-8") as out:
            for game_text in reservoir:
                out.write(game_text)
                out.write("\n")
    if empty:
        log.info("  empty bands (no qualifying games, skipped): %s", ", ".join(empty))
    return names, [len(r) for r in reservoirs]


def build_dataset_command(source, output, num_games, args):
    """Command line for build_dataset.py; every position-level option is passed on."""
    return [sys.executable, str(HERE / "build_dataset.py"), str(source), str(output),
            "--max-games", str(num_games),
            "--min-clock", str(args.min_clock),
            "--min-ply", str(args.min_ply),
            "--max-per-game", str(args.max_per_game),
            "--min-game-plies", str(args.min_plies),
            "--seed", str(args.seed),
            "--quiet"]


def build_band(name, num_games, args):
    source = HERE / f"filtered_{name}.pgn"
    output = HERE / f"dataset_{name}.npz"
    command = build_dataset_command(source, output, num_games, args)
    result = subprocess.run(command)
    if result.returncode != 0 or not output.exists():
        log.info("  band %s: no positions produced", name)
        return None
    with np.load(output) as data:
        log.info("  band %-10s %s games -> %s positions", name, f"{num_games:,}", f"{len(data['moves']):,}")
    return output


def merge(files, output_path):
    parts = [np.load(f) for f in files]
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0].files}
    np.savez_compressed(output_path, **merged)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="lichess .pgn.zst file")
    parser.add_argument("--min-rating", type=int, default=100)
    parser.add_argument("--max-rating", type=int, default=3500,
                        help="ceiling for the generated bands; empty bands are skipped")
    parser.add_argument("--band-width", type=int, default=200)
    parser.add_argument("--bands", default=None,
                        help="explicit bands, e.g. 1000-1199,1400-1599 (overrides the range options)")
    parser.add_argument("--games-per-band", type=int, default=5000)
    parser.add_argument("--max-diff", type=int, default=100)
    parser.add_argument("--min-plies", type=int, default=20)
    parser.add_argument("--min-clock", type=float, default=30.0,
                        help="skip moves with fewer seconds left (passed to build_dataset.py)")
    parser.add_argument("--min-ply", type=int, default=10,
                        help="skip the first N plies of each game (passed to build_dataset.py)")
    parser.add_argument("--max-per-game", type=int, default=15,
                        help="keep at most N positions per game (passed to build_dataset.py)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for game sampling and position sampling")
    parser.add_argument("--max-scan-games", type=int, default=0,
                        help="stop after scanning this many games (0 = whole file)")
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)

    if args.bands:
        bands = check_bands(parse_bands(args.bands))
    else:
        bands = generate_bands(args.min_rating, args.max_rating, args.band_width)
    log.info("%d bands from %d to %d", len(bands), bands[0][0], bands[-1][1])

    names, sizes = sample_into_bands(args.input, bands, args)

    log.info("\nBuilding datasets")
    built = []
    for name, size in zip(names, sizes):
        if size:
            path = build_band(name, size, args)
            if path is not None:
                built.append(path)
    if not built:
        raise SystemExit("No band produced any positions. Loosen the filters or scan more games.")

    merged = merge(built, HERE / "dataset_all.npz")
    split, ratings = merged["split"], merged["ratings"]
    log.info("\n=== Combined dataset: dataset_all.npz ===")
    log.info("positions: %s   (train %s / val %s / test %s)", f"{len(ratings):,}",
             f"{int((split == 0).sum()):,}", f"{int((split == 1).sum()):,}", f"{int((split == 2).sum()):,}")
    log.info("rating range in the data: %d-%d", ratings.min(), ratings.max())
    overlap_report(merged)


if __name__ == "__main__":
    main()