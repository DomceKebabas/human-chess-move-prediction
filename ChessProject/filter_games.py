"""
filter_games.py - keep only clean, rated Rapid/Classical games from a Lichess dump.

Filters
  * Event is "Rated Rapid ..." or "Rated Classical ..." (Lichess's own
    definition: estimated duration base + 40 * increment >= 480 s)
  * Termination is Normal or Time forfeit (moves before a flag fall are real)
  * no bots
  * both players inside the rating range and within --max-diff of each other
  * clock data present, and at least --min-plies half-moves

This script stops after the first --max-games kept games. Lichess dumps are in
time order, so that sample comes from the start of the month only. It is fine
for quick tests; use make_datasets.py for an unbiased sample.

Usage:
  python filter_games.py lichess_db_standard_rated_2026-09.pgn.zst filtered.pgn \
      --min-rating 1000 --max-rating 2000 --max-games 5000
"""
import argparse
import io
import logging
import re
from collections import Counter

import zstandard as zstd

from common import add_logging_args, setup_logging

log = logging.getLogger(__name__)

TAG_RE = re.compile(r'\[(\w+) "(.*)"\]')
MOVE_NUMBER_RE = re.compile(r"(\d+)\.")
COMMENT_RE = re.compile(r"\{[^}]*\}")  # { [%clk ...] [%eval ...] } comments
WANTED_EVENT_RE = re.compile(r"(?<![A-Za-z])Rated (Rapid|Classical) ")
ALLOWED_TERMINATIONS = ("Normal", "Time forfeit")


def is_wanted_event(text):
    """True for 'Rated Rapid game', 'Rated Classical game', '[Event "Rated Rapid ..."]'."""
    return bool(WANTED_EVENT_RE.search(text))


def parse_headers(game_lines):
    """Turn the [Tag "value"] lines of one game into a dictionary."""
    headers = {}
    for line in game_lines:
        if line.startswith("["):
            tag_match = TAG_RE.match(line)
            if tag_match:
                tag_name, tag_value = tag_match.groups()
                headers[tag_name] = tag_value
    return headers


def reject_reason_with_headers(headers, game_lines, args):
    """None if the game should be kept, otherwise a short reason."""
    if not is_wanted_event(headers.get("Event", "")):
        return "not rated rapid/classical"
    if headers.get("Termination") not in ALLOWED_TERMINATIONS:
        return "termination not allowed"
    if "BOT" in (headers.get("WhiteTitle"), headers.get("BlackTitle")):
        return "bot"

    try:
        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
    except KeyError:
        return "missing rating"
    except ValueError:
        return "rating not an integer"  # e.g. "1500?"
    if not (args.min_rating <= white_elo <= args.max_rating
            and args.min_rating <= black_elo <= args.max_rating):
        return "rating out of range"
    if abs(white_elo - black_elo) > args.max_diff:
        return "rating gap too big"

    movetext = "".join(line for line in game_lines if not line.startswith("["))
    if "[%clk" not in movetext:
        return "no clock data"
    # Remove { ... } comments first: "[%eval 0.17]" would otherwise look like move "0."
    moves_only = COMMENT_RE.sub("", movetext)
    move_numbers = MOVE_NUMBER_RE.findall(moves_only)
    if not move_numbers or int(move_numbers[-1]) * 2 < args.min_plies:
        return "game too short"
    return None


def reject_reason(game_lines, args):
    return reject_reason_with_headers(parse_headers(game_lines), game_lines, args)


def stream_games(path, stats):
    """
    Yield the lines of every rated Rapid/Classical game in a .pgn.zst file.

    Bullet and blitz games (most of the file) are skipped after reading only
    their first line, which makes scanning much faster. stats["scanned"] counts
    every game in the file, including the skipped ones.
    """
    stats.setdefault("scanned", 0)
    with open(path, "rb") as compressed:
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(compressed),
                                encoding="utf-8", errors="replace")
        game_lines = None
        for line in text:
            if line.startswith("[Event "):
                if game_lines:
                    yield game_lines
                stats["scanned"] += 1
                game_lines = [line] if is_wanted_event(line) else None
            elif game_lines is not None:
                game_lines.append(line)
        if game_lines:
            yield game_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="lichess .pgn.zst file")
    parser.add_argument("output", help="filtered .pgn to write")
    parser.add_argument("--min-rating", type=int, default=1000)
    parser.add_argument("--max-rating", type=int, default=2000)
    parser.add_argument("--max-diff", type=int, default=100, help="max rating gap between the players")
    parser.add_argument("--min-plies", type=int, default=20)
    parser.add_argument("--max-games", type=int, default=5000, help="stop after this many kept games")
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)

    stats = {}
    kept = candidates = 0
    reasons = Counter()
    next_report = 100_000

    with open(args.output, "w", encoding="utf-8") as out:
        for game_lines in stream_games(args.input, stats):
            candidates += 1
            reason = reject_reason(game_lines, args)
            if reason is None:
                out.writelines(game_lines)
                out.write("\n")
                kept += 1
                if kept >= args.max_games:
                    break
            else:
                reasons[reason] += 1
            if stats["scanned"] >= next_report:
                log.info("scanned %s games, kept %s", f"{stats['scanned']:,}", f"{kept:,}")
                next_report += 100_000

    log.info("Done. Scanned %s games, %s were rated rapid/classical, kept %s.",
             f"{stats['scanned']:,}", f"{candidates:,}", f"{kept:,}")
    log.info("Rejected (among rapid/classical games) because:")
    for reason, count in reasons.most_common():
        log.info("  %-28s %s", reason, f"{count:,}")


if __name__ == "__main__":
    main()