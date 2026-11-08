"""
Step 1: stream a Lichess .pgn.zst and keep only clean, rapid-or-slower games.

Filters: rated, base time >= --min-base, normal termination, no bots,
both players inside the rating range and within --max-diff of each other,
clock data present, and at least --min-plies half-moves.

It prints WHY games were rejected, so a low "kept" count is easy to explain.

Usage:
  python filter_games.py lichess_db_standard_rated_2026-09.pgn.zst filtered.pgn \
      --min-base 600 --min-rating 1000 --max-rating 2000 --max-games 5000
"""
import argparse
import io
import re
from collections import Counter

import zstandard as zstd

TAG_RE = re.compile(r'\[(\w+) "(.*)"\]')
MOVE_NUM_RE = re.compile(r"(\d+)\.")


def parse_headers(lines):
    headers = {}
    for line in lines:
        if line.startswith("["):
            matched_objects = TAG_RE.match(line)
            if matched_objects:
                headers[matched_objects.group(1)] = matched_objects.group(2)
    return headers


def reject_reason(buffer, args):
    """Return None if the game should be kept, otherwise a short reason string."""
    headers_dictionary = parse_headers(buffer)

    if not headers_dictionary.get("Event", "").startswith("Rated"):
        return "not rated"
    if headers_dictionary.get("Termination") != "Normal":
        return "termination not Normal"
    if "BOT" in (headers_dictionary.get("WhiteTitle"), headers_dictionary.get("BlackTitle")):
        return "bot"

    time_control = headers_dictionary.get("TimeControl", "-")
    if "+" not in time_control:
        return "no time control"
    try:
        base = int(time_control.split("+")[0])
    except ValueError:
        return "bad time control"
    if base < args.min_base:
        return "time control too fast"

    try:
        white_elo, black_elo = int(headers_dictionary["WhiteElo"]), int(headers_dictionary["BlackElo"])
    except (KeyError, ValueError):
        return "missing rating"
    if not (args.min_rating <= white_elo <= args.max_rating and args.min_rating <= black_elo <= args.max_rating):
        return "rating out of range"
    if abs(white_elo - black_elo) > args.max_diff:
        return "rating gap too big"

    movetext = "".join(line for line in buffer if not line.startswith("["))
    if "[%clk" not in movetext:
        return "no clock data"
    nums = MOVE_NUM_RE.findall(movetext)
    if not nums or int(nums[-1]) * 2 < args.min_plies:
        return "game too short"

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--min-base", type=int, default=600, help="min base time in seconds")
    p.add_argument("--min-rating", type=int, default=1000)
    p.add_argument("--max-rating", type=int, default=2000)
    p.add_argument("--max-diff", type=int, default=100, help="max rating gap between players")
    p.add_argument("--min-plies", type=int, default=20)
    p.add_argument("--max-games", type=int, default=5000, help="stop after this many kept games")
    args = p.parse_args()

    kept = scanned = 0
    reasons = Counter()
    buffer = []

    def flush(out):
        nonlocal kept
        if not buffer:
            return
        why = reject_reason(buffer, args)
        if why is None:
            out.writelines(buffer)
            out.write("\n")
            kept += 1
        else:
            reasons[why] += 1

    with open(args.input, "rb") as fh, open(args.output, "w", encoding="utf-8") as out:
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh),
                                encoding="utf-8", errors="replace")
        for line in text:
            if line.startswith("[Event "):
                flush(out)
                buffer = []
                scanned += 1
                if kept >= args.max_games:
                    break
                if scanned % 20_000 == 0:
                    print(f"scanned {scanned:,}  kept {kept:,}")
            buffer.append(line)
        else:
            flush(out)

    print(f"\nDone. Scanned {scanned:,} games, kept {kept:,}.")
    print("Rejected because:")
    for reason, n in reasons.most_common():
        print(f"  {reason:28s} {n:,}")


if __name__ == "__main__":
    main()