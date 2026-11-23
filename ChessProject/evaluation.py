"""
Shared result reporting for every baseline and model.

Every evaluator produces one dict per predicted position ("result row"):
    {"band": 1100, "seen_player": True, "top1": False, "topk": True, ...}
  * "band" and "seen_player" are used for GROUPING,
  * every other True/False column is a METRIC and is turned into a percentage
    (e.g. "top1" -> "top1_pct"). A model can report several variants side by
    side by using several columns (baseline 1 uses "all_top1", "band_top1", ...).

report() then prints the same three tables for every model
  1. per rating band + overall,
  2. seen vs unseen players (the mover also plays in train games or not),
and optionally saves table 1 to a CSV, so all results files have the same shape
and summary_table.py can merge them.
"""
import csv
import logging
from collections import defaultdict

import numpy as np

from bands import BAND_WIDTH, MIN_RATING, TOP_BAND_START, band_label, band_of
from splits import TRAIN

log = logging.getLogger(__name__)

GROUP_KEYS = ("band", "seen_player")


def bands_of(ratings, min_rating=MIN_RATING, band_width=BAND_WIDTH, top_band_start=TOP_BAND_START):
    """Band (lower edge) of every rating, as a list of ints."""
    return [band_of(int(r), min_rating, band_width, top_band_start) for r in np.asarray(ratings).tolist()]


def train_players(data):
    """Everyone who plays in at least one TRAIN game (as mover or as opponent)."""
    train_rows = data["split"] == TRAIN
    return set(data["players"][train_rows].tolist()) | set(data["opponents"][train_rows].tolist())


def sample_rows_per_band(rows, bands, per_band, seed=0):
    """
    Up to `per_band` rows from every band, chosen at random with a fixed seed.
    rows: indices into the dataset. bands: band of EVERY dataset row.
    Returns the chosen indices, sorted.
    """
    rng = np.random.default_rng(seed)
    by_band = defaultdict(list)
    for row in rows:
        by_band[bands[row]].append(row)
    chosen = []
    for band in sorted(by_band):
        candidates = np.array(by_band[band])
        if len(candidates) > per_band:
            candidates = rng.choice(candidates, size=per_band, replace=False)
        chosen.extend(candidates.tolist())
    return sorted(chosen)


def metric_columns(results):
    """The True/False metric columns, in the order the evaluator wrote them."""
    return [key for key in results[0] if key not in GROUP_KEYS]


def summarize(results, group_of):
    """Group the rows and turn the True/False columns into percentages."""
    metrics = metric_columns(results)
    groups = defaultdict(list)
    for row in results:
        groups[group_of(row)].append(row)
    table = []
    for name in sorted(groups, key=str):
        rows = groups[name]
        n = len(rows)
        entry = {"group": name, "positions": n}
        for metric in metrics:
            entry[f"{metric}_pct"] = 100.0 * sum(bool(r[metric]) for r in rows) / n
        table.append(entry)
    return table


def by_band_table(results, band_width=BAND_WIDTH, top_band_start=TOP_BAND_START):
    """Per-band rows (sorted by rating, readable labels) followed by an 'all' row."""
    table = summarize(results, lambda r: r["band"])
    table.sort(key=lambda row: row["group"])
    for row in table:
        row["group"] = band_label(row["group"], band_width, top_band_start)
    return table + summarize(results, lambda r: "all")


def by_player_table(results):
    return summarize(results, lambda r: "seen player" if r["seen_player"] else "unseen player")


def log_table(title, table):
    """Print a table: group, number of positions, then every *_pct column."""
    columns = [key for key in table[0] if key.endswith("_pct")]
    headers = [key[:-len("_pct")] for key in columns]
    width = max(8, *(len(h) for h in headers))
    log.info("")
    log.info(title)
    log.info("  %-14s %9s " + " ".join(f"%{width}s" for _ in headers), "group", "positions", *headers)
    for row in table:
        values = " ".join(f"{row[c]:{width - 1}.1f}%" for c in columns)
        log.info("  %-14s %9d %s", row["group"], row["positions"], values)


def write_csv(path, table):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    log.info("Saved per-band table to %s", path)


def report(results, title, csv_path=None, band_width=BAND_WIDTH, top_band_start=TOP_BAND_START):
    """Print the per-band and seen/unseen tables; save the per-band one if csv_path is given."""
    if not results:
        raise ValueError("no result rows to report")
    band_table = by_band_table(results, band_width, top_band_start)
    log_table(f"{title} ({len(results)} positions)", band_table)
    log_table("By player (seen = the mover also plays in train games)", by_player_table(results))
    if csv_path:
        write_csv(csv_path, band_table)
    return band_table


def topk_from_scores(scores, legal, k):
    """
    Best k LEGAL move classes per row, best first.
    scores: (n, NUM_MOVE_CLASSES) floats (logits or probabilities).
    legal:  (n, NUM_MOVE_CLASSES) bool mask.
    Ties are broken by the smaller class index (stable sort), so results are repeatable.
    """
    masked = np.where(legal, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    return order


def hit_rows(bands, seen, targets, topk):
    """Result rows from top-k predictions: top1 = first guess right, topk = any guess right."""
    targets = np.asarray(targets)
    topk = np.asarray(topk)
    top1 = topk[:, 0] == targets
    anyk = (topk == targets[:, None]).any(axis=1)
    return [{"band": b, "seen_player": bool(s), "top1": bool(t1), "topk": bool(tk)}
            for b, s, t1, tk in zip(bands, seen, top1.tolist(), anyk.tolist())]
