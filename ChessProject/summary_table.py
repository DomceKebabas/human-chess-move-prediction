"""
One table: every baseline / model x every rating band.

Reads the per-band CSVs written by the baselines and train.py (they all have
the columns group, positions, <variant>top1_pct, <variant>topk_pct) and prints
a Markdown table you can paste into the README, one row per model variant:
  results_most_common_val.csv  ->  rows "most_common (all)" and "most_common (band)"
  results_cnn_rating_val.csv   ->  row  "cnn_rating"

Usage:
    python summary_table.py results_*_val.csv
    python summary_table.py results_*_val.csv --metric topk --csv results_summary_val.csv
"""
import argparse
import csv
import glob
from pathlib import Path


def model_name(path):
    """results_cnn_rating_val.csv -> cnn_rating"""
    name = Path(path).stem
    name = name[len("results_"):] if name.startswith("results_") else name
    for suffix in ("_val", "_test"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def read_rows(path, metric):
    """[(row label, {band: value})] for every variant in one results CSV."""
    with open(path, newline="", encoding="utf-8") as handle:
        lines = list(csv.DictReader(handle))
    suffix = f"{metric}_pct"
    variants = [column[:-len(suffix)] for column in lines[0] if column.endswith(suffix)]
    rows = []
    for variant in variants:
        label = model_name(path) + (f" ({variant.rstrip('_')})" if variant else "")
        rows.append((label, {line["group"]: float(line[variant + suffix]) for line in lines}))
    return rows


def band_order(groups):
    """Bands sorted by rating ('300-499' < '2500+'), with 'all' last."""
    def key(group):
        if group == "all":
            return (1, 0)
        return (0, int(group.split("-")[0].rstrip("+")))
    return sorted(groups, key=key)


def build_table(paths, metric="top1"):
    rows = [row for path in paths for row in read_rows(path, metric)]
    groups = band_order({group for _, values in rows for group in values})
    return groups, rows


def to_markdown(groups, rows):
    lines = ["| Model | " + " | ".join(groups) + " |",
             "|---|" + "---:|" * len(groups)]
    for label, values in rows:
        cells = [f"{values[g]:.1f}" if g in values else "" for g in groups]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="results CSVs (wildcards are fine)")
    parser.add_argument("--metric", choices=["top1", "topk"], default="top1")
    parser.add_argument("--csv", default=None, help="also save the table as CSV")
    args = parser.parse_args()

    # Windows shells do not expand *, so expand wildcards here.
    paths = sorted({p for pattern in args.files for p in (glob.glob(pattern) or [pattern])})
    groups, rows = build_table(paths, args.metric)
    print(f"{args.metric} accuracy (%) by rating band\n")
    print(to_markdown(groups, rows))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model"] + groups)
            for label, values in rows:
                writer.writerow([label] + [f"{values[g]:.2f}" if g in values else "" for g in groups])


if __name__ == "__main__":
    main()
