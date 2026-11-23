"""
Legal-move cache: the legal move classes of every dataset row, computed once.

Why: a model must only give probability to LEGAL moves, so every training
batch needs a legal-move mask. A full mask for all 831k rows would be
831k x 4168 bytes (about 3.5 GB), and re-parsing every FEN in every epoch is
slow. Instead we store the legal class indices of all rows one after another
in a flat array plus offsets (about 30 million int16 numbers, ~60 MB):

    legal moves of row i  =  flat[offsets[i] : offsets[i + 1]]

and expand only the rows of the current batch into a mask.

The legal moves come from encoding.legal_move_indices, the same function the
dataset builder uses, so the cache can never disagree with the encoding.

Usage (writes dataset_all_legal.npz next to the dataset):
    python legal_cache.py dataset_all.npz
"""
import argparse
import hashlib
import logging
from pathlib import Path

import chess
import numpy as np

from common import add_logging_args, load_dataset, setup_logging
from encoding import NUM_MOVE_CLASSES, legal_move_indices

log = logging.getLogger(__name__)


def fingerprint(fens):
    """Short hash of all FENs, so a cache is never used with the wrong dataset."""
    digest = hashlib.sha1()
    for fen in np.asarray(fens).tolist():
        digest.update(fen.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build(fens):
    """(flat int16 array of legal classes, int64 offsets of length len(fens) + 1)."""
    fens = np.asarray(fens).tolist()
    parts = []
    offsets = np.zeros(len(fens) + 1, dtype=np.int64)
    for i, fen in enumerate(fens):
        legal = sorted(legal_move_indices(chess.Board(fen)))
        parts.append(np.array(legal, dtype=np.int16))
        offsets[i + 1] = offsets[i] + len(legal)
        if (i + 1) % 100_000 == 0:
            log.info("legal moves computed for %s / %s rows", f"{i + 1:,}", f"{len(fens):,}")
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
    return flat, offsets


def default_path(dataset_path):
    path = Path(dataset_path)
    return path.with_name(f"{path.stem}_legal.npz")


def save(path, flat, offsets, fens):
    np.savez_compressed(path, flat=flat, offsets=offsets, fingerprint=np.array(fingerprint(fens)))


def load(path, fens):
    """Load a cache and check it belongs to this dataset (same FENs, same order)."""
    with np.load(path) as cache:
        flat, offsets, stored = cache["flat"], cache["offsets"], str(cache["fingerprint"])
    if len(offsets) != len(fens) + 1 or stored != fingerprint(fens):
        raise SystemExit(f"{path} was built for a different dataset. Delete it and rebuild: "
                         f"python legal_cache.py <dataset>.npz")
    return flat, offsets


def load_or_build(dataset_path, fens, cache_path=None):
    """Use the cache file if it exists, otherwise build and save it."""
    cache_path = Path(cache_path) if cache_path else default_path(dataset_path)
    if cache_path.exists():
        log.info("Using legal-move cache %s", cache_path)
        return load(cache_path, fens)
    log.info("Building legal-move cache %s (only needed once, takes a few minutes)", cache_path)
    flat, offsets = build(fens)
    save(cache_path, flat, offsets, fens)
    return flat, offsets


def masks(flat, offsets, rows):
    """(len(rows), NUM_MOVE_CLASSES) bool mask of the legal moves of the given rows."""
    rows = np.asarray(rows)
    starts, ends = offsets[rows], offsets[rows + 1]
    lengths = ends - starts
    mask = np.zeros((len(rows), NUM_MOVE_CLASSES), dtype=bool)
    batch_index = np.repeat(np.arange(len(rows)), lengths)
    # Position of every legal move in `flat`: start of its row + 0, 1, 2, ...
    within_row = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    mask[batch_index, flat[np.repeat(starts, lengths) + within_row]] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="merged .npz, e.g. dataset_all.npz")
    parser.add_argument("--output", default=None, help="default: <dataset>_legal.npz")
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)

    fens = load_dataset(args.dataset)["fens"]
    output = args.output or default_path(args.dataset)
    flat, offsets = build(fens)
    save(output, flat, offsets, fens)
    log.info("Saved %s: %s rows, %s legal moves (%.1f per position)",
             output, f"{len(fens):,}", f"{len(flat):,}", len(flat) / max(1, len(fens)))


if __name__ == "__main__":
    main()
