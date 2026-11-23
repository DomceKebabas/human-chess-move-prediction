"""Small shared helper: logging setup (replaces print)."""
import logging
import sys


def setup_logging(log_file=None, quiet=False):
    """Log to the console (unless quiet) and, optionally, to a file."""
    handlers = []
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)
    # python-chess logs every bad move it meets; we count those games ourselves.
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)


def add_logging_args(parser):
    parser.add_argument("--log-file", default=None, help="also write log messages to this file")
    parser.add_argument("--quiet", action="store_true", help="no console output")