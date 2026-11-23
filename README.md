# Predicting human chess moves by rating

Given a chess position and the rating of the player to move, predict the move a
**human** of that strength would play (in the spirit of the
[Maia Chess](https://maiachess.com) project). Data: the
[Lichess open database](https://database.lichess.org).

**Research question:** *Does conditioning on player rating improve human move
prediction, and where do predictions for weak and strong players differ?*

> Results figure (top-1 accuracy by rating band, per model): *coming soon.*

## Current results (validation split)

| Model | Top-1 | Top-3 |
|---|---:|---:|
| Baseline 1: most common move in this position | 8.9% | 19.8% |
| Baseline 2: Stockfish's top move (depth 12) | *to do* | *to do* |
| Baseline 3: logistic regression (board + rating) | *to do* | *to do* |
| CNN without rating | *to do* | *to do* |
| CNN with rating | *to do* | *to do* |

## Installation

Python 3.13 or newer. The code is in the `ChessProject/` folder; run every
command from there.

**Windows (PowerShell):**

```bash
git clone https://github.com/DomceKebabas/human-chess-move-prediction.git
cd human-chess-move-prediction/ChessProject
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
```

On macOS / Linux use `source .venv/bin/activate` instead of the activate line.
`requirements.txt` installs the CPU build of PyTorch, which is enough for the
tests and for Baseline 3. Train the CNN on a GPU (see *Training on Colab*).

**Stockfish** (Baseline 2 only): download the Windows build from
<https://stockfishchess.org/download/>, unzip it, and pass the `.exe` with
`--engine`, or set the `STOCKFISH_PATH` environment variable once.

## Reproducing the dataset

1. Download a monthly file from <https://database.lichess.org>, e.g.
   `lichess_db_standard_rated_2026-09.pgn.zst` (about 30 GB), into the `ChessProject/` folder.
2. Build the dataset (one pass over the whole file; takes hours):

   ```bash
   python make_datasets.py lichess_db_standard_rated_2026-09.pgn.zst
   ```

   For a quick timing test, add `--max-scan-games 2000000` (the sample then only
   covers the start of the month).
3. Check the split overlap:

   ```bash
   python splits.py dataset_all.npz
   ```

This writes `filtered_<band>.pgn` (sampled games), `dataset_<band>.npz` and the
merged `dataset_all.npz`. Data files are not stored in the repository.

The position filters (`--min-clock`, `--min-ply`, `--max-per-game`) and
`--seed` can be given to `make_datasets.py` and are passed on to
`build_dataset.py`. The defaults reproduce the dataset described below.

## Running the baselines and models

All results are on the **validation** split. Every script prints the same
tables (per rating band + overall, and seen vs unseen players) and saves the
per-band table as `results_<name>_val.csv`.

```bash
python baseline_most_common.py dataset_all.npz --csv results_most_common_val.csv
```

```bash
python baseline_stockfish.py dataset_all.npz --engine C:\tools\stockfish\stockfish.exe --csv results_stockfish_val.csv
```

```bash
python train.py dataset_all.npz --model linear --train-limit 100000
```

```bash
python train.py dataset_all.npz --model linear --train-limit 100000 --no-rating
```

```bash
python summary_table.py results_*_val.csv
```

- **Baseline 2** analyses 2,000 val positions per band (`--per-band`) at
  depth 12 with MultiPV 3. Engine output is cached in
  `stockfish_cache_val_d12.csv`, so a stopped run resumes where it left off.
- **Baseline 3** is softmax (multinomial logistic) regression on the
  flattened board + castling + rating, trained with the same code as the CNN.
  `--no-rating` gives the same model a constant rating. `--model mlp` adds
  one hidden layer.
- The first `train.py` run builds `dataset_all_legal.npz`, the legal moves of
  every position (a few minutes, only once). You can also build it with
  `python legal_cache.py dataset_all.npz`.
- `summary_table.py` prints one Markdown table: every model by every band.

## Training on Colab

1. Upload `dataset_all.npz` to Google Drive and open a new Colab notebook with
   *Runtime -> Change runtime type -> GPU*.
2. Run these cells:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   !git clone https://github.com/DomceKebabas/human-chess-move-prediction.git
   %cd human-chess-move-prediction/ChessProject
   !pip install -q python-chess
   !cp /content/drive/MyDrive/dataset_all.npz .
   !python train.py dataset_all.npz --model cnn
   !python train.py dataset_all.npz --model cnn --no-rating
   !cp -r checkpoints results_*_val.csv dataset_all_legal.npz /content/drive/MyDrive/
   ```

The CNN has 17 input planes (12 piece planes, 4 castling planes, 1 rating
plane), a stem conv, 3 residual blocks (6 conv layers) with 64 channels and a
policy head with 4,168 outputs. Illegal moves get probability 0. Training uses
AdamW and stops early when the validation loss stops improving. The best model
is saved to `checkpoints/<name>.pt`, and the log to `checkpoints/<name>.log`.
To re-evaluate a saved model:

```bash
python evaluate_model.py checkpoints/cnn_rating.pt dataset_all.npz
```

## How the data is built

**Sampling.** Ratings are cut into bands of 200 points. For each band, up to
5,000 qualifying games are drawn uniformly from the whole month (reservoir
sampling), so the sample is not biased to the start of the month.

**Game filters** (`filter_games.py`). A game is kept only if:
- it is `Rated Rapid` or `Rated Classical` (Lichess's own classification);
- it ended normally or on time (`Normal` / `Time forfeit`);
- neither player is a bot;
- both players are in the same band and within 100 rating points of each other;
- clock times (`[%clk]`) are present;
- it has at least 20 plies.

**Position filters** (`build_dataset.py`). From each game, at most 15 random
positions are kept, skipping:
- the first 10 plies (opening book moves);
- moves made with less than 30 s left on the clock;
- positions with only one legal move.

**Encoding** (`encoding.py`). Every position is seen from the side to move (if
Black is to move, the board is mirrored). The board is 12 planes of 8x8 (6 piece
types x 2 colours) plus 4 castling flags. A move is one of 4,168 classes:
from-square x to-square (4,096, queen promotion included) plus 72
under-promotion classes. Illegal moves are masked out.

**Split** (`splits.py`). 80 / 10 / 10 train / validation / test **by game**, so
all positions of one game stay together.

**Bands for analysis** (`bands.py`). 200-point bands; all ratings of 2500 and
above form a single `2500+` band because there are very few such games.

## Dataset (September 2026)

831,144 positions from 55,888 games. Roughly 72-75k positions per band from
400 to 2499, 19.8k for 2500+. The lowest rating in the data is 400, so the
lowest band effectively covers 400-499.

## Project files

| File | Purpose |
|---|---|
| `make_datasets.py` | Main entry point: sample games per band, build and merge datasets |
| `filter_games.py` | Game-level filters |
| `build_dataset.py` | Replay games, apply position filters, write `.npz` |
| `encoding.py` | Board and move encoding (used everywhere) |
| `splits.py` | Train/val/test split and overlap report |
| `bands.py` | Rating bands for analysis |
| `common.py` | Logging setup, dataset loading |
| `evaluation.py` | Shared result tables (per band, seen/unseen players) and CSV output |
| `baseline_most_common.py` | Baseline 1: most common move in this position |
| `baseline_stockfish.py` | Baseline 2: agreement with Stockfish |
| `legal_cache.py` | Legal moves of every position, computed once for training |
| `move_batches.py` | Dataset rows -> PyTorch batches (with legal-move masks) |
| `models.py` | Logistic regression, MLP and CNN |
| `train.py` | Train a model with early stopping, report val results |
| `evaluate_model.py` | Evaluate a saved checkpoint on val or test |
| `summary_table.py` | Merge all results CSVs into one table |
| `requirements.txt` | Python packages |
| `test_*.py` | Unit tests (`python -m pytest`) |

## Limitations

- **Player overlap:** the split is by game, so about 45% of validation/test
  positions are played by someone who also appears in training games. The model
  receives no player identity, and accuracy is reported separately for seen and
  unseen players.
- **Provisional ratings:** the Lichess dump stores ratings as plain numbers
  (no `?` marker), so players whose rating has not settled yet cannot be
  identified and may be included. About 2.3% of player-games show a rating change
  of more than 30 points, a rough upper bound on how often this matters.
- **Cheating accounts** cannot be removed reliably.
- **Clock filter** uses the time left *after* the move, not before (small approximation).
- **Equal games per band** means rare ratings are over-represented compared with
  real Lichess.
- **Under-promotions** have their own classes but very few training examples.
- **Repeated positions:** about 2% of validation/test positions also occur in
  training (mostly common openings).

## Acknowledgements

Data from the [Lichess open database](https://database.lichess.org) (CC0).
Idea inspired by [Maia Chess](https://maiachess.com).
