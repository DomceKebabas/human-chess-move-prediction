# ChessProject - plan from here

Research question: *Does conditioning on player rating improve human move
prediction, and where do predictions for weak and strong players differ?*

Tick a box (`[x]`) when a step is done. Each phase ends with a **Done when**
line so you know when to move on. Time estimates are rough, for a beginner.

---

## Where we are (2026-11-22)

- [x] Real dataset built: `dataset_all.npz`, 831,144 positions from 55,888 games,
      split 80/10/10 by game.
- [x] Overlap report on real data: 0 duplicate rows, 2.1% / 2.0% of val / test
      positions also in train, about 45% of val/test positions played by
      someone who also plays in train games.
- [x] Decision: **keep the game split**, and report seen vs unseen players.
- [x] Baseline 1 (most common move), val split: **top-1 8.9%, top-3 19.8%**,
      flat across bands; seen vs unseen players 8.8% vs 9.0% (no leakage sign).

---

## Phase 1 - Fix the setup (about 30 min)

- [x] Restore `test_encoding.py` (it is currently a copy of `make_datasets.py`).
      Try PyCharm first: right-click the file -> *Local History* -> *Show History*,
      pick a version from before 22 Nov afternoon. Backup: the 16-test copy in
      `Downloads/human-chess-move-prediction/ChessProject/files/`.
- [x] Recreate the `.venv` (its `activate` points to a deleted folder):
      PyCharm *Settings -> Project -> Python Interpreter -> Add Interpreter ->
      Virtualenv*, then in the Terminal tab:
      `python -m pip install python-chess numpy zstandard pytest`
- [x] Run `python -m pytest`, expect **47 passed** (fewer if only the
      Downloads copy could be restored; write down the real number).
- [x] Decide on the 2700-2899 band (150 positions, 0 in test): **merged into
      one top band "2500+"** for all analysis (`bands.py`; data files unchanged).

**Done when:** `python -m pytest` passes with the plain `python` command and
the band decision is written down.

---

## Phase 2 - Finish the pull request (about 3-5 h)

- [x] Add `.gitignore` with: `.venv/`, `*.npz`, `*.pgn`, `*.pgn.zst`,
      `__pycache__/`, `.pytest_cache/`, `.idea/`, `results_*.csv`
- [x] Find out why the 100-299 band is empty (no games passed the filters?
      check the `make_datasets.py` log).
- [x] Time order: open one `filtered_*.pgn`, check that `UTCDate`/`UTCTime`
      spread over the whole month (proves reservoir sampling worked).
- [x] (decided: limitation only) Provisional ratings: look for the `rating not an integer` count in the
      `make_datasets.py` log. If 0, add it to the limitations.
- [x] (draft in `drafts/pr_description.md`, fill in scan time) Write the PR description with: scan time, games and positions per band,
      split sizes, overlap report, baseline 1 table.
- [ ] Reply to each review thread (resolved / decided / open), using the table
      in `CLAUDE.md` section 5.
- [x] (first version) First README: what the project is, how to install, how to run each
      script, filter definitions, limitations, "results figure coming".

**Done when:** every review thread has a reply and the reviewer can reproduce
the dataset from the README.

---

## Phase 3 - More baselines (about 8-12 h)

All on the **validation** split. Report top-1 and top-3, per band and overall,
plus seen vs unseen players. Save each result table as a CSV.

- [ ] **Baseline 2: Stockfish agreement.** `brew install stockfish`. For about
      2,000 val positions per band, how often is the human move Stockfish's top
      move (and in Stockfish's top 3)? Expect this to rise with rating; it is
      the first result that really shows weak vs strong differences.
- [ ] **Baseline 3: logistic regression** on the flattened board
      (12x8x8 + 4 castling = 772 numbers), with and without the rating as an
      extra input. Start on a subset (e.g. 100k positions) so it runs on the laptop.
- [ ] (optional) Small MLP, same inputs.
- [ ] One summary table: every baseline x every band.

**Done when:** you have a table like "baseline vs band" with at least 3 rows and
can say in one sentence how accuracy changes with rating.

---

## Phase 4 - Main model (about 15-25 h)

- [ ] Upload `dataset_all.npz` (54 MB) to Google Drive; open a Colab notebook
      with a GPU.
- [ ] Data loader: boards + castling -> tensor; legal mask from
      `encoding.legal_mask(fen)` (import from `encoding.py`, never copy it).
- [ ] Small CNN (e.g. 6 conv layers, 64 channels), output `NUM_MOVE_CLASSES`
      (4168), illegal moves masked out before the softmax.
- [ ] Rating as input: extra constant plane(s) with the normalised rating
      (or a band embedding).
- [ ] Tests: output shape is (batch, 4168); a masked illegal move always gets
      probability 0; the model can overfit 100 positions (sanity check).
- [ ] Train with early stopping on validation loss; save the best checkpoint.
- [ ] Report val top-1/top-3 per band, seen vs unseen players.

**Done when:** the CNN clearly beats all baselines on val and the numbers are
saved.

---

## Phase 5 - Experiments / ablations (about 10-15 h)

The core experiment for the research question:

- [ ] **A. With rating vs without rating** (same CNN, rating input replaced
      by a constant). Does rating help, and in which bands most?
- [ ] **B. One model for all ratings vs a separate model per band** (fewer
      data per model, but specialised).
- [ ] Pick 1-2 extra ablations: model size (small vs bigger), amount of
      training data (25% / 50% / 100%), or encoding (no castling input).
- [ ] Cross-band test: a model trained only on 300-699 players vs only on
      2100-2499 players, each evaluated on every band (shows how different
      weak and strong play is).

**Done when:** you can answer "does rating conditioning help?" with numbers
and a figure.

---

## Phase 6 - Error analysis (about 8-12 h)

Limit Stockfish work to a few thousand positions per band.

- [ ] Accuracy by game phase (opening / middlegame / endgame, from `plies`
      or piece count).
- [ ] Accuracy by position complexity (number of legal moves).
- [ ] Accuracy by time pressure (stored `clocks`: e.g. < 1 min vs > 5 min).
- [ ] Blunders: does the model predict human mistakes better for low-rated
      players? (Stockfish evaluation drop of the human move.)
- [ ] Look at 10-20 concrete positions where the model is confidently wrong;
      describe patterns.

**Done when:** you have 3-4 findings, each with a small table or plot.

---

## Phase 7 - Final evaluation and write-up (about 10-15 h)

- [ ] Run the final models **once** on the **test** split (never tune on test).
- [ ] Main figure: top-1 accuracy by rating band, one line per model
      (baselines, CNN without rating, CNN with rating).
- [ ] Report: question, data and filters, method, baselines, results,
      ablations, error analysis, limitations (`CLAUDE.md` section 7 + player
      overlap), conclusion.
- [ ] README with the figure at the top and the final results table.
- [ ] Make the repo public; CV bullet: *Built a model to predict human moves
      conditioned on player rating (PyTorch, Lichess data); achieved X% top-1
      accuracy vs Y% baseline; analysed rating-specific error patterns.*

**Done when:** report submitted and the public repo reproduces the main figure.

---

## Rules to keep in mind

- Run `python -m pytest` after every change.
- Tune and compare on **val**; touch **test** only in Phase 7.
- Encoding lives only in `encoding.py`.
- Never commit data files; never overwrite `dataset_*.npz` without a backup.
