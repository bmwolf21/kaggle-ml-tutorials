# Collaboration note: Claude + Codex on ROGII Wellbore Geology Prediction

Hi Codex. Two coding agents (you and Claude Code) are collaborating on this Kaggle
competition. The human is the orchestrator: they will occasionally ask each of us
to sync with the other's latest work. We work **independently in our own
subdirectories**, review each other at checkpoints, and blend at the end. This
note is the shared source of truth; the machine-readable contract is in `shared/`.

## The goal

Build the best possible model for **ROGII - Wellbore Geology Prediction**
(<https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction>, Featured,
$50k, deadline 2026-08-05). Two independent pipelines that we then **blend** should
beat either alone, and mutual code review should catch the leakage traps this
competition is famous for.

## The task (what Claude has scoped so far)

- **Target:** `TVT` (True Vertical Thickness), a continuous value = the well's
  position in the formation column. **Regression.**
- **Per horizontal well, available at INFERENCE (test):** only
  `MD, X, Y, Z` (trajectory), `GR` (gamma ray), and `TVT_input` (the known TVT for
  the heel, ~first 1,442 rows). We predict `TVT` for the masked **toe** rows.
- **Train horizontal wells additionally include** the answer `TVT` plus marker
  columns (`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`) which are **train-only**.
- **Type well** (`<id>__typewell.csv`): `TVT, GR, Geology` = the reference GR
  signature with labeled formations. Core idea: correlate the lateral GR against
  this to locate stratigraphic position.
- **Submission:** `id = <wellID>_<rowIndex>`, column `tvt`. 14,151 toe rows.
- **PNGs are train-only label renderings, NOT model inputs.** No GPU needed.
- See `shared/DATA_SPEC.md` for the authoritative column/format spec.

## The crux: leakage and cross-validation

Well IDs appear to overlap between `train/` and `test/`; the community is actively
discussing leakage risk. **The single most important thing is honest, group-aware
validation.** We therefore share ONE canonical fold assignment so our scores are
comparable and our out-of-fold predictions are blendable:

- **`shared/folds.csv`** maps `well_id -> fold` (5 folds, whole wells per fold,
  stratified). **Use it. Do not invent your own folds.** If you believe the fold
  scheme is wrong, do not silently change it: propose a change in the handoff log
  and let the human arbitrate, so we stay aligned.
- Score with `shared/metric.py` so we report the same number.

## Working protocol (subdirectory split)

```
data/raw/              shared raw data (git-ignored; re-downloadable via kaggle CLI)
shared/                the CONTRACT (read-only-ish): folds.csv, metric.py,
                       make_folds.py, DATA_SPEC.md. Change only via the handoff log.
claude/                Claude's pipeline (Claude works only here + shared writes)
codex/                 Codex's pipeline (Codex works only here)
outputs/submissions/   both write here, agent-prefixed: claude_*.csv / codex_*.csv
COLLABORATION.md       this file, incl. the HANDOFF LOG at the bottom
```

Rules of the road:
1. **Stay in your own subdirectory.** Claude edits `claude/`; Codex edits `codex/`.
   Neither edits the other's folder. This avoids clobbering.
2. **`shared/` is a contract.** Treat it as read-only. If it must change (e.g. we
   confirm the metric is MAE not RMSE), announce it in the handoff log first.
3. **Save OOF + test predictions** for your best pipeline in your own folder as
   `oof.csv` (well_id,row_index,tvt_pred) and `test_pred.csv`, on the shared folds,
   so the other can blend with yours.
4. **Submissions** go to `outputs/submissions/` named `claude_<desc>_<date>.csv` or
   `codex_<desc>_<date>.csv`. Log every submission's CV and (when known) LB below.
5. **Communicate through git + this log.** We cannot talk in real time. Commit
   often with clear messages. When the human asks you to sync: `git pull`, read the
   other's recent commits and their handoff-log entries, incorporate what is worth
   it (cite it in your log), leave the rest.
6. **Original work only** (prize competition). Use public discussion/domain
   material for understanding, not code lifting.

## End state

Two independent pipelines, each validated on the shared folds, then a blend
(weights tuned on the shared out-of-fold predictions). Whoever runs the final
blend documents it here.

---

## Handoff log (append newest at the bottom; keep entries short)

### [Claude] setup (done)
- Scoped the task, created the subdirectory layout and the `shared/` contract.
- **Metric CONFIRMED from the task deck: plain RMSE on `tvt` in feet**
  (`dTVT = manualTVT - predictedTVT`, RMSE of all dTVT). See `shared/metric.py`.
- **`shared/folds.csv` BUILT**: 773 train wells, 5 whole-well group folds
  (155/155/155/154/154), stratified by median TVT + azimuth sign + spatial X bin.
  Use it. Regenerate identically with `python shared/make_folds.py` if needed.
- **Leakage finding (matters to both of us):** there are only **3 test wells**,
  and their IDs are also train wells, so their toe answers are in `train/`. The
  public LB is therefore near-meaningless. **Optimize the shared group-CV, not the
  public LB.** The scored set is a hidden private set. Details in
  `shared/DATA_SPEC.md`.
- **Domain hint from the deck:** the lateral's own GR *before* PS correlates with
  its GR *after* PS better than the type-well GR does; offset/neighboring wells
  share dip. Worth using.
- Next (Claude): cross-correlation baseline in `claude/` (align lateral GR to the
  type-well GR-vs-TVT profile, anchor on heel `TVT_input`), report CV on the shared
  folds, save `claude/oof.csv` + `claude/test_pred.csv` for blending.
- **Codex:** build an independent pipeline in `codex/` against `shared/folds.csv`
  and `shared/metric.py`. Save your `codex/oof.csv` (well_id,row_index,tvt_pred) on
  the shared folds so we can blend. Log anything you learn about leakage here first.
