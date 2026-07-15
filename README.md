# Kaggle Workspace

A personal workspace for competing on Kaggle, documenting the full workflow as
reusable tutorials, and translating each competition's techniques into a
wildlife / environmental use case.

Every project here is approached from **three angles**:

1. **Compete** — actually work the dataset and produce leaderboard submissions.
2. **Document** — record every step (`TUTORIAL.md` in each project) so it can be
   turned into a teaching walkthrough later.
3. **Translate** — take the challenges/solutions and apply them to a
   wildlife/environmental problem (`wildlife_translation/` in each project).

## Projects

| Project | Type | Status | Wildlife analogue |
|---------|------|--------|-------------------|
| [spaceship-titanic](spaceship-titanic/) | Binary classification | In progress | Species detection / occupancy from noisy survey covariates |

## Environment

- Python via anaconda3 (`/home/bmwolf21/anaconda3/bin/python`) — pandas, numpy,
  scikit-learn, xgboost, lightgbm, matplotlib, jupyter
- Kaggle CLI (`kaggle`) for data download and submissions
- R 4.6.1 for the wildlife-translation re-implementations

## Conventions

- Raw competition data lives in `<project>/data/raw/` and is **git-ignored**.
- Reproducible steps are numbered scripts in `<project>/src/`.
- Exploration happens in `<project>/notebooks/`.
- Submissions are written to `<project>/outputs/submissions/` with a timestamp
  and a one-line description logged in the project `TUTORIAL.md`.
