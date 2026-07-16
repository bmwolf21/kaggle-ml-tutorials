# Kaggle Workspace

A personal workspace for competing on Kaggle, documenting the full workflow as
reusable tutorials, and translating each competition's techniques into a
wildlife / environmental use case.

Every project here is approached from **three angles**:

1. **Compete** - actually work the dataset and produce leaderboard submissions.
2. **Document** - record every step (`TUTORIAL.md` in each project) so it can be
   turned into a teaching walkthrough later.
3. **Translate** - take the challenges/solutions and apply them to a
   wildlife/environmental problem (`wildlife_translation/` in each project).

## Featured tutorial

**[Spaceship Titanic: An Honest, End-to-End Tutorial](spaceship-titanic/README.md)**
walks through a full tabular Kaggle competition (final public score 0.805, about
top third of 1,931 teams) and then re-applies the entire workflow to a wildlife
species-detection problem in R. Start there. A [runnable notebook](spaceship-titanic/notebooks/spaceship_titanic_tutorial.ipynb)
and the reproducible [scripts](spaceship-titanic/src/) are included.

## Projects

| Project | Type | Status | Wildlife analogue |
|---------|------|--------|-------------------|
| [spaceship-titanic](spaceship-titanic/) | Binary classification | All 3 angles complete (LB 0.80547, ~top 33%) | Species detection with spatial-block CV |

## Environment

- Python via anaconda3 (`/home/bmwolf21/anaconda3/bin/python`) - pandas, numpy,
  scikit-learn, xgboost, lightgbm, matplotlib, jupyter
- Kaggle CLI (`kaggle`) for data download and submissions
- R 4.6.1 for the wildlife-translation re-implementations

## Conventions

- Raw competition data lives in `<project>/data/raw/` and is **git-ignored**.
- Reproducible steps are numbered scripts in `<project>/src/`.
- Exploration happens in `<project>/notebooks/`.
- Submissions are written to `<project>/outputs/submissions/` with a timestamp
  and a one-line description logged in the project `TUTORIAL.md`.
