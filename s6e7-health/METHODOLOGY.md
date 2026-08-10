# Methodology and findings - S6E7

An imbalanced 3-class problem (at-risk 86% / unhealthy 8% / fit 6%) scored on **balanced
accuracy** (macro-average of per-class recall). The goal here was less about squeezing the
leaderboard and more about a clean study: run independent modeling pipelines across
different model families, combine them honestly, and measure how much model diversity buys
you. The answer turned out to be "nothing", for an instructive reason.

## The honest-CV contract
Every pipeline shares one contract so the out-of-fold (OOF) predictions are directly
comparable and blendable:
- **Folds:** `shared/folds.csv`, stratified 5-fold, seed 42. Stratified because the classes
  are imbalanced; the rows are independent (no group structure), so a plain stratified split
  is honest here.
- **Metric:** `shared/metric.py`, `balanced_accuracy` plus `tune_decision_weights`. Because
  balanced accuracy weights all classes equally, predicting the majority class everywhere
  scores only 0.333. Minority recall is the whole game, so we model probabilities and tune a
  per-class decision rule (weighted argmax) on OOF to convert them to labels.
- **Deliverable:** each pipeline writes `oof.csv` (train, out-of-fold) and `test_proba.csv`
  (mean over folds) with columns `id,p_at-risk,p_fit,p_unhealthy`. The referee
  (`shared/blend.py`, `shared/stack.py`) combines them.

## Model families (chosen to decorrelate)
A blend only helps if its members disagree, so the members span different inductive biases:

| Pipeline | Family (decision boundary) | OOF balanced accuracy |
|----------|----------------------------|-----------------------|
| `lightgbm` | gradient-boosted trees (axis-aligned) | 0.9497 |
| `catboost` | gradient-boosted trees (axis-aligned) | 0.9492 |
| `neural_net` | entity-embedding MLP (smooth nonlinear) | 0.9478 |
| `lightgbm` (logreg probe) | linear | 0.9162 |
| `lightgbm` (Nystroem probe) | kernel-approx (RBF) | 0.8923 |
| `catboost` (QDA / NB probe) | generative (quadratic) | 0.8219 / 0.8935 |
| `catboost` (kNN probe) | instance-based (local) | 0.8115 |

## The experiment
1. **Two GBDTs are near-identical.** LightGBM vs CatBoost: label agreement 0.994, probability
   correlation 0.998. Blending them lifts balanced accuracy by +0.0001. Different boosting
   implementations on the same features do not decorrelate.
2. **Diversity exists, but does not convert.** The linear, kernel, generative, and instance
   probes genuinely decorrelate from the trees (label agreement 0.63 to 0.90). The neural net
   did not (0.99). Yet the best weighted 7-member blend still scored 0.9497, exactly the best
   single model, and a cross-fitted stacking meta-learner reached the same 0.9497.
3. **Why it is flat.** The decorrelated members disagree with the trees mostly on rows the
   trees already get right (the weak model is wrong there). The rows that actually remain
   wrong (~5%) are shared and largely irreducible: an oracle that could pick the better of the
   two trees per row reaches only 0.9517, and 24.6% of errors are confident misses, the
   signature of label noise. No family recovers unlearnable rows.

## Result
Single best model = weighted blend = stacking meta-learner = **0.9497** OOF. Submitted to
Kaggle: public leaderboard **0.94967**, a CV-to-LB gap of 0.00003. The honest shared-fold
contract predicted the leaderboard almost exactly.

## The takeaway
This is a hard **information ceiling**. The lesson worth keeping: a blend helps only when
its members are **both decorrelated and individually strong**. Decorrelation from weakness is
not the same as complementary signal. Diversity of inductive bias was real and measurable
here, but the residual was irreducible noise, so nothing moved. The companion
`wildlife_translation/` shows the positive counterpart, where distinct sensor modalities are
decorrelated and each strong, and the blend does pay off.

## Reproducibility / performance notes
- This machine has 8 physical / 16 logical cores. In the scikit-learn LightGBM wrapper the
  thread knob is `n_jobs` (not `num_threads`); leaving `n_jobs=-1` oversubscribes the logical
  cores and runs about 2x slower. Cap threads around the physical core count.
- When several pipelines train at once they contend for cores; cap each to roughly a third of
  the cores or stagger heavy fits.
- The blend/stack referees are pure I/O plus NumPy (no training), so they run in seconds.
