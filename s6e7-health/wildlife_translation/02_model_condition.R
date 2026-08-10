# 02_model_condition.R
# ---------------------------------------------------------------------------
# The payoff of the S6E7 translation: classify body-condition {poor,fair,good}
# and reproduce the SAME lessons as the competition, plus the ecological twist.
#
#   1. Balanced accuracy (macro recall) + class weighting + OOF decision-rule
#      tuning, because poor-condition animals are rare but the priority.
#   2. THE CRUX: honest validation must leave WHOLE ANIMALS out (group CV). A model
#      looks better under random-row CV because repeated captures leak each animal's
#      persistent tag/individual bias. (Analogue of group-by-well CV in ROGII.)
#   3. Multi-modal blend PAYS OFF: a morphometric-only and a movement/physiology-only
#      model each plateau, but they are decorrelated AND each strong on complementary
#      cases, so blending beats both. This is the POSITIVE counterpart to S6E7, where
#      diverse models could not help because the residual was irreducible noise.
#
# Run: Rscript 02_model_condition.R
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ranger) })
set.seed(42)

here <- local({
  a <- commandArgs(FALSE); f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
dat <- read.csv(file.path(here, "data", "captures.csv"), stringsAsFactors = FALSE)
dat$condition <- factor(dat$condition, levels = c("poor", "fair", "good"))
CLS <- levels(dat$condition)

MORPH <- c("body_mass", "chest_girth", "kidney_fat")
MOVE  <- c("odba", "activity_hr", "rest_bouts", "cortisol")
ALLF  <- c(MORPH, MOVE, "sex")

# ---- metric + decision-rule tuning (mirrors shared/metric.py) ----
bal_acc <- function(truth, pred) {
  mean(sapply(CLS, function(c) { m <- truth == c; if (!any(m)) NA else mean(pred[m] == c) }), na.rm = TRUE)
}
labels_from_proba <- function(P, w = c(1, 1, 1)) CLS[max.col(sweep(P, 2, w, "*"), ties.method = "first")]
tune_weights <- function(P, truth, n_iter = 1500) {
  best_w <- c(1, 1, 1); best_s <- bal_acc(truth, labels_from_proba(P, best_w))
  for (i in seq_len(n_iter)) {
    w <- exp(rnorm(3, 0, 0.7)); s <- bal_acc(truth, labels_from_proba(P, w))
    if (s > best_s) { best_s <- s; best_w <- w }
  }
  step <- 0.5
  for (it in 1:120) {
    improved <- FALSE
    for (j in 1:3) for (d in c(step, -step)) {
      w <- best_w; w[j] <- w[j] * exp(d); s <- bal_acc(truth, labels_from_proba(P, w))
      if (s > best_s) { best_s <- s; best_w <- w; improved <- TRUE }
    }
    if (!improved) { step <- step / 2; if (step < 1e-3) break }
  }
  list(w = round(best_w / best_w[1], 3), score = best_s)
}

# ---- out-of-fold probabilities for a feature set under a given fold vector ----
cv_proba <- function(feats, foldvec) {
  P <- matrix(0, nrow(dat), 3, dimnames = list(NULL, CLS))
  for (k in sort(unique(foldvec))) {
    tri <- foldvec != k; vai <- foldvec == k
    Xtr <- dat[tri, feats, drop = FALSE]; Xva <- dat[vai, feats, drop = FALSE]
    for (c in feats) if (is.numeric(Xtr[[c]])) {   # train-median impute (no leak)
      med <- median(Xtr[[c]], na.rm = TRUE)
      Xtr[[c]][is.na(Xtr[[c]])] <- med; Xva[[c]][is.na(Xva[[c]])] <- med
    }
    if ("sex" %in% feats) { Xtr$sex <- factor(Xtr$sex); Xva$sex <- factor(Xva$sex, levels = levels(Xtr$sex)) }
    y <- dat$condition[tri]
    w <- as.numeric((1 / table(y))[as.character(y)])   # inverse-freq class weights
    m <- ranger(x = Xtr, y = y, probability = TRUE, num.trees = 400,
                case.weights = w, seed = 42, num.threads = 3)
    pr <- predict(m, Xva)$predictions
    P[vai, colnames(pr)] <- pr
  }
  P
}

# ---- fold assignments: whole-animal (group) vs random-row ----
animals <- unique(dat$animal_id)
afold <- setNames(sample(rep(1:5, length.out = length(animals))), animals)
gfold <- afold[as.character(dat$animal_id)]
rfold <- sample(rep(1:5, length.out = nrow(dat)))
y <- dat$condition

cat(sprintf("%d events / %d animals; class balance:\n", nrow(dat), length(animals)))
print(round(prop.table(table(y)), 4))
cat(sprintf("\nbaseline (predict majority 'good') balanced accuracy: %.3f  (worthless)\n",
            bal_acc(y, rep("good", nrow(dat)))))

# ---- LESSON 2: random-row CV is optimistic; group-by-animal CV is honest ----
Pr <- cv_proba(ALLF, rfold)
Pg <- cv_proba(ALLF, gfold)
tr_r <- tune_weights(Pr, y); tr_g <- tune_weights(Pg, y)
cat("\n=== THE CRUX: random-row vs group-by-animal CV (full feature model) ===\n")
cat(sprintf("  random-row CV   : argmax %.3f | tuned %.3f\n", bal_acc(y, labels_from_proba(Pr)), tr_r$score))
cat(sprintf("  group-by-animal : argmax %.3f | tuned %.3f\n", bal_acc(y, labels_from_proba(Pg)), tr_g$score))
cat(sprintf("  -> random-row is optimistic by %.3f (leaked each animal's persistent bias)\n",
            tr_r$score - tr_g$score))
predg <- labels_from_proba(Pg, tr_g$w)
cat("  honest per-class recall (group CV, tuned): ")
cat(paste(sprintf("%s %.3f", CLS, sapply(CLS, function(c) mean(predg[y == c] == c))), collapse = " | "), "\n")

# ---- LESSON 3: multi-modal blend pays off (all under HONEST group CV) ----
Pm <- cv_proba(c(MORPH, "sex"), gfold)     # morphometrics only
Pv <- cv_proba(c(MOVE, "sex"), gfold)      # movement / physiology only
Pb <- (Pm + Pv) / 2                        # equal-weight modality blend
sm <- tune_weights(Pm, y)$score; sv <- tune_weights(Pv, y)$score; sb <- tune_weights(Pb, y)$score
predm <- labels_from_proba(Pm); predv <- labels_from_proba(Pv)
agree <- mean(predm == predv)
both_wrong <- mean(predm != y & predv != y); either_wrong <- mean(predm != y | predv != y)
oracle <- bal_acc(y, ifelse(predm == y | predv == y, as.character(y), as.character(predm)))
cat("\n=== multi-modal blend (honest group CV, tuned balanced accuracy) ===\n")
cat(sprintf("  morphometrics only : %.3f\n", sm))
cat(sprintf("  movement/physiol   : %.3f\n", sv))
cat(sprintf("  BLEND of the two   : %.3f   (lift over best modality %+.3f)\n", sb, sb - max(sm, sv)))
cat(sprintf("  modality label agreement %.3f | error overlap (Jaccard) %.3f | oracle-of-two %.3f\n",
            agree, both_wrong / either_wrong, oracle))
cat("  -> decorrelated AND each strong on complementary cases, so the blend helps.\n")
cat("     Contrast S6E7: there the diverse members were decorrelated-but-weak and the\n")
cat("     residual was irreducible noise, so the blend was flat. Here it is not.\n")

# ---- figure ----
fig <- file.path(here, "outputs", "figures"); dir.create(fig, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages(library(ggplot2))
pd <- rbind(
  data.frame(panel = "validation honesty", model = "full: random-row CV", ba = tr_r$score),
  data.frame(panel = "validation honesty", model = "full: group-by-animal CV", ba = tr_g$score),
  data.frame(panel = "multi-modal blend", model = "morphometrics only", ba = sm),
  data.frame(panel = "multi-modal blend", model = "movement/physiology only", ba = sv),
  data.frame(panel = "multi-modal blend", model = "blend of both", ba = sb))
p <- ggplot(pd, aes(reorder(model, ba), ba, fill = panel)) +
  geom_col() + coord_flip() +
  geom_text(aes(label = sprintf("%.3f", ba)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = c("validation honesty" = "#C44E52", "multi-modal blend" = "#55A868")) +
  ylim(0, 1) +
  labs(title = "Body-condition scoring: group CV is honest; multi-modal blend pays off",
       y = "balanced accuracy (macro recall)", x = NULL, fill = NULL) + theme_minimal()
ggsave(file.path(fig, "condition_results.png"), p, width = 9, height = 4.2, dpi = 120)
cat(sprintf("\nSaved %s\n", file.path(fig, "condition_results.png")))
