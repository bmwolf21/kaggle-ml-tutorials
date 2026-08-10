# 03_layer_model.R
# ---------------------------------------------------------------------------
# The payoff of the ROGII translation: predict the animal's layer position along
# the unknown part of its track, and show the SAME lessons as the wellbore task.
#
#   1. Geometry (raw depth) is useless - the animal holds its layer while depth
#      swings with the dipping thermocline (mirrors dTVT vs dZ in the well).
#   2. The sensor + the animal's OWN reference profile is the signal.
#   3. THE CRUX: honest validation must leave WHOLE ANIMALS out (group CV). A
#      raw-sensor model looks great under random-row CV but collapses under
#      group CV, because random folds leak each animal's tag bias / held layer.
#      This is the exact analogue of group-by-WELL CV in ROGII.
#   4. Neighbouring animals (shared thermocline) give a useful prior.
#
# Run:  Rscript 03_layer_model.R  (sources 02_features.R)
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ranger) })

here <- local({
  a <- commandArgs(FALSE); f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
source(file.path(here, "02_features.R"))
tracks <- read.csv(file.path(here, "data", "tracks.csv"), stringsAsFactors = FALSE)
reference <- read.csv(file.path(here, "data", "reference_profiles.csv"), stringsAsFactors = FALSE)
F <- engineer(tracks, reference)
rmse <- function(a, b) sqrt(mean((a - b)^2))

# fold assignments: whole-animal (group) vs random-row
animals <- unique(F$animal_id)
set.seed(42); afold <- setNames(sample(rep(1:5, length.out = length(animals))), animals)
F$gfold <- afold[F$animal_id]
set.seed(42); F$rfold <- sample(rep(1:5, length.out = nrow(F)))

cat(sprintf("toe points: %d across %d animals\n", nrow(F), length(animals)))
cat(sprintf("\nbaselines (RMSE of layer position, m):\n"))
cat(sprintf("  flat (hold last known layer): %.3f\n", rmse(F$target, 0)))
cat(sprintf("  geometry (delta = d_depth):   %.3f\n", rmse(F$target, F$d_depth)))

cv <- function(feats, foldcol) {
  oof <- numeric(nrow(F))
  for (k in 1:5) {
    tr <- F[[foldcol]] != k; va <- F[[foldcol]] == k
    m <- ranger(x = F[tr, feats], y = F$target[tr], num.trees = 300, seed = 42)
    oof[va] <- predict(m, F[va, feats])$predictions
  }
  rmse(F$target, oof)
}

A <- c("temp", "d_depth", "depth", "along_dist", "toe_frac")           # raw sensor + geometry
B <- c("ref_impl", "self_impl", "ref_impl_smooth", "d_depth", "toe_frac")  # own-reference calibrated

a_rand <- cv(A, "rfold"); a_grp <- cv(A, "gfold")
b_grp <- cv(B, "gfold")
cat("\nTHE CRUX - random-row CV vs group-by-animal CV (RMSE, m):\n")
cat(sprintf("  Model A (raw sensor+geometry): random-row %.3f  |  group-by-animal %.3f\n", a_rand, a_grp))
cat(sprintf("    -> random-row CV is optimistic by %.3f m; it leaked each animal's bias.\n", a_grp - a_rand))
cat(sprintf("  Model B (own-reference calibrated): group-by-animal %.3f  (robust)\n", b_grp))

# neighbour prior: nearest animals share the thermocline; use their delta-profile
loc <- tracks %>% group_by(animal_id) %>% summarise(x = mean(x), y = mean(y), .groups = "drop")
prof_grid <- seq(0, 1, length.out = 25)
prof <- lapply(split(F, F$animal_id), function(d) approx(d$toe_frac, d$target, prof_grid, rule = 2)$y)
F$nbr <- 0
for (k in 1:5) {
  tgt <- animals[afold[animals] == k]; poolids <- animals[afold[animals] != k]
  pl <- loc[match(poolids, loc$animal_id), ]
  for (aid in tgt) {
    q <- loc[loc$animal_id == aid, ]
    d <- sqrt((pl$x - q$x)^2 + (pl$y - q$y)^2)
    nb <- poolids[order(d)][1:8]
    pr <- rowMeans(sapply(nb, function(n) prof[[n]]))
    m <- F$animal_id == aid
    F$nbr[m] <- approx(prof_grid, pr, F$toe_frac[m], rule = 2)$y
  }
}
b_nbr <- cv(c(B, "nbr"), "gfold")
cat(sprintf("  Model B + neighbour prior: group-by-animal %.3f\n", b_nbr))

imp <- ranger(x = F[, c(B, "nbr")], y = F$target, num.trees = 400,
              importance = "permutation", seed = 42)$variable.importance
cat("\nfeature importance:\n"); print(round(sort(imp, decreasing = TRUE), 4))

# figure: the random-vs-group gap
fig <- file.path(here, "outputs", "figures"); dir.create(fig, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages(library(ggplot2))
pd <- data.frame(
  model = c("A: raw sensor", "A: raw sensor", "B: own-reference"),
  cv = c("random-row", "group-by-animal", "group-by-animal"),
  rmse = c(a_rand, a_grp, b_grp))
p <- ggplot(pd, aes(model, rmse, fill = cv)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("random-row" = "#C44E52", "group-by-animal" = "#55A868")) +
  labs(title = "Leave-whole-animals-out is the honest test (mirrors group-by-well CV)",
       y = "RMSE of layer position (m)", x = NULL) + theme_minimal()
ggsave(file.path(fig, "group_cv_crux.png"), p, width = 7.5, height = 4.5, dpi = 120)
cat(sprintf("\nSaved %s\n", file.path(fig, "group_cv_crux.png")))
