# 00_simulate_condition.R
# ---------------------------------------------------------------------------
# Wildlife translation of Kaggle S6E7 (imbalanced 3-class health scoring,
# balanced-accuracy metric). Here: classify an individual animal's BODY-CONDITION
# class {poor, fair, good} from bio-logger + capture data. Poor-condition animals
# are rare but the management priority, so balanced accuracy (macro recall) is the
# right metric, exactly as in the competition.
#
# The simulation is built to teach the transferable lessons faithfully:
#   1. Imbalanced classes -> majority shortcut is worthless under balanced accuracy.
#   2. GROUP structure: each animal is captured several times, with an individual
#      random effect (tag/animal bias). Random-row CV leaks that bias; only
#      leave-whole-animal-out CV is honest. (The Kaggle data had no groups; real
#      condition data always does. This is the enrichment.)
#   3. TWO sensor modalities carry COMPLEMENTARY condition signal:
#        - morphometrics (mass, girth): reveal the obviously emaciated,
#        - movement/physiology (ODBA, activity, cortisol): reveal the "hidden"
#          poor-condition animals that still look fine on mass.
#      So a morphometric-only and a movement-only model each plateau, but they are
#      decorrelated AND each strong on complementary cases -> the blend genuinely
#      helps. This is the POSITIVE counterpart to S6E7's information ceiling, where
#      diverse models could not help because the residual was irreducible noise.
#
# Everything is simulated -> reproducible, no field data. Run: Rscript 00_simulate_condition.R
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr) })
set.seed(2026)

here <- local({
  a <- commandArgs(FALSE); f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
dir.create(file.path(here, "data"), showWarnings = FALSE)

N       <- 900                         # individuals
priors  <- c(good = 0.62, fair = 0.30, poor = 0.08)   # imbalanced, poor is rare
mu_z    <- c(good = 1.2, fair = 0.0, poor = -1.6)      # latent condition by class
sd_z    <- 0.55                        # class overlap -> some irreducible Bayes error

cls <- sample(names(priors), N, replace = TRUE, prob = priors)
z   <- mu_z[cls] + rnorm(N, 0, sd_z)   # latent condition index per individual

# Per-individual random effects (the thing that LEAKS under random CV): each animal
# has a persistent morphometric bias and a persistent movement bias. Made sizeable
# (and animals are captured several times below) so random-row folds, which scatter an
# animal's captures across train and val, can memorise that bias.
b_morph <- rnorm(N, 0, 1.3)
b_move  <- rnorm(N, 0, 1.3)

# Complementary-modality trick: for a subset of animals one modality is uninformative.
#   "recently fed" -> mass looks fine, only movement/cortisol reveals poor condition.
#   "sedentary but healthy" -> low activity, only morphometrics confirm good condition.
recently_fed  <- rbinom(N, 1, 0.30) == 1
sedentary_ok  <- rbinom(N, 1, 0.30) == 1

# capture events per individual (repeated measures) - several per animal so the
# per-individual bias is learnable when random folds split an animal across train/val
K <- pmax(1, rpois(N, 4))
rows <- do.call(rbind, lapply(seq_len(N), function(i) {
  ni <- K[i]
  # event-level latent wobble (condition varies a little across captures)
  zi <- z[i] + rnorm(ni, 0, 0.20)

  # --- MORPHOMETRIC modality (obvious cases) ---
  morph_signal <- ifelse(recently_fed[i], 0.25, 1.0)   # muted if recently fed
  body_mass <- 45 + 6.0 * morph_signal * zi + b_morph[i] + rnorm(ni, 0, 1.6)
  chest_girth <- 70 + 4.5 * morph_signal * zi + 0.5 * b_morph[i] + rnorm(ni, 0, 1.8)
  kidney_fat <- pmax(0, 3 + 1.6 * morph_signal * zi + 0.4 * b_morph[i] + rnorm(ni, 0, 0.8))

  # --- MOVEMENT / PHYSIOLOGY modality (hidden cases) ---
  move_signal <- ifelse(sedentary_ok[i], 0.25, 1.0)    # muted if sedentary-but-ok
  odba <- 1.8 + 0.55 * move_signal * zi + 0.3 * b_move[i] + rnorm(ni, 0, 0.18)
  activity_hr <- 9 + 2.4 * move_signal * zi + 1.0 * b_move[i] + rnorm(ni, 0, 0.9)
  rest_bouts <- 12 - 1.8 * move_signal * zi - 0.6 * b_move[i] + rnorm(ni, 0, 1.1)
  cortisol <- pmax(1, 20 - 4.5 * move_signal * zi - 1.2 * b_move[i] + rnorm(ni, 0, 2.5))

  data.frame(
    animal_id = i,
    site = ((i - 1) %% 12) + 1,        # 12 sites (spatial clustering)
    sex = sample(c("F", "M"), ni, replace = TRUE),
    event = seq_len(ni),
    condition = cls[i],                # individual-level truth (constant per animal)
    body_mass, chest_girth, kidney_fat,
    odba, activity_hr, rest_bouts, cortisol,
    stringsAsFactors = FALSE
  )
}))

# introduce realistic missingness (sensors drop out)
for (c in c("cortisol", "kidney_fat", "odba", "rest_bouts")) {
  m <- rbinom(nrow(rows), 1, 0.07) == 1
  rows[[c]][m] <- NA
}

rows$condition <- factor(rows$condition, levels = c("poor", "fair", "good"))
write.csv(rows, file.path(here, "data", "captures.csv"), row.names = FALSE)

cat(sprintf("%d capture events across %d individuals, %d sites\n",
            nrow(rows), length(unique(rows$animal_id)), length(unique(rows$site))))
cat("class balance (events):\n"); print(round(prop.table(table(rows$condition)), 4))
cat(sprintf("mean captures/individual: %.2f\n", nrow(rows) / N))
cat("Saved data/captures.csv\n")
