# 03_detection_model.R
# ---------------------------------------------------------------------------
# The ecological mirror of the Kaggle modeling steps. A random forest (ranger)
# predicts species detection from habitat + effort features. The headline is a
# comparison the Kaggle tabular problem could not show:
#
#   random k-fold CV  vs  spatial-block CV
#
# Random CV lets spatially-adjacent sites fall in both train and test folds.
# Because detection is spatially autocorrelated, that leaks information and
# inflates the score. Spatial-block CV holds out whole regions and gives the
# honest estimate. This is the ecology-specific form of the Kaggle lesson
# "trust the honest held-out signal, not the optimistic in-sample one."
#
# Run:  Rscript 03_detection_model.R
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(ranger); library(pROC); library(dplyr); library(ggplot2)
})

get_script_dir <- function() {
  args <- commandArgs(FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa))))
  getwd()
}
here <- get_script_dir()
source(file.path(here, "02_features.R"))

dat <- read.csv(file.path(here, "data", "survey_sites.csv"),
                stringsAsFactors = FALSE)
fe <- build_features(dat)
X <- fe$X; y <- fe$y; coords <- fe$coords
cat(sprintf("Sites: %d | features: %d | detection rate: %.1f%%\n",
            nrow(X), ncol(X), 100 * mean(y == 1)))

fit_predict <- function(tr, va) {
  d_tr <- cbind(X[tr, ], detected = y[tr])
  rf <- ranger(detected ~ ., data = d_tr, num.trees = 500,
               probability = TRUE, respect.unordered.factors = "order",
               seed = 42)
  predict(rf, X[va, ])$predictions[, "1"]
}

evaluate_cv <- function(folds, label) {
  accs <- c(); aucs <- c()
  oof <- rep(NA_real_, nrow(X))
  for (f in sort(unique(folds))) {
    va <- which(folds == f); tr <- which(folds != f)
    p <- fit_predict(tr, va)
    oof[va] <- p
    accs <- c(accs, mean((p > 0.5) == (y[va] == 1)))
    aucs <- c(aucs, as.numeric(pROC::auc(y[va], p, quiet = TRUE)))
  }
  cat(sprintf("%-20s acc %.4f +/- %.4f  |  AUC %.4f +/- %.4f\n",
              label, mean(accs), sd(accs), mean(aucs), sd(aucs)))
  invisible(list(acc = mean(accs), auc = mean(aucs), oof = oof))
}

set.seed(42)

# --- 1. Random 5-fold CV ---------------------------------------------------
rand_folds <- sample(rep(1:5, length.out = nrow(X)))
cat("\n--- Cross-validation comparison ---\n")
res_rand <- evaluate_cv(rand_folds, "Random 5-fold CV")

# --- 2. Spatial-block 5-fold CV --------------------------------------------
# Tile the landscape into a 5x5 grid of blocks, then assign whole blocks to
# folds so training and test regions never touch.
bx <- cut(coords$x, breaks = seq(0, 100, length.out = 6), include.lowest = TRUE)
by <- cut(coords$y, breaks = seq(0, 100, length.out = 6), include.lowest = TRUE)
block <- factor(paste(bx, by))
block_ids <- levels(block)
block_fold <- setNames(sample(rep(1:5, length.out = length(block_ids))),
                       block_ids)
spatial_folds <- block_fold[as.character(block)]
res_spat <- evaluate_cv(spatial_folds, "Spatial-block CV")

cat(sprintf("\nOptimism from ignoring space: acc +%.4f, AUC +%.4f\n",
            res_rand$acc - res_spat$acc, res_rand$auc - res_spat$auc))
cat("=> Random CV is the optimistic (misleading) estimate; the spatial-block\n",
    "   number is what to trust and report.\n", sep = "")

# --- 3. Feature importance (permutation) -----------------------------------
d_all <- cbind(X, detected = y)
rf_imp <- ranger(detected ~ ., data = d_all, num.trees = 800,
                 probability = TRUE, importance = "permutation",
                 respect.unordered.factors = "order", seed = 42)
imp <- sort(rf_imp$variable.importance, decreasing = TRUE)
cat("\nTop 12 features (permutation importance):\n")
print(round(head(imp, 12), 5))

# --- 4. Collinearity caution (mirror: CryoSleep hidden by spend) -----------
cc <- cor(X$canopy_cover, X$ndvi, use = "complete.obs")
cat(sprintf("\ncor(canopy_cover, ndvi) = %.2f -> collinear; their importance is\n", cc))
cat("split between them, just as CryoSleep's signal hid inside the spend\n")
cat("features on Kaggle. Interpret importance with correlation in mind.\n")

# --- 5. Save importance figure ---------------------------------------------
fig_dir <- file.path(here, "outputs", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
imp_df <- data.frame(feature = names(imp), importance = as.numeric(imp)) %>%
  arrange(desc(importance)) %>% head(15)
p <- ggplot(imp_df, aes(reorder(feature, importance), importance)) +
  geom_col(fill = "#4C72B0") + coord_flip() +
  labs(title = "Detection model: permutation importance",
       x = NULL, y = "Importance") + theme_minimal()
ggsave(file.path(fig_dir, "wildlife_importance.png"), p,
       width = 7, height = 5, dpi = 120)
cat(sprintf("\nSaved %s\n", file.path(fig_dir, "wildlife_importance.png")))
