# 03_abundance_model.R
# ---------------------------------------------------------------------------
# Ecological mirror of the House Prices modeling steps. Predict log(abundance)
# from habitat covariates using a blend of three diverse learners, the same
# recipe that worked on House Prices:
#
#   regularized linear (glmnet Lasso)  <- Lasso/Ridge/ElasticNet on Kaggle
#   random forest (ranger)             <- tree diversity
#   gradient boosting (gbm)            <- GradientBoosting/XGBoost on Kaggle
#
# Honest CV: aggregation-event outliers are dropped from the TRAINING side of
# each fold only (never validation), the same fix we applied in House Prices
# `04_stack.py`. Metric: RMSE on the log scale.
#
# Run:  Rscript 03_abundance_model.R  (sources 02_features.R)
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(glmnet); library(ranger); library(gbm); library(dplyr)
})

get_script_dir <- function() {
  args <- commandArgs(FALSE); fa <- grep("^--file=", args, value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa))))
  getwd()
}
here <- get_script_dir()
source(file.path(here, "02_features.R"))

dat <- read.csv(file.path(here, "data", "abundance_survey.csv"),
                stringsAsFactors = FALSE)
fe <- engineer(dat)
df <- fe$df; y <- fe$y; outlier <- fe$outlier
n <- nrow(df)
cat(sprintf("Sites: %d | predictors: %d | outliers flagged: %d\n",
            n, ncol(df), sum(outlier)))

# Design matrix for glmnet (one-hot factors); ranger/gbm use df directly.
Xlin <- model.matrix(~ . - 1, data = df)

set.seed(42)
folds <- sample(rep(1:5, length.out = n))
rmse <- function(a, b) sqrt(mean((a - b)^2))

oof <- list(lasso = numeric(n), ranger = numeric(n), gbm = numeric(n))

for (f in 1:5) {
  va <- which(folds == f)
  tr <- which(folds != f)
  tr <- tr[!outlier[tr]]                     # drop outliers from TRAIN only

  # Lasso
  cvfit <- cv.glmnet(Xlin[tr, ], y[tr], alpha = 1, nfolds = 5)
  oof$lasso[va] <- as.numeric(predict(cvfit, Xlin[va, ], s = "lambda.min"))

  # Random forest
  rf <- ranger(y = y[tr], x = df[tr, ], num.trees = 500,
               respect.unordered.factors = "order", seed = 42)
  oof$ranger[va] <- predict(rf, df[va, ])$predictions

  # Gradient boosting
  gb <- gbm(y ~ ., data = cbind(df[tr, ], y = y[tr]),
            distribution = "gaussian", n.trees = 800, interaction.depth = 3,
            shrinkage = 0.02, bag.fraction = 0.8, verbose = FALSE)
  oof$gbm[va] <- predict(gb, df[va, ], n.trees = 800)
}

cat("\nIndividual CV RMSE(log) [honest: full validation]:\n")
for (nm in names(oof)) cat(sprintf("  %-7s %.5f\n", nm, rmse(y, oof[[nm]])))

# --- Grid-search blend weights on the 3-model simplex ----------------------
grid <- expand.grid(a = seq(0, 1, 0.05), b = seq(0, 1, 0.05))
grid <- grid[grid$a + grid$b <= 1, ]
grid$c <- 1 - grid$a - grid$b
best <- NULL; best_rmse <- Inf
for (i in seq_len(nrow(grid))) {
  w <- as.numeric(grid[i, c("a", "b", "c")])
  pred <- w[1] * oof$lasso + w[2] * oof$ranger + w[3] * oof$gbm
  r <- rmse(y, pred)
  if (r < best_rmse) { best_rmse <- r; best <- w }
}
cat(sprintf("\nBest blend weights  lasso=%.2f ranger=%.2f gbm=%.2f\n",
            best[1], best[2], best[3]))
cat(sprintf("Blended CV RMSE(log): %.5f\n", best_rmse))

# --- Importance (ranger, permutation) --------------------------------------
rf_full <- ranger(y = y, x = df, num.trees = 800, importance = "permutation",
                  respect.unordered.factors = "order", seed = 42)
imp <- sort(rf_full$variable.importance, decreasing = TRUE)
cat("\nTop 10 features (permutation importance):\n")
print(round(head(imp, 10), 4))

# --- Save importance figure ------------------------------------------------
fig_dir <- file.path(here, "outputs", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages(library(ggplot2))
imp_df <- data.frame(feature = names(imp), importance = as.numeric(imp))
imp_df <- head(imp_df[order(-imp_df$importance), ], 12)
p <- ggplot(imp_df, aes(reorder(feature, importance), importance)) +
  geom_col(fill = "#4C72B0") + coord_flip() +
  labs(title = "Abundance model: permutation importance", x = NULL) +
  theme_minimal()
ggsave(file.path(fig_dir, "abundance_importance.png"), p,
       width = 7, height = 5, dpi = 120)
cat(sprintf("\nSaved %s\n", file.path(fig_dir, "abundance_importance.png")))
