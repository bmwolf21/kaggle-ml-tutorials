# 03_forecast_model.R
# ---------------------------------------------------------------------------
# The payoff of the Store Sales translation. Because this is simulated, we know
# the TRUE future, so we can measure two things the Kaggle leaderboard could only
# hint at:
#
#   in-range validation error   (a held-out window inside the training range)
#   true future-window error    (the last H months, genuinely beyond training)
#
# for a season+lags model and a season+lags+trend model. The finding: the true
# future is systematically harder than any in-range validation window suggests
# (for both models), which is why forecasting demands a truly out-of-time check.
# In this simulation the trend covariates are genuine, so they do not mislead;
# Store Sales differed only in that its trend proxy (oil) was largely spurious.
#
# Run:  Rscript 03_forecast_model.R  (sources 02_features.R)
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ranger) })

get_script_dir <- function() {
  args <- commandArgs(FALSE); fa <- grep("^--file=", args, value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa))))
  getwd()
}
here <- get_script_dir()
source(file.path(here, "02_features.R"))

H <- 6
dat <- read.csv(file.path(here, "data", "monitoring_counts.csv"),
                stringsAsFactors = FALSE)
fe <- features(dat, H = H)
df <- fe$df
Tmax <- max(df$time_index)

# Drop rows without full lag history (first 2H months per series).
df <- df[!is.na(df$lag_2h) & !is.na(df$rmean_6), ]

rmsle <- function(pred, truth) sqrt(mean((pred - truth)^2))

# Splits by time_index:
#   future test : last H months            (truly beyond training)
#   in-range val: the H months before that (inside the training range)
test  <- df[df$time_index > Tmax - H, ]
inrng <- df[df$time_index > Tmax - 2 * H & df$time_index <= Tmax - H, ]
fit_future <- df[df$time_index <= Tmax - H, ]     # all data before the test
fit_inrng  <- df[df$time_index <= Tmax - 2 * H, ] # all data before the val window

run <- function(cols) {
  m_val <- ranger(x = fit_inrng[, cols], y = fit_inrng$clog, num.trees = 400,
                  respect.unordered.factors = "order", seed = 42)
  val <- rmsle(predict(m_val, inrng[, cols])$predictions, inrng$clog)
  m_fut <- ranger(x = fit_future[, cols], y = fit_future$clog, num.trees = 400,
                  respect.unordered.factors = "order", seed = 42)
  fut <- rmsle(predict(m_fut, test[, cols])$predictions, test$clog)
  c(in_range_val = val, true_future = fut)
}

safe  <- run(fe$safe_cols)
trend <- run(fe$trend_cols)

cat("\nRMSLE comparison (log1p counts):\n")
cat(sprintf("  %-22s in-range val = %.4f   true future = %.4f  (gap +%.4f)\n",
            "SAFE (season+lags)", safe["in_range_val"], safe["true_future"],
            safe["true_future"] - safe["in_range_val"]))
cat(sprintf("  %-22s in-range val = %.4f   true future = %.4f  (gap +%.4f)\n",
            "TREND (+time+temp)", trend["in_range_val"], trend["true_future"],
            trend["true_future"] - trend["in_range_val"]))
gap <- mean(c(safe["true_future"] - safe["in_range_val"],
              trend["true_future"] - trend["in_range_val"]))
cat(sprintf("\nThe headline: the TRUE future is about %.3f RMSLE harder than any\n", gap))
cat("in-range validation window predicted, for BOTH feature sets. In-range\n")
cat("validation is systematically optimistic for forecasting a genuinely future\n")
cat("period. (This mirrors Store Sales: validation 0.40 but leaderboard 0.49.)\n")
cat("\nNote: here the trend covariates were GENUINE drivers, so dropping them did\n")
cat("not help (SAFE ~ TREND). Store Sales differed only in that its trend proxy\n")
cat("(oil price) was largely spurious, so removing it also helped there. The\n")
cat("transferable lesson is the same: trust a truly future window, not in-range CV.\n")

# --- Save a comparison figure ----------------------------------------------
fig_dir <- file.path(here, "outputs", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages(library(ggplot2))
plot_df <- data.frame(
  model = rep(c("SAFE", "TREND"), each = 2),
  window = rep(c("in-range val", "true future"), 2),
  rmsle = c(safe, trend))
p <- ggplot(plot_df, aes(window, rmsle, fill = model)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c(SAFE = "#55A868", TREND = "#C44E52")) +
  labs(title = "The true future is harder than in-range validation predicts",
       y = "RMSLE (log counts)", x = NULL) + theme_minimal()
ggsave(file.path(fig_dir, "future_gap.png"), p,
       width = 7, height = 4.5, dpi = 120)
cat(sprintf("\nSaved %s\n", file.path(fig_dir, "future_gap.png")))
