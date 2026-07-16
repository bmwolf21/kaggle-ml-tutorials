# 00_simulate_monitoring_series.R
# ---------------------------------------------------------------------------
# Angle 3 (translate) for Store Sales: a simulated multi-series wildlife
# MONITORING program whose structure mirrors the Store Sales forecasting task.
# Many parallel series (site x species) of periodic COUNTS over time, with
# seasonality, an external climate driver, occasional interventions, and a slow
# multi-year population trend. We forecast the last H periods.
#
# Construct mapping (see wildlife_translation/README.md):
#   daily sales per (store, family)  -> monthly counts per (site, species)
#   16-day forecast horizon          -> H-month forecast horizon
#   weekly/yearly seasonality        -> breeding/phenology seasonality
#   oil price (macro trend driver)   -> temperature / a trending covariate
#   promotions                       -> management interventions
#   the extrapolation trap (year/oil)-> time_index and raw climate that a tree
#                                       cannot extend past the training range
#
# Because this is simulated, we KEEP the true values of the future test window,
# so the translation can show the extrapolation trap catching a model red-handed
# (the Kaggle leaderboard could only hint at it).
#
# Output: data/monitoring_counts.csv
# ---------------------------------------------------------------------------
set.seed(2026)
suppressPackageStartupMessages(library(dplyr))

get_script_dir <- function() {
  args <- commandArgs(FALSE); fa <- grep("^--file=", args, value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa))))
  getwd()
}
out_dir <- file.path(get_script_dir(), "data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Dimensions -------------------------------------------------------------
n_sites <- 15
n_species <- 6
months <- seq(as.Date("2015-01-01"), as.Date("2022-12-01"), by = "month")
T <- length(months)                       # 96 months

habitats <- c("Grassland", "Woodland", "Wetland", "Shrubland")
regions <- c("North", "Central", "South")
site_meta <- data.frame(
  site = sprintf("SITE%02d", 1:n_sites),
  habitat = sample(habitats, n_sites, replace = TRUE),
  region = sample(regions, n_sites, replace = TRUE),
  elevation = round(rnorm(n_sites, 900, 250)),
  site_effect = rnorm(n_sites, 0, 0.4),
  stringsAsFactors = FALSE
)
species <- sprintf("SP%02d", 1:n_species)
sp_base <- setNames(runif(n_species, 0.5, 3.0), species)      # log baseline
sp_peak <- setNames(sample(1:12, n_species, replace = TRUE), species)  # peak month
sp_amp  <- setNames(runif(n_species, 0.6, 1.4), species)      # seasonal amplitude

# --- Climate driver (seasonal + warming trend) ------------------------------
temp <- 15 + 8 * sin(2 * pi * (seq_len(T)) / 12) + 0.03 * seq_len(T) +
        rnorm(T, 0, 1)

rows <- list(); k <- 1
for (si in seq_len(n_sites)) {
  sm <- site_meta[si, ]
  for (sp in species) {
    # occasional management interventions at this series (like promotions)
    interv <- as.integer(runif(T) < 0.04)
    for (t in seq_len(T)) {
      m <- as.integer(format(months[t], "%m"))
      seasonal <- sp_amp[sp] * cos(2 * pi * (m - sp_peak[sp]) / 12)
      trend <- -0.010 * t                       # slow multi-year decline
      log_mu <- sp_base[sp] + sm$site_effect + seasonal + trend +
                0.02 * (temp[t] - 15) + 0.9 * interv[t]
      rows[[k]] <- data.frame(
        date = months[t], time_index = t,
        year = as.integer(format(months[t], "%Y")), month = m,
        site = sm$site, species = sp,
        habitat = sm$habitat, region = sm$region, elevation = sm$elevation,
        temperature = round(temp[t], 2), intervention = interv[t],
        count = rpois(1, exp(log_mu)), stringsAsFactors = FALSE)
      k <- k + 1
    }
  }
}
dat <- bind_rows(rows)

write.csv(dat, file.path(out_dir, "monitoring_counts.csv"), row.names = FALSE)
cat(sprintf("Wrote %s: %d rows (%d series x %d months)\n",
            file.path(out_dir, "monitoring_counts.csv"), nrow(dat),
            n_sites * n_species, T))
cat(sprintf("Count: median %d, max %d, zeros %.1f%%\n",
            median(dat$count), max(dat$count), 100 * mean(dat$count == 0)))
cat(sprintf("Date range: %s to %s\n", min(dat$date), max(dat$date)))
