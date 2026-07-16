# 02_features.R
# ---------------------------------------------------------------------------
# Horizon-aware feature engineering for the population-forecasting translation.
# Ecological mirror of the Store Sales `src/features.py`. Forecast horizon is
# H = 6 months, so every lag / rolling feature is shifted by >= 6 months and can
# never see inside the forecast window.
#
# Provides features(H) -> list(df, safe_cols, trend_cols). The two column sets
# differ only in whether they include non-extrapolating trend covariates
# (time_index, raw temperature), so the model script can compare them.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(dplyr))

features <- function(dat, H = 6) {
  dat <- dat %>% arrange(site, species, time_index) %>%
    group_by(site, species) %>%
    mutate(
      clog   = log1p(count),
      lag_h  = dplyr::lag(clog, H),
      lag_2h = dplyr::lag(clog, 2 * H),
      ma_3   = as.numeric(stats::filter(clog, rep(1 / 3, 3), sides = 1)),
      ma_6   = as.numeric(stats::filter(clog, rep(1 / 6, 6), sides = 1)),
      rmean_3 = dplyr::lag(ma_3, H),        # horizon-safe rolling means
      rmean_6 = dplyr::lag(ma_6, H)
    ) %>% ungroup()

  # Cyclical month encoding: repeats every year, so it generalizes forward
  # (unlike a raw time_index counter, which a tree cannot extend).
  dat$month_sin <- sin(2 * pi * dat$month / 12)
  dat$month_cos <- cos(2 * pi * dat$month / 12)

  for (c in c("habitat", "region", "site", "species"))
    dat[[c]] <- factor(dat[[c]])

  safe_cols <- c("month_sin", "month_cos", "lag_h", "lag_2h", "rmean_3",
                 "rmean_6", "elevation", "intervention",
                 "habitat", "region", "site", "species")
  # trend_cols add the two features that cannot extrapolate to a future window.
  trend_cols <- c(safe_cols, "time_index", "temperature")

  list(df = dat, safe_cols = safe_cols, trend_cols = trend_cols)
}
