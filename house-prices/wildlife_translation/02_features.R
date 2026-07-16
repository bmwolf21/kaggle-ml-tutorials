# 02_features.R
# ---------------------------------------------------------------------------
# Feature engineering for the abundance-regression translation. Ecological
# mirror of the House Prices `src/features.py`. Each block notes the House
# Prices step it corresponds to. Target is log1p(abundance).
#
# Provides engineer(dat) -> list(df = predictors (with factors), y = log target,
#                                 outlier = logical mask).
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(dplyr))

QMAP <- c(None = 0, Poor = 1, Fair = 2, Good = 3, Excellent = 4)

engineer <- function(dat) {
  y <- log(dat$density)
  # Aggregation-event outliers (mirror GrLivArea > 4000): flag with the standard
  # Tukey rule on the log target (Q3 + 1.5*IQR), a principled, general anomaly
  # test rather than a magic threshold. Flagged, not modeled on.
  q <- quantile(y, c(0.25, 0.75))
  outlier <- y > q[2] + 1.5 * (q[2] - q[1])

  df <- dat

  # --- Structural NA handling (mirror: NA garage/pool -> "None"/0) ----------
  # A missing riparian/canopy attribute means the feature is ABSENT, not
  # unrecorded, so fill categoricals with "None" and sizes with 0.
  df$riparian_quality[is.na(df$riparian_quality)] <- "None"
  df$canopy_quality[is.na(df$canopy_quality)]     <- "None"
  df$riparian_width_m[is.na(df$riparian_width_m)] <- 0
  df$canopy_height_m[is.na(df$canopy_height_m)]   <- 0

  # --- Ordinal quality encoding (mirror: ExterQual Ex/Gd/TA/Fa) ------------
  for (c in c("forage_quality", "cover_quality", "riparian_quality",
              "canopy_quality")) {
    df[[c]] <- unname(QMAP[ifelse(is.na(df[[c]]), "None", df[[c]])])
  }

  # --- Neighbor-based imputation (mirror: LotFrontage by Neighborhood) ------
  # soil_moisture is genuinely missing; fill with the median of its region.
  df$soil_moisture <- ave(df$soil_moisture, df$region, FUN = function(v) {
    v[is.na(v)] <- median(v, na.rm = TRUE); v })
  df$soil_moisture[is.na(df$soil_moisture)] <-
    median(df$soil_moisture, na.rm = TRUE)          # regions with all-NA

  # --- Remaining imputation -------------------------------------------------
  num_cols <- c("ndvi", "elevation_m", "temp_c", "shrub_cover_pct")
  for (c in num_cols) df[[c]][is.na(df[[c]])] <- median(df[[c]], na.rm = TRUE)
  mode_of <- function(v) names(sort(table(v), decreasing = TRUE))[1]
  df$land_cover[is.na(df$land_cover)] <- mode_of(df$land_cover)

  # --- Engineered features (mirror: TotalSF, HasGarage, etc.) --------------
  df$has_riparian <- as.integer(df$riparian_present)
  df$has_canopy   <- as.integer(df$canopy_present)
  df$quality_sum  <- df$forage_quality + df$cover_quality +
                     df$riparian_quality + df$canopy_quality
  df$structure_size <- df$riparian_width_m + df$canopy_height_m
  df$resource_index <- scale(df$ndvi) + scale(df$shrub_cover_pct) +
                       scale(df$soil_moisture)

  # --- Log-transform skewed positive numerics (mirror: skew > 0.75) --------
  for (c in c("riparian_width_m", "canopy_height_m", "structure_size")) {
    df[[c]] <- log1p(df[[c]])
  }

  # --- Assemble: drop ids/raw target; factors for ranger --------------------
  df$region <- factor(df$region)
  df$land_cover <- factor(df$land_cover)
  df$riparian_present <- as.integer(df$riparian_present)
  df$canopy_present <- as.integer(df$canopy_present)
  df <- df[, setdiff(names(df), c("site_id", "density"))]

  list(df = df, y = y, outlier = outlier)
}
