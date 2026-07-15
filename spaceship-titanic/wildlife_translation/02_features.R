# 02_features.R
# ---------------------------------------------------------------------------
# Feature engineering for the wildlife detection walkthrough. This is the
# ecological mirror of the Kaggle `src/features.py`. Each block notes the
# Spaceship Titanic step it corresponds to.
#
# Provides build_features(dat) -> list(X = model-ready data.frame,
#                                      y = detected, coords = x/y).
# Sourced by 03_detection_model.R.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(dplyr))

# Fill missing values of `col` with the most common value within each `by`
# group (mirror of features.py _fill_by_group_mode). Leak-safe: no outcome used.
fill_by_group_mode <- function(df, col, by) {
  mode_of <- function(v) {
    v <- v[!is.na(v)]
    if (!length(v)) return(NA)
    names(sort(table(v), decreasing = TRUE))[1]
  }
  lookup <- tapply(df[[col]], df[[by]], mode_of)
  filled <- df[[col]]
  na_idx <- is.na(filled)
  filled[na_idx] <- lookup[as.character(df[[by]][na_idx])]
  filled
}

build_features <- function(dat) {
  methods <- c("visual_scan_min", "call_playback_min", "track_survey_min",
               "scat_survey_min", "spotlight_min")

  # --- Decode transect structure (mirror: PassengerId -> Group/GroupSize) ---
  dat <- dat %>%
    group_by(transect_id) %>%
    mutate(transect_size = n()) %>%
    ungroup()
  dat$is_isolated <- as.integer(dat$transect_size == 1)

  # --- Decode location_code unit/plot/aspect (mirror: Cabin deck/num/side) --
  parts <- do.call(rbind, strsplit(ifelse(is.na(dat$location_code), "NA/NA/NA",
                                           dat$location_code), "/"))
  dat$unit <- ifelse(parts[, 1] == "NA", NA, parts[, 1])
  dat$plot_no <- suppressWarnings(as.numeric(parts[, 2]))
  dat$aspect <- ifelse(parts[, 3] == "NA", NA, parts[, 3])

  # --- Deterministic effort logic (mirror: CryoSleep <-> spend) -------------
  # Any active survey minutes recorded => the site was actively surveyed.
  active_total <- rowSums(dat[, methods], na.rm = TRUE)
  dat$passive_site[active_total > 0] <-
    ifelse(is.na(dat$passive_site[active_total > 0]), FALSE,
           dat$passive_site[active_total > 0])
  # Passive sites have zero active effort => fill missing method minutes with 0.
  passive <- which(dat$passive_site %in% TRUE)
  for (m in methods) dat[[m]][passive] <- ifelse(is.na(dat[[m]][passive]), 0,
                                                  dat[[m]][passive])
  # Any remaining missing method minutes -> 0 (no record = no effort logged).
  for (m in methods) dat[[m]][is.na(dat[[m]])] <- 0
  dat$total_effort_min <- rowSums(dat[, methods])
  dat$has_effort <- as.numeric(dat$total_effort_min > 0)

  # --- Group-based imputation of categoricals (mirror: HomePlanet by Group) -
  # land_cover and mgmt_unit are constant within a transect, so recover from
  # transect-mates instead of a global guess.
  dat$land_cover <- fill_by_group_mode(dat, "land_cover", "transect_id")
  dat$mgmt_unit  <- fill_by_group_mode(dat, "mgmt_unit",  "transect_id")
  dat$aspect     <- fill_by_group_mode(dat, "aspect",     "transect_id")

  # --- Effort structure (mirror: Luxury vs Basic spend) ---------------------
  # Ground-sign methods vs. active-detection methods behave differently.
  dat$ground_effort <- rowSums(dat[, c("track_survey_min", "scat_survey_min")])
  dat$active_effort <- rowSums(dat[, c("visual_scan_min", "call_playback_min",
                                       "spotlight_min")])
  dat$n_methods_used <- rowSums(dat[, methods] > 0)
  dat$total_effort_log <- log1p(dat$total_effort_min)
  for (m in methods) dat[[paste0(m, "_log")]] <- log1p(dat[[m]])

  # --- Transect-level aggregates (mirror: GroupSpendMean etc.) --------------
  dat <- dat %>%
    group_by(transect_id) %>%
    mutate(transect_effort_mean = mean(total_effort_min, na.rm = TRUE),
           transect_canopy_mean = mean(canopy_cover, na.rm = TRUE)) %>%
    ungroup()

  # --- Numeric imputation (median) + flags ----------------------------------
  num_cols <- c("canopy_cover", "ndvi", "dist_to_water_m", "elevation_m",
                "shrub_height_cm", "plot_no", "transect_canopy_mean")
  for (c in num_cols) {
    med <- median(dat[[c]], na.rm = TRUE)
    dat[[c]][is.na(dat[[c]])] <- med
  }
  dat$is_child_habitat <- as.numeric(dat$shrub_height_cm < 20)  # sparse-shrub flag

  # --- Assemble model matrix ------------------------------------------------
  cat_cols <- c("land_cover", "mgmt_unit", "aspect", "unit")
  for (c in cat_cols) dat[[c]] <- factor(ifelse(is.na(dat[[c]]), "Unknown",
                                                dat[[c]]))
  dat$passive_site <- factor(ifelse(is.na(dat$passive_site), "Unknown",
                                    as.character(dat$passive_site)))

  feature_cols <- c(
    "x", "y",  # site coordinates: common SDM predictors; also the mechanism
               # that lets random CV interpolate spatial structure (see 03).
    "transect_size", "is_isolated", "plot_no",
    "canopy_cover", "ndvi", "dist_to_water_m", "elevation_m", "shrub_height_cm",
    "is_child_habitat",
    "total_effort_min", "total_effort_log", "has_effort",
    "ground_effort", "active_effort", "n_methods_used",
    "transect_effort_mean", "transect_canopy_mean",
    methods, paste0(methods, "_log"),
    cat_cols, "passive_site"
  )

  list(
    X = dat[, feature_cols],
    y = factor(dat$detected, levels = c(0, 1)),
    coords = dat[, c("x", "y")],
    transect = dat$transect_id
  )
}
