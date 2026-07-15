# 00_simulate_survey_data.R
# ---------------------------------------------------------------------------
# Angle 3 (translate): build a simulated wildlife detection survey whose data
# structure mirrors the Spaceship Titanic competition one-to-one, so the same
# workflow can be walked through in an ecological setting.
#
# Mapping of constructs (see wildlife_translation/README.md for the full table):
#   PassengerId group   -> transect_id (sites are nested in transects)
#   Cabin deck/num/side -> unit/plot/aspect location code
#   CryoSleep (asleep)  -> passive_site (camera-only, no active search)
#   5 spend amenities   -> minutes on 5 active survey methods
#   spend == 0 if asleep-> effort == 0 if passive (deterministic link)
#   HomePlanet          -> land_cover class (constant within a transect)
#   Age                 -> shrub_height (continuous habitat covariate)
#   Transported (0/1)   -> detected (species detected at the site, 0/1)
#
# Extra ecological realism the Kaggle data did not have:
#   - spatial coordinates + spatial autocorrelation, so that naive random CV
#     is optimistic and spatial-block CV is the honest estimate.
#
# Output: data/survey_sites.csv  (one row per surveyed site)
# ---------------------------------------------------------------------------

set.seed(2026)
suppressPackageStartupMessages(library(dplyr))

# Resolve this script's folder so it runs correctly no matter the working dir.
get_script_dir <- function() {
  args <- commandArgs(FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  getwd()
}
this_dir <- get_script_dir()
out_dir <- file.path(this_dir, "data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- 1. Transects laid across a 100 x 100 landscape ------------------------
n_transects <- 250
tx <- runif(n_transects, 0, 100)
ty <- runif(n_transects, 0, 100)

# Sites per transect: many small parties, a few large (mirrors passenger group
# sizes). Draw 1..8 with a decaying probability.
grp_probs <- c(0.42, 0.20, 0.13, 0.07, 0.06, 0.05, 0.04, 0.03)
sites_per <- sample(1:8, n_transects, replace = TRUE, prob = grp_probs)

# land cover is a property of the transect (constant within transect), which is
# exactly what lets group-based imputation recover missing values later.
land_levels <- c("Grassland", "Shrubland", "Woodland", "Riparian")
tx_landcover <- sample(land_levels, n_transects, replace = TRUE,
                       prob = c(0.4, 0.3, 0.2, 0.1))
mgmt_levels <- c("North", "Central", "South")
tx_mgmt <- sample(mgmt_levels, n_transects, replace = TRUE)

rows <- list()
sid <- 1
for (t in seq_len(n_transects)) {
  for (p in seq_len(sites_per[t])) {
    # site location jittered around its transect -> spatial clustering
    x <- pmin(pmax(tx[t] + rnorm(1, 0, 3), 0), 100)
    y <- pmin(pmax(ty[t] + rnorm(1, 0, 3), 0), 100)
    rows[[sid]] <- data.frame(
      transect_id = sprintf("T%04d", t),
      site_in_transect = sprintf("%02d", p),
      x = x, y = y,
      land_cover = tx_landcover[t],
      mgmt_unit = tx_mgmt[t],
      stringsAsFactors = FALSE
    )
    sid <- sid + 1
  }
}
dat <- bind_rows(rows)
n <- nrow(dat)

# Combined site id mirrors PassengerId "gggg_pp".
dat$site_id <- paste0(dat$transect_id, "_", dat$site_in_transect)

# location code "unit/plot/aspect" mirrors Cabin "deck/num/side".
aspect <- sample(c("N", "S"), n, replace = TRUE)          # slope aspect
plot_no <- sample(1:1800, n, replace = TRUE)
unit <- sample(LETTERS[1:6], n, replace = TRUE)
dat$location_code <- paste(unit, plot_no, aspect, sep = "/")

# --- 2. Habitat covariates -------------------------------------------------
# canopy_cover and ndvi are deliberately collinear (to reproduce the
# correlated-features lesson from the Kaggle importance analysis).
dat$canopy_cover <- pmin(pmax(rnorm(n, 40, 20), 0), 100)
dat$ndvi <- pmin(pmax(0.2 + 0.006 * dat$canopy_cover + rnorm(n, 0, 0.05), 0), 1)
dat$dist_to_water_m <- round(rexp(n, rate = 1 / 400))
dat$elevation_m <- round(rnorm(n, 900, 150))
dat$shrub_height_cm <- round(pmax(rnorm(n, 60, 30), 0))   # the "Age"-like var

# --- 3. Survey effort with the deterministic zero structure ----------------
# ~ 30% of sites are "passive" (camera-only): no active search minutes at all.
dat$passive_site <- runif(n) < 0.30
methods <- c("visual_scan_min", "call_playback_min", "track_survey_min",
             "scat_survey_min", "spotlight_min")
for (m in methods) {
  minutes <- round(pmax(rexp(n, rate = 1 / 15), 0))
  minutes[dat$passive_site] <- 0          # passive sites: zero active effort
  dat[[m]] <- minutes
}
dat$total_effort_min <- rowSums(dat[, methods])

# --- 4. Detection outcome (with spatial autocorrelation) -------------------
z <- function(v) as.numeric(scale(v))
# Latent spatial field: several localized hotspots of unmeasured suitability
# (prey density, microclimate, etc. that we did not record as covariates). This
# creates genuine spatial autocorrelation in detections. Because it is driven by
# location and NOT by our measured covariates, a model given site coordinates
# can interpolate it locally, which is exactly what inflates random CV relative
# to spatial-block CV.
set.seed(11)
n_hot <- 8
hcx <- runif(n_hot, 0, 100); hcy <- runif(n_hot, 0, 100)
hsig <- runif(n_hot, 7, 12); hamp <- runif(n_hot, -2.2, 2.2)
spatial_field <- rowSums(sapply(seq_len(n_hot), function(k)
  hamp[k] * exp(-((dat$x - hcx[k])^2 + (dat$y - hcy[k])^2) / (2 * hsig[k]^2))))
spatial_field <- 1.8 * as.numeric(scale(spatial_field))
set.seed(2026)  # restore main stream

lc_effect <- c(Grassland = 0.5, Shrubland = 0.2, Woodland = -0.3,
               Riparian = 0.9)[dat$land_cover]

lp <- -0.4 +
  0.9 * z(log1p(dat$total_effort_min)) +   # more effort -> more detections
  0.6 * z(dat$canopy_cover) +
  0.7 * z(-dat$dist_to_water_m) +          # closer to water -> more
  0.3 * z(dat$shrub_height_cm) +
  as.numeric(lc_effect) +
  spatial_field
p <- 1 / (1 + exp(-lp))
dat$detected <- rbinom(n, 1, p)

# --- 5. Inject realistic missingness (~2.5% per column, pervasive) ---------
miss_cols <- c("land_cover", "mgmt_unit", "canopy_cover", "ndvi",
               "dist_to_water_m", "elevation_m", "shrub_height_cm",
               "passive_site", methods, "location_code")
for (c in miss_cols) {
  idx <- runif(n) < 0.025
  dat[[c]][idx] <- NA
}

# --- 6. Save ---------------------------------------------------------------
keep <- c("site_id", "transect_id", "site_in_transect", "x", "y",
          "location_code", "land_cover", "mgmt_unit",
          "canopy_cover", "ndvi", "dist_to_water_m", "elevation_m",
          "shrub_height_cm", "passive_site", methods, "total_effort_min",
          "detected")
dat <- dat[, keep]
write.csv(dat, file.path(out_dir, "survey_sites.csv"), row.names = FALSE)

cat(sprintf("Wrote %s: %d sites across %d transects\n",
            file.path(out_dir, "survey_sites.csv"), n, n_transects))
cat(sprintf("Detection rate: %.1f%%\n", 100 * mean(dat$detected)))
cat(sprintf("Passive (zero-effort) sites: %.1f%%\n",
            100 * mean(dat$passive_site, na.rm = TRUE)))
