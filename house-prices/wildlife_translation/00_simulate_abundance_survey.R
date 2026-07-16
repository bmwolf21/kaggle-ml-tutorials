# 00_simulate_abundance_survey.R
# ---------------------------------------------------------------------------
# Angle 3 (translate) for House Prices: a simulated abundance survey whose data
# structure mirrors the House Prices regression one-to-one. Here the continuous
# target is animal ABUNDANCE at a survey site (a count), the House Prices
# analogue of SalePrice.
#
# Construct mapping (see wildlife_translation/README.md for the full table):
#   SalePrice (continuous, right-skewed)   -> abundance (count, right-skewed)
#   Garage/Pool present or absent          -> riparian/canopy feature present or absent
#   NA garage attrs when no garage         -> NA riparian attrs when no riparian zone
#   ExterQual Ex/Gd/TA/Fa (ordinal)        -> forage/cover quality (ordinal)
#   LotFrontage missing -> Neighborhood    -> soil_moisture missing -> region median
#   GrLivArea > 4000 outliers              -> aggregation-event sites (anomalous counts)
#
# Output: data/abundance_survey.csv (one row per surveyed site)
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

n <- 800
z <- function(v) as.numeric(scale(v))

# --- Regions (sites nested in regions; some covariates constant per region) --
n_region <- 40
region <- sprintf("R%02d", sample(seq_len(n_region), n, replace = TRUE))
land_levels <- c("Grassland", "Shrubland", "Woodland", "Wetland", "Cropland")
region_landcover <- setNames(
  sample(land_levels, n_region, replace = TRUE), sprintf("R%02d", 1:n_region))

dat <- data.frame(
  site_id = sprintf("S%04d", seq_len(n)),
  region = region,
  land_cover = region_landcover[region],
  stringsAsFactors = FALSE
)

# --- Continuous habitat covariates -----------------------------------------
dat$ndvi <- pmin(pmax(rnorm(n, 0.5, 0.15), 0), 1)
dat$elevation_m <- round(rnorm(n, 900, 200))
dat$temp_c <- round(rnorm(n, 15, 4), 1)
dat$shrub_cover_pct <- pmin(pmax(rnorm(n, 35, 18), 0), 100)
dat$soil_moisture <- pmin(pmax(rnorm(n, 30, 10), 0), 100)   # will get genuine NAs

# --- Ordinal habitat quality (always present) ------------------------------
qual_levels <- c("Poor", "Fair", "Good", "Excellent")
dat$forage_quality <- sample(qual_levels, n, replace = TRUE, prob = c(.3, .3, .25, .15))
dat$cover_quality  <- sample(qual_levels, n, replace = TRUE, prob = c(.25, .35, .25, .15))

# --- Structural "feature present / absent" covariates ----------------------
# Riparian zone (like a garage): present ~35%; its attributes are NA if absent.
dat$riparian_present <- runif(n) < 0.35
dat$riparian_quality <- ifelse(dat$riparian_present,
  sample(qual_levels, n, replace = TRUE), NA)
dat$riparian_width_m <- ifelse(dat$riparian_present,
  round(pmax(rnorm(n, 12, 6), 1)), NA)

# Canopy (like a pool/2nd floor): present ~60%.
dat$canopy_present <- runif(n) < 0.60
dat$canopy_quality <- ifelse(dat$canopy_present,
  sample(qual_levels, n, replace = TRUE), NA)
dat$canopy_height_m <- ifelse(dat$canopy_present,
  round(pmax(rnorm(n, 15, 6), 1)), NA)

# --- Density outcome (continuous, right-skewed) ----------------------------
# A continuous density index (animals per km2), the SalePrice analogue.
# Lognormal so log(density) is near-normal. Includes a nonlinear threshold
# effect and an interaction so tree models add value beyond the linear model
# (exactly the linear-plus-tree diversity that helped on House Prices).
qnum <- function(x) c(Poor = 0, Fair = 1, Good = 2, Excellent = 3)[x]
lp <- 1.1 +
  0.45 * z(dat$ndvi) +
  0.35 * z(dat$shrub_cover_pct) +
  0.30 * z(-dat$elevation_m) +
  0.25 * z(dat$soil_moisture) +
  0.45 * qnum(dat$forage_quality) / 3 +
  0.25 * qnum(dat$cover_quality) / 3 +
  0.55 * dat$riparian_present +
  0.30 * dat$canopy_present +
  0.50 * (dat$soil_moisture > 45) +               # nonlinear threshold effect
  0.45 * dat$riparian_present * z(dat$ndvi) +     # riparian x greenness interaction
  c(Grassland = 0.2, Shrubland = 0.1, Woodland = -0.1,
    Wetland = 0.6, Cropland = -0.3)[dat$land_cover]
dat$density <- round(exp(lp + rnorm(n, 0, 0.30)), 2)

# --- Aggregation-event outliers (like GrLivArea > 4000) --------------------
# Transient swarms multiply local density roughly an order of magnitude.
out_idx <- sample(seq_len(n), 6)
dat$density[out_idx] <- round(dat$density[out_idx] * runif(6, 8, 14), 2)

# --- Genuine (non-structural) missingness ----------------------------------
dat$soil_moisture[runif(n) < 0.16] <- NA          # like LotFrontage
for (c in c("ndvi", "temp_c", "forage_quality", "cover_quality",
            "land_cover", "shrub_cover_pct")) {
  dat[[c]][runif(n) < 0.02] <- NA
}

write.csv(dat, file.path(out_dir, "abundance_survey.csv"), row.names = FALSE)
skew <- function(x) { m <- mean(x); mean((x - m)^3) / sd(x)^3 }
cat(sprintf("Wrote %s: %d sites\n", file.path(out_dir, "abundance_survey.csv"), n))
cat(sprintf("Density: median %.1f, max %.1f, skew raw %.2f, skew log %.2f\n",
            median(dat$density), max(dat$density),
            skew(dat$density), skew(log(dat$density))))
cat(sprintf("Riparian present: %.0f%% | Canopy present: %.0f%%\n",
            100 * mean(dat$riparian_present), 100 * mean(dat$canopy_present)))
