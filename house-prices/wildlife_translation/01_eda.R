# 01_eda.R
# ---------------------------------------------------------------------------
# Exploratory analysis of the abundance survey, mirroring House Prices
# `src/01_eda.py`. Prints findings and saves figures to outputs/figures/.
# Run:  Rscript 01_eda.R
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ggplot2) })

get_script_dir <- function() {
  args <- commandArgs(FALSE); fa <- grep("^--file=", args, value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa))))
  getwd()
}
here <- get_script_dir()
fig <- file.path(here, "outputs", "figures")
dir.create(fig, showWarnings = FALSE, recursive = TRUE)
dat <- read.csv(file.path(here, "data", "abundance_survey.csv"),
                stringsAsFactors = FALSE)

skew <- function(x) { m <- mean(x); mean((x - m)^3) / sd(x)^3 }
cat("=== TARGET (density) ===\n")
cat(sprintf("median %.1f | max %.1f | skew raw %.2f | skew log %.2f\n",
            median(dat$density), max(dat$density),
            skew(dat$density), skew(log(dat$density))))
cat("-> right-skewed density; model on log scale (like SalePrice).\n")

cat("\n=== MISSINGNESS (% per column, where > 0) ===\n")
miss <- sort(round(100 * colMeans(is.na(dat)), 2), decreasing = TRUE)
print(miss[miss > 0])
cat("NOTE: riparian_* and canopy_* NAs are STRUCTURAL (feature absent), not\n")
cat("      unrecorded. soil_moisture is genuinely missing -> region-median fill.\n")

cat("\n=== STRUCTURAL CHECK: attrs are NA exactly when feature absent ===\n")
cat(sprintf("riparian absent: %d | riparian_quality NA: %d (should match)\n",
            sum(!dat$riparian_present), sum(is.na(dat$riparian_quality))))

cat("\n=== DENSITY BY LAND COVER ===\n")
print(dat %>% filter(!is.na(land_cover)) %>% group_by(land_cover) %>%
        summarise(mean_density = round(mean(density), 1), n = n(), .groups = "drop"))

# Figures (saved separately to avoid extra layout dependencies)
p1 <- ggplot(dat, aes(density)) + geom_histogram(bins = 40, fill = "#4C72B0") +
  labs(title = "Density (raw, right-skewed)") + theme_minimal()
ggsave(file.path(fig, "abundance_target_raw.png"), p1, width = 6, height = 4, dpi = 120)
p2 <- ggplot(dat, aes(log(density))) + geom_histogram(bins = 40, fill = "#55A868") +
  labs(title = "log(density) (near-normal)") + theme_minimal()
ggsave(file.path(fig, "abundance_target_log.png"), p2, width = 6, height = 4, dpi = 120)
cat("\nSaved outputs/figures/abundance_target_raw.png, abundance_target_log.png\n")
