# 01_eda.R
# ---------------------------------------------------------------------------
# Exploratory analysis of the simulated survey, mirroring the Kaggle
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
dat <- read.csv(file.path(here, "data", "survey_sites.csv"),
                stringsAsFactors = FALSE)

cat("=== SHAPE ===\n"); cat(sprintf("%d sites x %d columns\n", nrow(dat), ncol(dat)))

cat("\n=== OUTCOME BALANCE (detected) ===\n")
print(round(prop.table(table(dat$detected)), 3))

cat("\n=== MISSINGNESS (% per column) ===\n")
miss <- sort(round(100 * colMeans(is.na(dat)), 2), decreasing = TRUE)
print(miss[miss > 0])
cat(sprintf("Rows with >=1 missing value: %.1f%%\n",
            100 * mean(apply(dat, 1, function(r) any(is.na(r))))))

cat("\n=== DETERMINISTIC LINK: effort vs passive_site ===\n")
# Passive (camera-only) sites should record zero active survey minutes.
print(dat %>% group_by(passive_site) %>%
        summarise(mean_effort = round(mean(total_effort_min, na.rm = TRUE), 1),
                  max_effort = max(total_effort_min, na.rm = TRUE), .groups = "drop"))

cat("\n=== DETECTION RATE BY LAND COVER ===\n")
print(dat %>% filter(!is.na(land_cover)) %>% group_by(land_cover) %>%
        summarise(detect_rate = round(mean(detected), 3), n = n(), .groups = "drop"))

cat("\n=== TRANSECT (GROUP) SIZE DISTRIBUTION ===\n")
ts <- dat %>% count(transect_id) %>% count(n, name = "n_transects")
print(as.data.frame(ts))

# --- Figures ---------------------------------------------------------------
# 1. Spatial pattern of detections (shows the autocorrelation we exploit later)
p1 <- ggplot(dat, aes(x, y, color = factor(detected))) +
  geom_point(alpha = 0.8, size = 1.6) +
  scale_color_manual(values = c("0" = "#C44E52", "1" = "#55A868"),
                     name = "Detected") +
  labs(title = "Detections across the landscape (note spatial clustering)") +
  theme_minimal()
ggsave(file.path(fig, "wildlife_spatial_detections.png"), p1,
       width = 6.5, height = 5.5, dpi = 120)

# 2. Effort vs detection
p2 <- ggplot(dat, aes(x = log1p(total_effort_min), fill = factor(detected))) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
  scale_fill_manual(values = c("0" = "#C44E52", "1" = "#55A868"),
                    name = "Detected") +
  labs(title = "Survey effort vs detection", x = "log1p(total effort, min)") +
  theme_minimal()
ggsave(file.path(fig, "wildlife_effort_vs_detection.png"), p2,
       width = 6.5, height = 4.5, dpi = 120)

cat("\nSaved figures to outputs/figures/: wildlife_spatial_detections.png, ",
    "wildlife_effort_vs_detection.png\n", sep = "")
