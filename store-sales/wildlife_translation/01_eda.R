# 01_eda.R
# ---------------------------------------------------------------------------
# Exploratory analysis of the monitoring series, mirroring Store Sales
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
dat <- read.csv(file.path(here, "data", "monitoring_counts.csv"),
                stringsAsFactors = FALSE)
dat$date <- as.Date(dat$date)

skew <- function(x) { m <- mean(x); mean((x - m)^3) / sd(x)^3 }
cat("=== TARGET (count) ===\n")
cat(sprintf("median %d | max %d | zeros %.1f%% | skew raw %.2f | skew log1p %.2f\n",
            median(dat$count), max(dat$count), 100 * mean(dat$count == 0),
            skew(dat$count), skew(log1p(dat$count))))
cat("-> right-skewed count with zeros; model log1p(count) (matches RMSLE).\n")

cat("\n=== SERIES ===\n")
cat(sprintf("%d sites x %d species = %d series over %d months\n",
            length(unique(dat$site)), length(unique(dat$species)),
            length(unique(dat$site)) * length(unique(dat$species)),
            length(unique(dat$date))))

cat("\n=== SEASONALITY: mean count by month ===\n")
print(dat %>% group_by(month) %>% summarise(mean_count = round(mean(count), 1),
                                            .groups = "drop"), n = 12)

cat("\n=== TREND: mean count by year ===\n")
print(dat %>% group_by(year) %>% summarise(mean_count = round(mean(count), 2),
                                           .groups = "drop"))

# Figures
tot <- dat %>% group_by(date) %>% summarise(total = sum(count), .groups = "drop")
p1 <- ggplot(tot, aes(date, total)) + geom_line(color = "#4C72B0") +
  labs(title = "Total monthly count across all series (seasonal + declining)",
       y = "count") + theme_minimal()
ggsave(file.path(fig, "monitoring_trend.png"), p1, width = 10, height = 3.5, dpi = 120)

seas <- dat %>% group_by(month) %>% summarise(mean_count = mean(count), .groups = "drop")
p2 <- ggplot(seas, aes(factor(month), mean_count)) +
  geom_col(fill = "#55A868") + labs(title = "Seasonality: mean count by month",
                                    x = "month", y = "mean count") + theme_minimal()
ggsave(file.path(fig, "monitoring_seasonality.png"), p2, width = 7, height = 4, dpi = 120)

cat("\nSaved outputs/figures/monitoring_trend.png, monitoring_seasonality.png\n")
