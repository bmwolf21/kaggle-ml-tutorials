# 01_eda.R
# ---------------------------------------------------------------------------
# Exploratory look at the simulated condition data: class imbalance and the
# COMPLEMENTARY structure of the two sensor modalities (why the blend will help).
# Run: Rscript 01_eda.R
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ggplot2); library(tidyr) })

here <- local({
  a <- commandArgs(FALSE); f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
fig <- file.path(here, "outputs", "figures"); dir.create(fig, showWarnings = FALSE, recursive = TRUE)
d <- read.csv(file.path(here, "data", "captures.csv"), stringsAsFactors = FALSE)
d$condition <- factor(d$condition, levels = c("poor", "fair", "good"))

cat("=== class balance (events) ===\n"); print(round(prop.table(table(d$condition)), 4))
cat(sprintf("\n%d events across %d animals; poor-condition is the rare priority class.\n",
            nrow(d), length(unique(d$animal_id))))

# each modality separates condition only PARTLY, and on different animals
num <- function(x) suppressWarnings(as.numeric(x))
cat("\n=== how well each single feature separates condition (eta-like via anova F) ===\n")
for (v in c("body_mass", "kidney_fat", "odba", "cortisol")) {
  f <- summary(aov(num(d[[v]]) ~ d$condition))[[1]][["F value"]][1]
  cat(sprintf("  %-12s F = %6.1f\n", v, f))
}

# morphometric axis vs movement axis, coloured by condition: neither alone is clean,
# together they separate the classes -> the visual case for the multi-modal blend
d2 <- d %>% mutate(morph = scale(num(body_mass)) + scale(num(kidney_fat)),
                   move = -(scale(num(cortisol))) + scale(num(odba)))
p1 <- ggplot(d2, aes(morph, move, colour = condition)) +
  geom_point(alpha = 0.4, size = 0.7) +
  scale_colour_manual(values = c(poor = "#C44E52", fair = "#DD8452", good = "#55A868")) +
  labs(title = "Two modalities are each partial; together they separate condition",
       x = "morphometric axis (mass + kidney fat)", y = "movement/physiology axis (activity - cortisol)") +
  theme_minimal()
ggsave(file.path(fig, "modality_axes.png"), p1, width = 7, height = 5, dpi = 120)

pl <- d %>% select(condition, body_mass, cortisol) %>%
  pivot_longer(-condition) %>% mutate(value = num(value))
p2 <- ggplot(pl, aes(condition, value, fill = condition)) +
  geom_boxplot(outlier.size = 0.4) + facet_wrap(~name, scales = "free_y") +
  scale_fill_manual(values = c(poor = "#C44E52", fair = "#DD8452", good = "#55A868")) +
  labs(title = "One feature per modality: overlapping, so no single sensor suffices",
       x = NULL, y = NULL) + theme_minimal() + theme(legend.position = "none")
ggsave(file.path(fig, "modality_boxplots.png"), p2, width = 7.5, height = 4, dpi = 120)

cat("\nSaved outputs/figures/modality_axes.png, modality_boxplots.png\n")
