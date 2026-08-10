# 01_eda.R
# ---------------------------------------------------------------------------
# Exploratory analysis of the simulated biologging tracks. Prints findings and
# saves figures. Run:  Rscript 01_eda.R
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(dplyr); library(ggplot2) })

here <- local({
  a <- commandArgs(FALSE); f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
fig <- file.path(here, "outputs", "figures"); dir.create(fig, showWarnings = FALSE, recursive = TRUE)
tr <- read.csv(file.path(here, "data", "tracks.csv"), stringsAsFactors = FALSE)

cat("=== SHAPE ===\n")
cat(sprintf("%d track steps across %d animals\n", nrow(tr), length(unique(tr$animal_id))))

cat("\n=== the key structure: depth swings, layer holds ===\n")
rng <- tr %>% group_by(animal_id) %>%
  summarise(depth_range = max(depth) - min(depth),
            layer_range = max(layer_pos) - min(layer_pos), .groups = "drop")
cat(sprintf("mean within-track depth range: %.1f m\n", mean(rng$depth_range)))
cat(sprintf("mean within-track layer range: %.1f m\n", mean(rng$layer_range)))
cat(sprintf("corr(layer_pos, depth) = %.3f  (geometry uninformative)\n", cor(tr$layer_pos, tr$depth)))
cat(sprintf("corr(layer_pos, temp)  = %.3f  (the sensor carries the signal)\n", cor(tr$layer_pos, tr$temp)))

# example track: depth vs layer_pos over the track
ex <- tr[tr$animal_id == tr$animal_id[1], ]
p1 <- ggplot(ex, aes(step)) +
  geom_line(aes(y = depth, colour = "depth (raw)")) +
  geom_line(aes(y = 80 + layer_pos, colour = "layer position (+80)")) +
  scale_colour_manual(values = c("depth (raw)" = "#C44E52", "layer position (+80)" = "#55A868"), name = NULL) +
  labs(title = "One animal: raw depth swings, layer position is held",
       y = "metres", x = "track step") + theme_minimal()
ggsave(file.path(fig, "example_track.png"), p1, width = 9, height = 3.6, dpi = 120)

# temperature vs layer position (the water-mass profile the sensor rides)
p2 <- ggplot(tr[sample(nrow(tr), 4000), ], aes(temp, layer_pos)) +
  geom_point(alpha = 0.2, size = 0.6, colour = "#4C72B0") +
  labs(title = "Sensor (temperature) vs layer position (blurred by per-animal bias)",
       x = "temperature", y = "layer position (m rel. thermocline)") + theme_minimal()
ggsave(file.path(fig, "temp_vs_layer.png"), p2, width = 6.5, height = 4.5, dpi = 120)

cat("\nSaved outputs/figures/example_track.png, temp_vs_layer.png\n")
