# 00_simulate_biologging.R
# ---------------------------------------------------------------------------
# Wildlife translation of the ROGII wellbore geology task. The wellbore problem
# is: infer position within a LAYERED medium along a 1-D path, given the position
# for the first stretch, a sensor log along the path, a reference vertical
# profile, and neighbouring paths that share the structure.
#
# Ecological analogue: a tagged diving animal (seal/tuna) moving through a
# thermally layered water column whose thermocline dips/shoals across space.
#
#   wellbore trajectory (MD,X,Y,Z)      -> the animal's dive track
#   gamma-ray log along the well        -> the tag's temperature sensor
#   TVT (position in formation column)  -> layer_pos (depth relative to thermocline)
#   TVT_input known for the heel        -> layer position known for the track start
#   type well (GR vs TVT)               -> reference CTD profile (temp vs layer_pos)
#   neighbouring wells share dip         -> nearby animals share the thermocline
#   predict the masked toe               -> predict layer_pos for the rest of the track
#
# Key structural feature (as in ROGII): the animal HOLDS a foraging layer, so its
# raw depth Z swings a lot as the thermocline dips, while layer_pos barely moves.
# So geometry alone fails and only the SENSOR (via profile matching) locates it.
#
# Output: data/tracks.csv (one row per track step) + data/reference_<id>.csv (CTD).
# ---------------------------------------------------------------------------
set.seed(2026)
suppressPackageStartupMessages(library(dplyr))

get_script_dir <- function() {
  args <- commandArgs(FALSE); fa <- grep("^--file=", args, value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa))))
  getwd()
}
out <- file.path(get_script_dir(), "data")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

n_animals <- 60
DOMAIN <- 100

# --- Thermocline depth field D(x,y): dips strongly on a short spatial scale --
set.seed(11)
nh <- 8
hcx <- runif(nh, 0, DOMAIN); hcy <- runif(nh, 0, DOMAIN)
hs <- runif(nh, 10, 16); ha <- runif(nh, -35, 35)
thermocline_depth <- function(x, y) {
  # strong regional dip (the thermocline slopes across the region) + local bumps
  80 + 0.55 * (x - 50) - 0.40 * (y - 50) +
    rowSums(sapply(seq_len(nh), function(k)
      ha[k] * exp(-((x - hcx[k])^2 + (y - hcy[k])^2) / (2 * hs[k]^2))))
}
set.seed(2026)

# --- Canonical water-mass profile: temperature as a function of layer_pos ----
# Warm above the thermocline (layer_pos < 0), cold below, sharp gradient at 0,
# plus a mild non-monotonic subsurface bump so single-value matching is a bit
# ambiguous (mirrors GR ambiguity).
temp_profile <- function(lp) {
  18 - 12 / (1 + exp(-lp / 6)) + 1.5 * exp(-((lp + 12)^2) / (2 * 5^2))
}

rows <- list(); refs <- list(); k <- 1
for (a in seq_len(n_animals)) {
  T <- sample(200:340, 1)                     # track length (steps)
  x0 <- runif(1, 10, DOMAIN - 10); y0 <- runif(1, 10, DOMAIN - 10)
  # a directed traverse across the dipping thermocline (so depth swings a lot)
  dist <- runif(1, 35, 60); ang <- runif(1, 0, 2 * pi)
  ex <- min(max(x0 + dist * cos(ang), 5), DOMAIN - 5)
  ey <- min(max(y0 + dist * sin(ang), 5), DOMAIN - 5)
  frac0 <- seq(0, 1, length.out = T)
  x <- x0 + (ex - x0) * frac0 + as.numeric(arima.sim(list(ar = 0.9), T, sd = 0.6))
  y <- y0 + (ey - y0) * frac0 + as.numeric(arima.sim(list(ar = 0.9), T, sd = 0.6))
  x <- pmin(pmax(x, 0), DOMAIN); y <- pmin(pmax(y, 0), DOMAIN)

  # the animal HOLDS a foraging layer just above the thermocline (tight wiggle)
  target_layer <- runif(1, -12, -2)
  lp <- target_layer + as.numeric(arima.sim(list(ar = 0.9), T, sd = 0.6))
  # occasional excursions (brief dives out of the layer) for a few animals
  if (runif(1) < 0.3) {
    s <- sample(seq_len(T), 1); w <- 20
    idx <- max(1, s - w):min(T, s + w)
    lp[idx] <- lp[idx] + rnorm(1, 0, 4) * exp(-((idx - s)^2) / (2 * (w / 2)^2))
  }
  D <- thermocline_depth(x, y)
  z <- D + lp                                 # actual depth = thermocline + layer_pos
  # per-animal tag/water-mass bias: a global sensor->layer map is blurred, but
  # the animal's OWN reference profile + known heel calibrate it (why group CV matters)
  animal_bias <- rnorm(1, 0, 1.5)
  temp <- temp_profile(lp) + animal_bias + rnorm(T, 0, 0.4)

  ps <- round(runif(1, 0.35, 0.45) * T)       # known "heel" fraction
  layer_input <- lp; layer_input[(ps + 1):T] <- NA

  rows[[a]] <- data.frame(
    animal_id = sprintf("A%03d", a), step = seq_len(T),
    x = round(x, 2), y = round(y, 2), depth = round(z, 2),
    temp = round(temp, 3), layer_pos = round(lp, 3),
    layer_input = round(layer_input, 3), stringsAsFactors = FALSE)

  # reference CTD profile for this animal (its own calibrated water mass)
  grid <- seq(-40, 20, by = 0.5)
  refs[[a]] <- data.frame(animal_id = sprintf("A%03d", a),
                          ref_layer = grid,
                          ref_temp = round(temp_profile(grid) + animal_bias +
                                             rnorm(length(grid), 0, 0.25), 3))
  k <- k + 1
}
tracks <- bind_rows(rows); reference <- bind_rows(refs)
write.csv(tracks, file.path(out, "tracks.csv"), row.names = FALSE)
write.csv(reference, file.path(out, "reference_profiles.csv"), row.names = FALSE)

skew <- function(x) { m <- mean(x); mean((x - m)^3) / sd(x)^3 }
cat(sprintf("Wrote %d track steps across %d animals\n", nrow(tracks), n_animals))
cat(sprintf("depth (Z) range within a track: mean %.1f m\n",
            mean(tapply(tracks$depth, tracks$animal_id, function(v) max(v) - min(v)))))
cat(sprintf("layer_pos range within a track: mean %.1f m  (animal holds its layer)\n",
            mean(tapply(tracks$layer_pos, tracks$animal_id, function(v) max(v) - min(v)))))
