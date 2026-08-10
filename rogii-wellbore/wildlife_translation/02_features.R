# 02_features.R
# ---------------------------------------------------------------------------
# Feature engineering for the biologging layer-position task. Ecological mirror
# of the ROGII wellbore feature step. Predict the CHANGE in layer position from
# the last known point (delta = layer_pos - layer_ps), the analogue of dTVT.
#
# Two reference-based features do the real work (both inference-available):
#   ref_implied : match the tag temperature to the animal's OWN reference CTD
#                 profile -> a bias-calibrated layer estimate (type-well analogue)
#   self_implied: match temperature to the animal's own KNOWN-heel temp<->layer
#                 relationship (pre-PS self-log analogue)
# plus geometry (depth, along-track distance) which we expect to be useless.
#
# Provides engineer(tracks, reference) -> data.frame of toe-point features.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(dplyr))


engineer <- function(tracks, reference) {
  out <- list()
  for (aid in unique(tracks$animal_id)) {
    tr <- tracks[tracks$animal_id == aid, ]
    tr <- tr[order(tr$step), ]
    ref <- reference[reference$animal_id == aid, ]
    ps <- max(which(!is.na(tr$layer_input)))
    layer_ps <- tr$layer_input[ps]
    n <- nrow(tr)
    temp_s <- as.numeric(stats::filter(tr$temp, rep(1 / 5, 5), sides = 2))
    temp_s[is.na(temp_s)] <- tr$temp[is.na(temp_s)]
    toe <- (ps + 1):n

    # ref-implied layer: nearest ref_temp to the point's temp, near the last layer
    ro <- order(ref$ref_layer)
    rl <- ref$ref_layer[ro]; rt <- ref$ref_temp[ro]
    win <- rl >= layer_ps - 25 & rl <= layer_ps + 25
    rl_w <- rl[win]; rt_w <- rt[win]
    ref_impl <- sapply(temp_s[toe], function(v) rl_w[which.min(abs(rt_w - v))])

    # self-implied layer: invert the heel's own (temp -> layer) relationship
    heel_t <- temp_s[1:ps]; heel_l <- tr$layer_input[1:ps]
    ok <- !is.na(heel_t) & !is.na(heel_l)
    self_impl <- sapply(temp_s[toe], function(v) heel_l[ok][which.min(abs(heel_t[ok] - v))])

    depth_ps <- tr$depth[ps]
    dist <- cumsum(sqrt(c(0, diff(tr$x))^2 + c(0, diff(tr$y))^2))
    out[[aid]] <- data.frame(
      animal_id = aid, step = toe, layer_ps = layer_ps,
      target = tr$layer_pos[toe] - layer_ps,
      ref_impl = ref_impl - layer_ps,
      self_impl = self_impl - layer_ps,
      ref_impl_smooth = as.numeric(stats::filter(ref_impl - layer_ps, rep(1 / 15, 15), sides = 2)),
      temp = temp_s[toe],
      d_depth = tr$depth[toe] - depth_ps,            # geometry (expected useless)
      depth = tr$depth[toe],
      along_dist = dist[toe] - dist[ps],
      toe_frac = (toe - ps) / (n - ps),
      stringsAsFactors = FALSE)
    out[[aid]]$ref_impl_smooth[is.na(out[[aid]]$ref_impl_smooth)] <-
      out[[aid]]$ref_impl[is.na(out[[aid]]$ref_impl_smooth)]
  }
  bind_rows(out)
}
