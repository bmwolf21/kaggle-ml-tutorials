# Wildlife Translation: From Wellbore Geosteering to Biologging

Ecological translation of the ROGII Wellbore Geology Prediction task. The wellbore
problem is really: **infer position within a layered medium along a 1-D path**,
given the position for the first stretch, a sensor log along the path, a reference
vertical profile, and neighbouring paths that share the structure.

The clean wildlife analogue is a **tagged diving animal** (seal, tuna, penguin)
moving through a thermally layered water column whose thermocline dips across
space. Everything runs on a **simulated** dataset (`00_simulate_biologging.R`), so
it is reproducible and uses no field data.

> A general-technique demo on **simulated** data: it does not contain the competition
> solution. The source competition has closed; the reusable pattern is also collected in
> the standalone wildlife-modeling cookbook.

## Why this translation is faithful

| ROGII wellbore | Biologging analogue |
|----------------|---------------------|
| horizontal well trajectory (MD, X, Y, Z) | the animal's dive track |
| gamma-ray log along the well | the tag's temperature sensor along the track |
| `TVT` (position in the formation column) | `layer_pos` (depth relative to the thermocline) |
| `TVT_input` known for the heel | layer position known for the start of the track |
| type well (GR vs TVT) | the animal's reference CTD profile (temp vs layer_pos) |
| neighbouring wells share dip | nearby animals share the (dipping) thermocline |
| predict the masked toe | predict layer position for the rest of the track |
| group-by-well CV (the leakage crux) | group-by-individual CV (leave whole animals out) |

The defining feature carries over exactly: the animal **holds a foraging layer**,
so its raw depth swings ~30 m as the thermocline dips while `layer_pos` moves only
~7 m. Geometry is therefore useless; only the sensor, matched to a reference
profile, locates the animal.

## The scripts

| Wildlife | Purpose |
|----------|---------|
| `00_simulate_biologging.R` | build the reproducible tagged-animal dataset |
| `01_eda.R` | depth-vs-layer structure, sensor-vs-layer, an example track |
| `02_features.R` | reference/self profile matching + geometry features |
| `03_layer_model.R` | baselines, the group-CV crux, ranger model, neighbour prior |

Run in order:

```bash
Rscript 00_simulate_biologging.R
Rscript 01_eda.R
Rscript 03_layer_model.R   # sources 02_features.R
```

## The transferable lessons (with numbers from this run)

**1. Geometry is uninformative; the sensor is the signal.** `corr(layer_pos, depth)
= 0.04`; a "delta = change in depth" model scores 12.55 m RMSE, far worse than
just holding the last known layer (1.91 m). Only the temperature sensor, matched
to the reference profile, helps. This mirrors ROGII, where `dZ` was useless and GR
was the only signal.

**2. THE CRUX - honest validation must leave whole individuals out.** A raw-sensor
model looks excellent under **random-row CV (0.71 m)** but collapses to **1.92 m
under group-by-animal CV**: random folds put other rows of the same animal in
training, leaking that animal's tag bias / held layer. The honest,
own-reference-calibrated model reaches **0.86 m under group CV**, beating both.
This is the exact analogue of **group-by-well CV** in ROGII, the single most
important guard against the competition's leakage trap.

**3. Per-unit calibration beats a global map.** Because each tag has a small bias,
a global sensor->layer map only explains about half the variance. Matching each
animal's temperature to **its own reference profile** (and its own known-heel
temp<->layer relationship) calibrates that bias, which is why the honest model
generalises to held-out animals. In ROGII this is the type well and the lateral's
own pre-PS log.

**4. Neighbour prior (honest note):** nearby animals share the thermocline, so
their layer profiles are a plausible prior, but here it barely moved the score
(0.87 vs 0.86) because the own-reference calibration already does the work. In
ROGII the offset-well prior helped more, because that pipeline leaned on a geometry
anchor that the neighbours could correct. Same idea, different payoff.

---

## Extracted pattern for the wildlife cookbook

### Pattern: Sequence-position inference along a path through a layered medium
- **Ecological problem:** infer an animal's (or a probe's) position within a
  layered environment along a 1-D path - depth relative to the thermocline, soil
  horizon along a core, isopycnal along a glider transect - when position is known
  only for the first stretch.
- **Technique:** match a path-borne sensor to a reference vertical profile (and to
  the path's own known segment) to invert sensor -> layer position; anchor on the
  last known position; predict the rest. Geometry alone fails when the subject
  holds a layer while the layer surface dips.
- **Validation (critical):** leave WHOLE individuals/paths out (group CV). Random
  splits leak per-individual calibration and are dangerously optimistic (here 0.71
  vs 0.86-1.92 m honest).
- **Key code:** `02_features.R` (profile matching), `03_layer_model.R` (group CV).
- **Status:** ready to extract (after the competition; currently local-only).
