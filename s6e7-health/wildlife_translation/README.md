# Wildlife Translation: From Health-Risk Scoring to Body-Condition Classification

Ecological translation of the Kaggle S6E7 task ("Predicting Student Health Risk", an
imbalanced 3-class problem scored on balanced accuracy). The competition is really:
**classify individuals into an imbalanced set of condition/risk classes, where the rare
class is the one you most need to catch.**

The clean wildlife analogue is **individual body-condition scoring**: classify a captured
animal as `poor`, `fair`, or `good` condition from bio-logger activity, capture
morphometrics, and physiology. Poor-condition animals are rare but are the management
priority, so balanced accuracy (macro recall) is the right metric, exactly as in the
competition. Everything runs on a **simulated** dataset (`00_simulate_condition.R`), so it
is reproducible and uses no field data.

> A general-technique demo on **simulated** data: it does not contain the competition
> solution. The source competition has closed; the reusable pattern is also collected in
> the standalone wildlife-modeling cookbook.

## Why this translation is faithful

| Kaggle S6E7 | Body-condition analogue |
|-------------|-------------------------|
| health_condition {at-risk, unhealthy, fit} | condition class {poor, fair, good} |
| heavy class imbalance, rare priority class | poor-condition animals rare but the priority |
| balanced accuracy (macro recall) | same: minority recall counts as much as majority |
| step_count / activity_level / calories | accelerometer activity, ODBA, daily displacement |
| heart_rate / sleep / bmi | bio-logger physiology, rest bouts, body mass |
| stress_level | glucocorticoids (cortisol) |
| class-weighting + decision-rule tuning | same lever, to lift poor-condition recall |

Two things real condition data adds that the synthetic Kaggle data lacked, and that the
sim builds in on purpose:

1. **Group structure.** Each animal is captured several times, with a persistent
   individual bias (tag/animal effect). This creates the leave-whole-animal-out CV crux.
2. **Multiple sensor modalities** carrying complementary signal, which is where model
   diversity actually pays off (the positive counterpart to S6E7's flat blend).

## The scripts

| Script | Purpose |
|--------|---------|
| `00_simulate_condition.R` | build the reproducible tagged-population dataset |
| `01_eda.R` | class imbalance + the complementary structure of the two modalities |
| `02_model_condition.R` | class-weighted classifier, the group-CV crux, the multi-modal blend |

Run in order:

```bash
Rscript 00_simulate_condition.R
Rscript 01_eda.R
Rscript 02_model_condition.R
```

## The transferable lessons (with numbers from this run)

**1. Balanced accuracy makes the rare class matter.** Predicting the majority class
(`good`) everywhere scores **0.333**. A class-weighted random forest with an OOF-tuned
decision rule reaches **0.729** under honest CV, with poor-condition recall ~0.78. As in
S6E7, plain accuracy would have rewarded ignoring the animals you care about most.

**2. THE CRUX: honest validation leaves WHOLE ANIMALS out.** The same model scores
**0.752 under random-row CV but 0.729 under group-by-animal CV** (optimistic by 0.022):
random folds scatter an animal's repeat captures across train and val, leaking its
persistent individual bias. This is the direct analogue of group-by-well CV in ROGII and
group-by-site CV in occupancy work. Always leave whole individuals (and ideally whole
sites) out.

**3. Multi-modal diversity PAYS OFF here (unlike S6E7).** Under honest group CV:
morphometrics-only **0.686**, movement/physiology-only **0.634**, and their **blend
0.735 (+0.049 over the best single modality)**. The two modalities agree on only 62% of
labels (error overlap Jaccard 0.26, oracle-of-two 0.84): they are decorrelated AND each
strong, because each reveals condition in animals the other misses (a recently-fed animal
looks fine on mass but not on cortisol; a sedentary-but-healthy animal looks inactive but
fine on morphometrics). This is the POSITIVE counterpart to the competition, where the
diverse members were decorrelated-but-weak and the residual was irreducible label noise,
so the blend was flat (0.9497 single = blend = stack). **The rule the two runs teach
together: a blend helps only when members are decorrelated AND each individually strong;
otherwise you are at an information ceiling and should stop.**

---

## Extracted pattern for the wildlife cookbook

### Pattern: Imbalanced multiclass condition/health scoring from multi-sensor individual data
- **Ecological problem:** classify individuals into an imbalanced set of condition,
  health, or disease classes (poor/fair/good; susceptible/exposed/infected) from
  bio-logger, morphometric, and physiological features, where the at-risk class is rare
  but the management priority.
- **Metric:** balanced accuracy (macro recall), so minority recall counts as much as
  majority. Accuracy would reward ignoring the rare class. Use class weighting plus an
  OOF-tuned per-class decision rule to convert probabilities to labels.
- **Validation (critical):** leave WHOLE individuals out (and ideally whole sites). Random
  folds leak the persistent per-animal bias across an animal's repeat captures and are
  optimistic (here 0.752 random vs 0.729 honest).
- **Ensemble diagnostic:** before stacking models, measure error overlap and the
  oracle-of-two. A blend helps only if members are decorrelated AND each strong. Distinct
  sensor MODALITIES are a reliable source of that (here +0.049); swapping learner families
  on one feature set often is not (S6E7 blend was flat at an information ceiling).
- **Key code:** `02_model_condition.R` (group CV crux + modality blend).
- **Status:** ready to extract (after the S6E7 deadline; currently local-only).
