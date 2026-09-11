# Robustness Scores Have No Sign — code and computed statistics

Everything needed to check the numbers in the paper, except the model predictions
themselves (725 MB of HDF5, see **Data** below).

The measurements are all **GT-free**: they compare a frozen model's clean-input
predictions against its corrupted-input predictions on the RoCo/RobustSpring
official test split, which has no public ground truth.

## Layout

```
scripts/     inference and statistics used to produce the paper's tables
analysis/    the self-contained analysis scripts for the two image-side studies,
             with the config each was run under
data/        every computed statistic the paper cites, as JSON
```

## Which file backs which table

| Paper | File |
|---|---|
| Table 1, `Delta` and `1px` columns | `data/OFFICIAL_SCORES.json` — returned by the Spring server, not recomputed by us |
| Table 1, `Ratio` column | `data/LOCAL_RECOMPUTE.json` |
| Table 1, `Ratio 95% CI` | `data/SCENE_CI.json` |
| Table 1, `g` column | `data/SIGNED_AXES.json` → `by_condition[c].directional_gain` |
| Table 2 (untouched pixels under spatter) | `data/UNTOUCHED_PIXELS.json` |
| Table 3 (cross-architecture) | `data/CROSS_ARCH_TABLE.json` |
| Table 4 (estimators, orthogonal and one-sided fractions) | `data/SIGNED_AXES.json` |
| Table 5 (resolution control) | `data/RESOLUTION_TABLE.json`, with the two run provenance files |
| Section 6.1–6.2 image statistics | `data/CORRUPTION_AXES_SUMMARY.json` |
| Zoom-blur global-scale test | `data/zoom_scale_v1.json` |
| Appendix C, training branches | not included; the branch checkpoints are not released |

## Definitions, in one place

For a record, `f_clean` and `f_corrupt` are the two predicted flow fields.

- **ratio** — `sum(|f_corrupt|) / sum(|f_clean|)`, pooled as a ratio of pixel sums
  over every record in a condition, not as a mean of per-record ratios. The
  magnitude is `sqrt(u^2+v^2)` per pixel, so opposing motions do not cancel.
- **along-clean gain `g`** — `sum(<f_clean, f_corrupt>) / sum(|f_clean|^2)`, the
  least-squares scaling of the clean prediction that best explains the corrupted
  one. Implemented as `directional_gain` in `scripts/official_delta_metrics_v1.py`.
- **orthogonal fraction** — `sum(|f_corrupt - g*f_clean|^2) / sum(|f_corrupt|^2)`.
- **collapse / inflation fraction** — among points whose *clean predicted*
  magnitude exceeds 1 px, the share whose corrupted predicted magnitude falls
  below half of it, or rises above twice it. There is no test ground truth, so
  neither counts genuinely moving points.

Confidence intervals are a scene-block bootstrap: all 10 test sequences resampled
as whole blocks, 10,000 draws, seed 20260911 (20260910 for Table 2). The 10 blocks
carry very unequal shares of the denominator, so the effective block count is
about 6.6; read the intervals as sensitivity to scene composition, not as
significance tests.

## Reproducing

The statistics scripts need `numpy`, `h5py` and `scipy`. The inference scripts
additionally need `torch`, `ptlflow` and the published Spring checkpoints for
WAFT, DPFlow and MEMFOF, which we do not redistribute.

```bash
# Table 1 ratio and CI, from the submitted prediction file
python scripts/public_waft_scene_ci.py

# Table 1 g, Table 5 estimators and fractions
python scripts/signed_axes_v1.py

# Table 3, one architecture per invocation
CUDA_VISIBLE_DEVICES=0 python scripts/cross_arch_ratio_v1.py --model dpflow
python scripts/summarize_cross_arch_v1.py

# Resolution control
CUDA_VISIBLE_DEVICES=0 python scripts/waft_resolution_v1.py
python scripts/summarize_waft_resolution_v1.py
```

Paths in the scripts are absolute to the machine the work was done on; adjust
`ROOT` before running.

## Data

The paper's primary numbers are computed from our own submission file:
predictions of the public `waft_dav2_a2` Spring checkpoint on all 21 conditions
of the official test split, 725 MB as HDF5. We do not host it here.

The corrupted input images belong to the RobustSpring benchmark and are
distributed by its organisers under their own terms; we redistribute none of
them. The official `Delta` and `1px` scores in Table 1 are the values the Spring
evaluation server returned for our submission.

## Licence

Code in `scripts/` and `analysis/` is MIT. The JSON files in `data/` are
measurements, released under CC0.
