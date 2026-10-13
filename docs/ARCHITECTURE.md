# Architecture

This document describes how `wds-crossmatch` is organized internally: the
pipeline stages, the statistical model, the assignment step and the main
design decisions. For usage, see [`USER_GUIDE.md`](USER_GUIDE.md).

---

## 1. Repository layout

| Path | Content |
| :--- | :--- |
| `crossmatch.py` | Component-level WDS → Gaia DR3 crossmatch (main pipeline) |
| `summary.py` | Pair-level combination and physical diagnostics |
| `example_data/` | Sample WDS inputs (64 and 3,409 components) |
| `example_outputs/` | Reference outputs for the sample data |
| `outputs/` | Results of your runs (not versioned) |
| `cache/` | Cached Gaia query results (not versioned) |
| `docs/` | Documentation |

---

## 2. Pipeline stages

`crossmatch.py` runs these stages in order (see `main()`):

| # | Stage | Function |
| :--- | :--- | :--- |
| 1 | Parse and validate the WDS component file | `parse_wds_components` |
| 2 | Propagate coordinates from epoch 2000.0 to 2016.0 | `propagate_coordinates` |
| 3 | Query the 15-arcsec RAW model sample | `query_gaia_cached` |
| 4 | Query the 15-arcsec displaced control sample | `query_gaia_cached` |
| 5 | Query the 5-arcsec candidate catalog | `query_gaia_cached` |
| 6 | Check quasi-colors against the frozen grid | `validate_quasicolor_support` |
| 7 | Build the empirical Bayesian models | `build_empirical_bayesian_models` |
| 8 | Save model diagnostics | `save_model_diagnostics` |
| 9 | Score every candidate | `score_candidates` |
| 10 | Solve the global one-to-one assignment | `optimize_global_assignments` |
| 11 | Write the final catalog | `main` |

---

## 3. Input parsing and coordinate propagation

- The WDS file is fixed-width. Each record must have at least 130 characters
  and valid coordinates (`HHMMSS.ss`, signed `DDMMSS.s`); malformed records
  stop the run. A seconds field of exactly 60.0 is accepted as a normal carry.
- Missing magnitudes are stored as `1234.0` and missing proper motions as `0`.
- Secondary components (`**s`) use their own proper motion when available and
  inherit the primary's otherwise. WDS records with a `P` note are converted
  from arcsec per century to mas/yr.
- Propagation applies the proper-motion vector in the tangent plane to a unit
  direction vector and renormalizes it. This avoids the `cos(δ)` factor and
  manual clipping near the poles. No distance, parallax or radial velocity is
  used.

---

## 4. Gaia retrieval

Three queries are sent through `TAP_UPLOAD`, in batches of 10,000 targets,
with up to three retries per batch:

| Query | Radius | Gaia columns | Purpose |
| :--- | :--- | :--- | :--- |
| RAW model | 15 arcsec | `source_id`, position, G | Calibrate the model (true + background sources) |
| DISPLACED model | 15 arcsec | `source_id`, position, G | Calibrate the background (random field) |
| RAW catalog | 5 arcsec | All columns | Final candidates |

- The displaced field is shifted 2 arcmin toward the celestial equator, using a
  spherical offset, as in Medan et al. (2021).
- Two services are supported: ARI Heidelberg (default) and the ESA Gaia
  Archive. On ARI, the model queries use the lighter `gaia_source_lite` table.
- `query_gaia_cached` wraps the queries with an on-disk cache in VOTable
  (`binary2`) format, which preserves the column types and missing values of a
  live query.

---

## 5. Empirical Bayesian model

### 5.1. Strata

The model is calibrated separately in 18 strata: 3 bins of absolute Galactic
latitude `|b|` × 6 bins of WDS magnitude.

| `|b|` bins (deg) | WDS magnitude bins |
| :--- | :--- |
| [0, 19.5), [19.5, 41.8), [41.8, 90.1) | < 10, [10, 11), [11, 12), [12, 13), [13, 15.5), ≥ 15.5 |

### 5.2. Model variables

- Angular separation `θ`, in 0.2-arcsec bins up to 15 arcsec.
- Quasi-color `q = G_Gaia − m_WDS`, on a frozen grid of [−19, 22] mag with
  0.25-mag bins. A run stops if any modeled `q` falls outside this grid.

### 5.3. Construction of one 2-D model

1. **Background model (H0).** The displaced sample is histogrammed in `(θ, q)`.
   Its angular marginal is fitted with a straight line, excluding the outer
   1 arcsec of the cone. Its quasi-color marginal is fitted with a mixture of
   2 to 10 Gaussians, selected by BIC. The product of both marginals is scaled
   by a constant `C` fitted to the occupied cells.
2. **Scale factor S.** The background is scaled by the factor at the elbow of
   the negative-residual curve, found with a broken-line fit.
3. **Smoothing.** Both histograms are smoothed with
   `percentile_filter(percentile=60, size=5)`.
4. **Signal (H1).** The scaled background is subtracted from the RAW histogram;
   negative cells are set to zero.
5. **Priors and posterior.** `P1` is the signal fraction and `P0 = 1 − P1`.
   The posterior in each cell is `f1·P1 / (f1·P1 + f0·P0)`. Cells with no
   support under either hypothesis are left undefined.

### 5.4. Fallback models

| Model | Used when | Construction |
| :--- | :--- | :--- |
| 2-D | WDS magnitude and Gaia G known | Section 5.3 |
| θ-only | WDS magnitude known, Gaia G missing | Same procedure in `θ` only, trained without requiring Gaia G |
| `|b|`-only | WDS magnitude missing | Combines the six θ-only magnitude strata of that latitude bin; disabled unless all six are valid |

### 5.5. Deterministic Gaussian fits

For each number of Gaussians (2 to 10), seven fixed starting points are tried;
no random seed is used. Within each count the fit with the lowest weighted cost
is kept, and BIC selects the count. Every attempt is recorded in
`<model-summary>.fit_audit.csv`.

---

## 6. Candidate scoring

`score_candidates` assigns each candidate within 5 arcsec the posterior of the
cell it falls into, using the 2-D, θ-only or `|b|`-only model as described in
section 5.4. It also stores the log-odds, clipped to avoid infinite values.

---

## 7. Global one-to-one assignment

`optimize_global_assignments` follows the assignment formalism of Budavári &
Basu (2016), applied to the empirical posteriors:

1. **Graph.** Only edges with posterior ≥ 0.5 are kept. Weaker edges can never
   improve the solution, because every node can stay unmatched at zero cost.
2. **Components.** The graph is split into independent connected components.
3. **Hungarian algorithm.** Each component is solved with
   `scipy.optimize.linear_sum_assignment`, using `−log-odds` as cost and dummy
   rows and columns for the unmatched option.
4. **Margins.** For each assigned edge, the solver is re-run with that edge
   forbidden; the objective loss is reported as `assigned_edge_margin`.

Gaia `source_id` values are kept as nullable 64-bit integers throughout the
output assembly, because converting them to floating point would round many
19-digit identifiers.

---

## 8. Adaptations relative to Medan et al. (2021)

- WDS is the reference catalog, so strata use the WDS magnitude rather than
  Gaia G.
- The angular edge trim is a fixed 1 arcsec, instead of the grid-specific rule
  of the original code.
- Gaussian centers are constrained to the occupied quasi-color range of each
  stratum.
- Coordinates are propagated with a unit-vector method instead of first-order
  spherical equations.
- Negative angular-model predictions are floored at zero.

---

## 9. Performance

On the extended sample, profiling shows that about 92% of the runtime is spent
in `build_empirical_bayesian_models`, mostly inside `scipy.optimize.least_squares`
fitting the Gaussian mixtures. Candidate scoring and assignment together take a
few seconds. The disk cache removes the network wait on repeated runs.