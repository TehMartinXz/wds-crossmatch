# API Reference

Reference for the functions in `crossmatch.py` and `summary.py`. Signatures are
taken from the source code. For the statistical method behind them, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

Functions whose names start with `_` are internal helpers and are not listed.

---

## 1. Using the pipeline from Python

Both scripts can be imported. `crossmatch.main` accepts the same arguments as
the command line:

```python
import crossmatch

crossmatch.main([
    "--components", "example_data/wds_components.summ.extended.txt",
    "--backend", "ari",
])
```

```python
import summary

df = summary.generate_summary_with_gaia(
    summary_path="./data/wdsweb_summ2.txt",
    crossmatch_path="./outputs/components_crossmatch_bayesian.csv",
    output_path="./outputs/wds_summary_with_gaia.csv",
    elbadry_path=None,
)
```

---

## 2. `crossmatch.py`

### 2.1. Configuration constants

| Constant | Value | Meaning |
| :--- | :--- | :--- |
| `DEFAULT_GAIA_BACKEND` | `"ari"` | Default Gaia TAP service |
| `ARI_BATCH_SIZE`, `ESA_BATCH_SIZE` | `10000` | Targets per `TAP_UPLOAD` batch |
| `MODEL_RADIUS_ARCSEC` | `15.0` | Radius of the calibration queries |
| `ASSIGNMENT_RADIUS_ARCSEC` | `5.0` | Radius of final candidate eligibility |
| `displacement_deg` | `2/60` | Offset of the displaced control field (2 arcmin) |
| `b_bins` | 3 bins | Absolute Galactic latitude strata |
| `wds_mag_strata` | 6 bins | WDS magnitude strata |
| `DIST_BIN_WIDTH` | `0.20` | Angular bin width (arcsec) |
| `DMAG_MIN`, `DMAG_MAX`, `DMAG_BIN_WIDTH` | `-19`, `22`, `0.25` | Frozen quasi-color grid (mag) |
| `ANGULAR_H0_EDGE_TRIM_ARCSEC` | `1.0` | Outer cone edge excluded from the angular fit |
| `FIT_MAX_NFEV` | `10000` | Optimizer evaluation budget per Gaussian fit |
| `DETERMINISTIC_START_SPECS` | 7 entries | Fixed starting points for the Gaussian fits |
| `CACHE_DIR` | `"./cache"` | Gaia query cache directory |
| `FIT_AUDIT` | list | Global record of every Gaussian fit attempt; cleared by `main` |

### 2.2. Input and coordinates

#### `parse_wds_components(components_file)`
Parses a fixed-width WDS component file.
- **Returns:** `pandas.DataFrame`, one row per component, with WDS fields, parsed coordinates and an `upload_id` column.
- **Raises:** `ValueError` if the file is empty, a record is shorter than 130 characters, or a coordinate is malformed or out of range.

#### `propagate_coordinates(wds_df)`
Propagates positions from epoch 2000.0 to 2016.0 using the WDS proper motions.
- **Modifies `wds_df` in place:** adds `wds_mag`, `ra_j2016`, `dec_j2016` and `gal_b`.
- **Returns:** `None`.

#### `get_displaced_coordinates(ra_deg, dec_deg)`
Shifts positions 2 arcmin toward the celestial equator.
- **Returns:** tuple `(ra_deg, dec_deg)` of NumPy arrays.

### 2.3. Gaia queries

#### `query_gaia_cached(wds_df, backend, components_file, is_displaced=False, model_sample=False)`
Runs one Gaia query with an on-disk cache. This is the function used by `main`.

| Parameter | Description |
| :--- | :--- |
| `wds_df` | Output of `propagate_coordinates` |
| `backend` | `"ari"` or `"esa"` |
| `components_file` | Path of the input file; its SHA-256 fingerprint is part of the cache key |
| `is_displaced` | `True` for the displaced control sample |
| `model_sample` | `True` for the 15-arcsec model samples, `False` for the 5-arcsec catalog |

- **Returns:** `astropy.table.Table` of WDS–Gaia candidate pairs, with `dist_arcsec` and `row_id`.

#### `query_gaia_selected_backend(wds_df, backend, is_displaced=False, model_sample=False)`
Same query without cache; dispatches to the ARI or ESA function.
- **Raises:** `ValueError` for an unknown backend.

#### `query_gaia_batches_ari(...)` and `query_gaia_batches_esa(...)`
Send the `TAP_UPLOAD` query in batches, with three retries per batch.
- **Raises:** `RuntimeError` if a batch fails after all retries.

### 2.4. Model construction

#### `validate_quasicolor_support(raw_table, disp_table)`
Checks that every finite quasi-color lies inside the frozen grid and prints a summary.
- **Raises:** `RuntimeError` if a value falls outside the grid or a sample has no finite magnitudes.

#### `build_empirical_bayesian_models(raw_table, disp_table)`
Builds all models from the RAW and displaced model samples.
- **Returns:** `dict` with keys `models_2d`, `models_theta` (both keyed by `(b_min, b_max, mag_min, mag_max)`) and `models_1d_b` (keyed by `(b_min, b_max)`). A value of `None` means the stratum could not be modeled.

#### `build_bayesian_model_2d(raw_valid, disp_valid)`
Builds one 2-D model in `(θ, q)` for a single stratum.
- **Returns:** `dict` with `bayes_2d` (posterior grid), `prior_H1`, `prior_H0`, `scale_S`, the fit diagnostics and the smoothed histograms, or `None` if the model is not identifiable.

#### `build_bayesian_model_theta_only(raw_valid, disp_valid)`
Builds one angular-only model for a single stratum.
- **Returns:** `dict` with `bayes_1d` and the same diagnostics, or `None`.

#### `build_b_only_astrometric_models(models_theta)`
Combines the six θ-only strata of each latitude bin, for components without WDS magnitude.
- **Returns:** `dict` keyed by `(b_min, b_max)`; `None` if any of the six source strata is missing.

#### `fit_displaced_h0_model(disp_dist, disp_dmag)`
Fits the background model: linear angular marginal × Gaussian-mixture quasi-color marginal × scalar `C`.
- **Returns:** `dict` with `Hmod`, `background_C` and fit diagnostics, or `None`.

#### `fit_medan_quasicolor_model_from_counts(measured, centers)`
Fits 2 to 10 Gaussians with seven deterministic starts each and selects the count by BIC.
- **Returns:** `dict` with `counts`, `n_components`, `bic`, `params` and diagnostics, or `None`.

#### `find_optimal_scale_factor(Htrue, Hmod)`
Finds the background scale `S` at the elbow of the negative-residual curve.
- **Returns:** `dict` with `scale_S`, `scale_grid`, `negative_residual_sum` and `broken_line_params`, or `None`.

#### `fit_linear_scale_through_origin(model_values, data_values)`
Fits the scalar `C` in `data ≈ C · model`.
- **Returns:** `float`, or `None` if the fit fails or `C ≤ 0`.

#### `medan_smooth(data)`
Applies `scipy.ndimage.percentile_filter(data, percentile=60, size=5)`.

### 2.5. Scoring

#### `score_candidates(raw_table, model_bundle)`
Assigns a posterior to every candidate.
- **Returns:** `pandas.DataFrame` with the added columns `quasi_color`, `posterior_prob`, `log_odds`, `bayes_model`, `model_prior_H1`, `model_prior_H0`, `scale_S` and `assignment_eligible`.
- **Raises:** `RuntimeError` if a candidate's quasi-color lies outside the frozen grid.

#### `get_hist_bin(value, bins)`, `get_b_key(b_abs)`, `get_wds_mag_key(wds_mag)`
Return the histogram bin index, latitude stratum or magnitude stratum of a value, or `None` if it lies outside every bin.

### 2.6. Assignment

#### `optimize_global_assignments(scored_df, wds_df)`
Solves the global one-to-one assignment.
- **Returns:** `pandas.DataFrame`, one row per WDS component, with `best_*` columns, `assigned_*` Gaia columns, `assignment_status`, the `assignment_group_*` diagnostics and `assigned_edge_margin`.

#### `build_graph_components(scored_df)`
Builds the candidate graph from edges with `log_odds ≥ 0` and splits it into connected components.
- **Returns:** tuple `(components, edges_by_w)`.

#### `assignment_margins(cost, M, N, pair_to_row, row_ind, col_ind)`
Computes the objective, the group margin and the per-edge margins of one solved component.
- **Returns:** tuple `(optimum, group_gap, edge_margins)`.
- **Raises:** `RuntimeError` if an alternative assignment exceeds the reported optimum.

### 2.7. Output and entry point

#### `save_model_diagnostics(model_bundle, model_summary_path, model_arrays_path)`
Writes the per-stratum summary CSV and the compressed model arrays (`.npz`).

#### `build_argument_parser()`
Returns the `argparse.ArgumentParser` with the options described in the User Guide.

#### `main(argv=None)`
Runs the full pipeline. `argv` is a list of command-line arguments; `None` reads them from the command line.

---

## 3. `summary.py`

#### `generate_summary_with_gaia(summary_path, crossmatch_path, output_path, elbadry_path=None)`
Combines WDS pairs with their assigned Gaia counterparts.
- **Returns:** `pandas.DataFrame`, one row per WDS pair. Also writes the CSV and an `.xlsx` copy.
- **Raises:** `ValueError` if the crossmatch file lacks `wds_id`, `component`, `assignment_status` or `assigned_source_id`.

#### `parse_summary_line(line)`
Parses one line of the WDS Summary file.
- **Returns:** `dict` of WDS fields, or `None` for lines that are too short.

#### `extract_component_letters(components_str)`
Splits a component label such as `AB`, `Aa,Ab` or `AB-C` into a tuple `(primary, secondary)`.

#### `find_pair_components(wds_id, discoverer, comp1, comp2, lookup)`
Finds the two distinct crossmatch rows for a pair, preferring `**p` for the primary and `**s` for the secondary.

#### `load_elbadry_pairs(fits_path)`
Loads the El-Badry et al. (2021) catalog.
- **Returns:** `dict` mapping sorted `(source_id1, source_id2)` pairs to `R_chance_align`; empty if the file is missing or unreadable.

#### `get_elbadry_info(gaia_id1, gaia_id2, elbadry_pairs)`
- **Returns:** tuple `(in_catalog, R_chance_align)`.