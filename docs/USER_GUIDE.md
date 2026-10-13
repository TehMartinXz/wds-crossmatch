# User Guide

This guide explains how to run `wds-crossmatch`, what each output contains and
how to interpret the results. For installation, see
[`INSTALLATION.md`](INSTALLATION.md).

All commands are run from the repository root.

---

## 1. Pipeline overview

The project has two scripts that run in sequence:

```
crossmatch.py  →  outputs/components_crossmatch_bayesian.csv   (one row per WDS component)
                              ↓
summary.py     →  outputs/wds_summary_with_gaia.csv            (one row per WDS pair)
```

1. `crossmatch.py` assigns a Gaia DR3 source to each WDS component.
2. `summary.py` combines the components into WDS pairs and adds pair-level
   physical diagnostics.

---

## 2. Running `crossmatch.py`

### 2.1. Basic run

```bash
python3 crossmatch.py --components example_data/wds_components.summ.extended.txt
```

### 2.2. Command-line options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--components` | (always set it) | Input WDS component file, e.g. one of the samples in `example_data/` |
| `--backend` | `ari` | Gaia TAP service: `ari` (ARI Heidelberg) or `esa` (ESA Gaia Archive) |
| `--output` | `./outputs/components_crossmatch_bayesian.csv` | Final catalog, one row per component |
| `--candidates` | `./outputs/components_crossmatch_candidates_scored.csv` | Every scored candidate within 5 arcsec |
| `--model-summary` | `./outputs/components_crossmatch_model_summary.csv` | Per-stratum model diagnostics |
| `--model-arrays` | `./outputs/components_crossmatch_model_arrays.npz` | Compressed model arrays |

### 2.3. Example inputs

| File | Components | Use |
| :--- | ---: | :--- |
| `example_data/wds_components.summ.txt` | 64 | Installation check only: too small to calibrate the model, assigns nothing |
| `example_data/wds_components.summ.extended.txt` | 3,409 | Realistic test: calibrates the model and produces assignments |

### 2.4. Gaia query cache

`crossmatch.py` runs three Gaia queries: a 15-arcsec model sample, a displaced
control sample, and a 5-arcsec candidate catalog. Their results are saved in
`cache/` as VOTable files named after the query role, the backend and a
fingerprint of the input file, for example:

```
cache/gaia_raw_model_ari_8fdb05e3c362.vot
```

On later runs with the same input file and backend, the program prints
`Cache hit:` and reads the results from disk instead of querying Gaia. A query
is saved only after all its batches finish, so an interrupted run does not leave
partial files.

When the input is large, each query is split into batches of 25,000 components
and up to 5 batches are downloaded at the same time. The console shows lines
such as `[ARI] Batch 3 / 12 matched ... sources.`; batches may finish in any
order. The samples in `example_data/` fit in one batch, so they do not use
parallel downloads.

**Delete `cache/` manually** if you change query parameters in the code (search
radii, selected Gaia columns or Gaia table). The fingerprint covers only the
input file, not the source code, so old results would otherwise be reused.

### 2.5. Console output

A successful run ends with:

```text
Crossmatch complete!
WDS components: 3409
Scored candidates within 5.0 arcsec: 2865
Globally assigned components: 1350
```

(values shown for the extended sample).

Messages such as `Unmodeled 2-D stratum: ...` are expected. They mean that a
(magnitude, Galactic latitude) stratum had too few sources to calibrate the
full 2-D model. Candidates in those strata fall back to the angular-only
models when possible.

---

## 3. Understanding the `crossmatch.py` outputs

### 3.1. Final catalog: `components_crossmatch_bayesian.csv`

One row per WDS component. Column groups:

| Group | Examples | Content |
| :--- | :--- | :--- |
| WDS fields | `wds_id`, `component`, `wds_mag`, `ra_j2016`, `dec_j2016`, `gal_b` | Parsed WDS data and coordinates propagated to epoch 2016.0 |
| Best candidate | `best_source_id`, `best_posterior_prob`, `best_dist_arcsec`, `best_bayes_model` | Highest-posterior eligible candidate, kept even if it was not assigned |
| Assigned source | `assigned_source_id`, `assigned_posterior_prob`, `assigned_dist_arcsec`, `assigned_parallax`, ... | All Gaia DR3 columns of the globally assigned source, prefixed with `assigned_` |
| Assignment | `assignment_status`, `assignment_group_id`, `assignment_group_margin`, `assigned_edge_margin` | Result and diagnostics of the one-to-one assignment |

### 3.2. `assignment_status`

| Value | Meaning |
| :--- | :--- |
| `assigned_global` | A Gaia source was assigned to this component |
| `not_assigned_global` | The component has a modeled candidate, but none was assigned: either its posterior is below 0.5, or another component won that source |
| `no_modeled_candidate` | No candidate within 5 arcsec received a defined posterior |

Result on the extended sample:

| `assignment_status` | Components | % |
| :--- | ---: | ---: |
| `assigned_global` | 1,350 | 39.6 % |
| `not_assigned_global` | 328 | 9.6 % |
| `no_modeled_candidate` | 1,731 | 50.8 % |

### 3.3. `bayes_model`

Which empirical model scored each candidate:

| Value | Used when |
| :--- | :--- |
| `2D_WDSref` | WDS magnitude and Gaia G are both available: angular separation + quasi-color model |
| `1D_magb_missing_G_theta_calibrated` | WDS magnitude is known but Gaia G is missing: angular-only model for the same stratum |
| `1D_b_missing_WDS_mag` | WDS magnitude is missing: latitude-only angular model |
| `1D_b_missing_WDS_and_G` | Both magnitudes are missing: latitude-only angular model |
| `unmodeled` | No calibrated model applies to the candidate |

### 3.4. How to read the probabilities and margins

- `posterior_prob` is a **pairwise** probability: how likely this candidate is a
  true association rather than a random field star. It is not a probability over
  all possible global catalog assignments.
- `assigned_edge_margin` is how much the assignment objective would drop if this
  specific match were forbidden. Larger values mean a more robust assignment.
- `assignment_group_margin` is the gap to the best different assignment within
  the component's connected group.
- Neither margin is a probability.

### 3.5. Other outputs

| File | Content |
| :--- | :--- |
| `components_crossmatch_candidates_scored.csv` | Every candidate within 5 arcsec with its posterior, before assignment |
| `components_crossmatch_model_summary.csv` | One row per model stratum: validity, priors, scale factors, fit statistics |
| `components_crossmatch_model_summary.csv.fit_audit.csv` | Every Gaussian fit attempt: component count, start, cost, BIC |
| `components_crossmatch_model_arrays.npz` | Calibrated model arrays, for reproducibility and plotting |

---

## 4. Running `summary.py`

### 4.1. Inputs

| Option | Default | Description |
| :--- | :--- | :--- |
| `--summary` | `./data/wdsweb_summ2.txt` | WDS Summary file |
| `--crossmatch` | `./outputs/components_crossmatch_bayesian.csv` | Output of `crossmatch.py` |
| `--output` | `./outputs/wds_summary_with_gaia.csv` | Pair-level output (an `.xlsx` copy is also written) |
| `--elbadry` | `./data/all_columns_catalog.fits` | El-Badry et al. (2021) catalog (optional) |
| `--skip-elbadry` | off | Skip the El-Badry cross-reference |

The WDS Summary file and the El-Badry catalog are not included in the
repository; obtain them separately and place them in `data/`. Only WDS systems
present in the crossmatch output are processed.

### 4.2. Basic run

```bash
python3 summary.py --skip-elbadry
```

The console prints how many pairs were processed and how many have both
components matched (`FULL`), one (`PARTIAL`) or none (`NONE`).

### 4.3. Key output columns

| Column | Meaning |
| :--- | :--- |
| `completeness_flag` | `FULL`, `PARTIAL` or `NONE`: how many of the two components were assigned |
| `gaia_sep_arcsec`, `gaia_pa_deg` | Separation and position angle between the two assigned Gaia sources |
| `projected_sep_au` | Projected separation, using the primary's parallax |
| `sigma_delta_parallax` | Parallax difference in units of its uncertainty (inflated for RUWE > 1) |
| `delta_pm_norm` | Proper-motion difference normalized by the expected orbital motion |
| `sigma_delta_rv` | Radial-velocity difference in units of its uncertainty |
| `binary_type` | Simple classification from the Gaia color-magnitude diagram: `WDWD`, `WDMS`, `MSMS`, `WD??`, `MS??` or `????` |
| `in_elbadry_2021`, `R_chance_align` | Whether the pair appears in El-Badry et al. (2021), and its chance-alignment value |
| `best_equals_assigned_1/2` | Whether the assigned source is also the best individual candidate |

Large values of `sigma_delta_parallax` or `delta_pm_norm` suggest that the two
components may not form a physical binary.