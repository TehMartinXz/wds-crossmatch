# wds-crossmatch
Repository for the project: "A Statistical View of the Washington Double Star Catalog using Gaia DR3"

Quick description: This is an attempt to crossmatch every component of the Washington Double Star Catalog (WDS) to a Gaia DR3 source. The base crossmatching work (including the Summary file generation) was originally made for the [AstraKairos project](https://github.com/AstraKairos/astrakairos), which consisted of a simple weight system to assign a crossmatching confidence value based on the distance between a WDS coordinate and a Gaia DR3 source, as well as the magnitude difference between these catalogs. For this work, we opted for a more statistically-sound method by updating the per-component matching logic to include adaptations of the Bayesian crossmatching algorithm presented by [Medan et al.](https://doi.org/10.3847/1538-3881/abe878) as well as the Hungarian algorithm optimization shown by [Budavári & Basu](http://dx.doi.org/10.3847/0004-6256/152/4/86) (B&B's method was used as an attempt to partially solve the multiplicity problem mentioned in Medan's work).

## Table of Contents
1. [Quick Start](#1-quick-start)
   1.1. [Installation](#11-installation)
   1.2. [Basic Usage](#12-basic-usage)
   1.3. [What You Get](#13-what-you-get)
2. [What It Does](#2-what-it-does)
3. [Key Features](#3-key-features)
4. [Project Status](#4-project-status)
5. [Scientific Background](#5-scientific-background)
6. [Documentation Directory](#6-documentation-directory)

---

## 1. Quick Start

### 1.1. Installation

```bash
git clone https://github.com/TehMartinXz/wds-crossmatch.git
cd wds-crossmatch
python3 -m pip install numpy pandas scipy astropy astroquery openpyxl
```

Tested with Python 3.13. See [`docs/INSTALLATION.md`](docs/INSTALLATION.md) for details and troubleshooting.

### 1.2. Basic Usage

```bash
python3 crossmatch.py --components example_data/wds_components.summ.extended.txt
```

The first run queries Gaia DR3 and takes a few minutes. Query results are cached in `cache/`, so later runs on the same input file are faster.

### 1.3. What You Get

The main output is `outputs/components_crossmatch_bayesian.csv`, with one row per WDS component. Key columns:

- `wds_id`, `component`: WDS identifier and component label
- `assigned_source_id`: Gaia DR3 source globally assigned to the component (empty if none)
- `best_posterior_prob`: highest pairwise association probability among eligible candidates, in [0, 1]
- `assignment_status`: one of `assigned_global`, `not_assigned_global`, `no_modeled_candidate`

Result on the extended sample (3,409 components):

| `assignment_status` | Components | % |
| :--- | ---: | ---: |
| `assigned_global` | 1,350 | 39.6 % |
| `not_assigned_global` | 328 | 9.6 % |
| `no_modeled_candidate` | 1,731 | 50.8 % |

Additional outputs in `outputs/`: the scored candidate table, per-stratum model diagnostics and the compressed model arrays.

## 2. What It Does

wds-crossmatch matches WDS components to their Gaia DR3 counterparts when several candidates lie nearby:

- **Reads:** a WDS component list and propagates coordinates from epoch 2000.0 to the Gaia epoch 2016.0.
- **Queries:** Gaia DR3 within 15 arcsec (plus a displaced control field) to calibrate the statistical model, and within 5 arcsec to collect final candidates.
- **Scores:** each candidate with an empirical Bayesian association probability, following Medan et al. (2021).
- **Assigns:** components to Gaia sources one to one, resolving competing matches with the assignment formalism of Budavári & Basu (2016).

`summary.py` then combines the WDS pairs with the assigned Gaia counterparts and adds pair-level diagnostics (parallax, proper-motion and radial-velocity consistency, and an optional cross-reference with El-Badry et al. 2021).

## 3. Key Features

| Feature | Benefit |
| :--- | :--- |
| **Empirical Bayesian model** | Quantified association probability, not just "closest source" |
| **Global one-to-one assignment** | Resolves competing matches (Budavári & Basu, 2016) |
| **Disk cache** | Repeated runs reuse Gaia query results (2 min 10 s → 59 s on the extended sample) |
| **Validated** | Outputs identical to the baseline on the extended sample (3,409 components) |

Vectorized candidate scoring and parallel Gaia downloads are available in the `feature/vectorization` branch, pending review. They produce identical results; no speedup factor is claimed yet, because profiling shows that most of the runtime is spent fitting the Bayesian models.

## 4. Project Status

- **Extended sample (3,409 components):** 1,350 components (39.6%) globally assigned.
- **Runtime:** about 1 minute with a warm cache on the extended sample.
- **Next steps:** review the `feature/vectorization` branch and run the full catalog (288,618 components).

## 5. Scientific Background

The methodology implemented in this pipeline is based on:

1. Medan, I., Lépine, S., & Hartman, Z. (2021). Bayesian Cross-matching of High Proper-motion Stars in Gaia DR2 and Photometric Metallicities for ~1.7 million K and M Dwarfs. *The Astronomical Journal*, 161, 234. https://doi.org/10.3847/1538-3881/abe878
2. Budavári, T., & Basu, A. (2016). Probabilistic Cross-identification in Crowded Fields as an Assignment Problem. *The Astronomical Journal*, 152, 86. https://doi.org/10.3847/0004-6256/152/4/86

## 6. Documentation Directory

- [`docs/INSTALLATION.md`](docs/INSTALLATION.md): requirements, installation and troubleshooting
- `docs/USER_GUIDE.md` (in preparation)
- `docs/ARCHITECTURE.md` (in preparation)
- `docs/API_REFERENCE.md` (in preparation)