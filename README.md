# wds-crossmatch
Repository for the project: "A Statistical View of the Washington Double Star Catalog using Gaia DR3"

Quick description: This is an attempt to crossmatch every component of the Washington Double Star Catalog (WDS) to a Gaia DR3 source. The base crossmatching work (including the Summary file generation) was originally made for the [AstraKairos project](https://github.com/AstraKairos/astrakairos), which consisted of a simple weight system to assign a crossmatching confidence value based on the distance between a WDS coordinate and a Gaia DR3 source, as well as the magnitude difference between these catalogs. For this work, we opted for a more statistically-sound method by updating the per-component matching logic to include adaptations of the Bayesian crossmatching algorithm presented by [Medan et al.](https://doi.org/10.3847/1538-3881/abe878) as well as the Hungarian algorithm optimization shown by [Budavári & Basu](http://dx.doi.org/10.3847/0004-6256/152/4/86) (B&B's method was used as an attempt to partially solve the multiplicity problem mentioned in Medan's work).

> Note: This repository only contains a technical description of the code used in this project. A short scientific description is available in the poster we presented at the XXI SOCHIAS Annual Meeting. Further details of our methods will be mentioned in the project's paper (in-preparation).

Acknowledgements:
This research has made use of the Washington Double Star Catalog maintained at the U.S. Naval Observatory. This work has also made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement.

This work was made by the AstroKern student initivative collaboration at Universidad Técnica Federico Santa María (UTFSM), Santiago, Chile. We thank the "Fondos Concursables 2026-1" by the DREE at UTFSM for financially supporting AstroKern.

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

The first run queries Gaia DR3 and takes a few minutes. Query results are cached in `cache/`, so later runs on the same input file skip the Gaia queries. This feature was added to speed-up the testing process.

### 1.3. What You Get

The main output is `outputs/components_crossmatch_bayesian.csv`, with one row per WDS component. Key columns:

- `wds_id`, `component`: WDS identifier and component label
- `assigned_source_id`: Gaia DR3 source globally assigned to the component (empty if none)
- `best_posterior_prob`: highest pairwise association probability among eligible candidates, in [0, 1]
- `assignment_status`: one of `assigned_global`, `not_assigned_global`, `no_modeled_candidate`

> Note: The components output file also includes all gaiadr3.gaia_source columns for ease of use.

Validation: Results on the extended test sample (3,409 WDS components):

| `assignment_status` | Components | % |
| :--- | ---: | ---: |
| `assigned_global` | 1,350 | 39.6 % |
| `not_assigned_global` | 328 | 9.6 % |
| `no_modeled_candidate` | 1,731 | 50.8 % |

Additional outputs in `outputs/`: the scored candidate table, per-stratum model diagnostics and the compressed model arrays.

## 2. What It Does

crossmatch.py matches WDS components to their most likely Gaia DR3 counterparts, even when several candidates lie nearby:

- **Reads:** a WDS component list and propagates coordinates from epoch 2000.0 to the Gaia epoch 2016.0.
- **Queries:** Gaia DR3 within 15 arcsec (plus a displaced control field) to calibrate the statistical model, and within 5 arcsec to collect final candidates.
- **Scores:** each candidate with an empirical Bayesian association probability, following Medan et al. (2021).
- **Assigns:** components to Gaia sources one to one, resolving competing matches with the assignment formalism of Budavári & Basu (2016).

`summary.py` then combines the WDS pairs with the assigned Gaia counterparts and adds pair-level diagnostics, including: parallax, proper-motion and radial-velocity consistency, and an optional additional cross-reference with "A million binaries from Gaia eDR3" (El-Badry et al. 2021). This outputs .csv and .xlsx summary files.

## 3. Key Features

| Feature | Benefit |
| :--- | :--- |
| **Empirical Bayesian model** | Bayesian association probability (Medan et al., 2021) |
| **Global one-to-one assignment** | Resolves competing matches (Budavári & Basu, 2016) |
| **Disk cache** | Repeated runs reuse Gaia query results instead of querying Gaia again |
| **Parallel Gaia downloads** | Up to 5 simultaneous `TAP_UPLOAD` requests (25,000 targets per batch) |
| **Vectorized scoring** | Candidate scoring done with NumPy array operations |

Most of the runtime is spent fitting the Bayesian models, not downloading or scoring. Parallel downloads only help when the input needs more than one batch: the extended sample (3,409 components) fits in a single batch, while the full catalog (288,618 components) needs 12.

## 4. Scientific Background

The methodology implemented in this pipeline is primarily based on:

1. Medan, I., Lépine, S., & Hartman, Z. (2021). Bayesian Cross-matching of High Proper-motion Stars in Gaia DR2 and Photometric Metallicities for ~1.7 million K and M Dwarfs. *The Astronomical Journal*, 161, 234. https://doi.org/10.3847/1538-3881/abe878
2. Budavári, T., & Basu, A. (2016). Probabilistic Cross-identification in Crowded Fields as an Assignment Problem. *The Astronomical Journal*, 152, 86. https://doi.org/10.3847/0004-6256/152/4/86
3. El-Badry, K., Rix H-W., Heintz T. (2021). A million binaries from Gaia eDR3: sample selection and validation of Gaia parallax uncertainties. *Monthly Notices of the Royal Astronomical Society*, 506, 2. https://doi.org/10.1093/mnras/stab323

## 5. Documentation Directory

- [`docs/INSTALLATION.md`](docs/INSTALLATION.md): requirements, installation and troubleshooting
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md): running the pipeline and interpreting the outputs
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): pipeline stages, statistical model and design decisions
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md): functions, parameters and return values

## 6. Experiments

During the process of making the poster for the XXI SOCHIAS Annual Meeting, we did several tests to validate our results (and out of curiosity). Some of these *experiments* can be found at the `experiments`. We consider these some of the most interesting/cool plots and statistical results that didn't make it to the poster presentation. If you're interested in our work, we'd suggest to check them out! Some of these results are:

- [`experiments/orbit_ensemble.png`](experiments/orbit_ensemble.png): Attempt to fit the orbit of the system WDS 12204+0035, which has a very high *F* factor. It shows potential for an orbit computation, but better priors are needed (such as mass).
- [`experiments/full_fraction_vs_separation.png`](experiments/full_fraction_vs_separation.png): Comparing the completeness vs angular separation between this, the AstraKairos' crossmatch, and USNO's crossmatch.
- [`experiments/full_difference_new_minus_usno.png`](experiments/full_difference_new_minus_usno.png): The difference in completeness vs angular separation between this catalog and the USNO's crossmatch.
- [`experiments/hungarian_vs_no_hungarian.png`](experiments/hungarian_vs_no_hungarian.png): The fraction of selections affected by the Hungarian algorithm. This shows that the algorithm does cause greater completeness at lower angular separations.
- Overall recovery: 254,037/288,618 components matched (88.02%); 120,194/157,689 pairs classified as FULL (76.22%).
- Only 23 out of 126 characterized hierarchical triples were found in another higher-order binaries catalog (specifically, all 23 of them are in the Multiple Star Catalog, by Andrei Tokovinin). 