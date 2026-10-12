"""
WDS component-level crossmatch to Gaia DR3.

The calculation uses the following algorithm:

1. Propagate WDS component coordinates from epoch 2000.0 to Gaia epoch 2016.0
   on the celestial sphere using the WDS proper-motion vector.
2. Query Gaia DR3 for a 15-arcsec RAW sample and a 15-arcsec 2-arcmin
   displaced control sample.
3. Construct an empirical Bayesian H1-versus-H0 association model following
   the framework and released implementation of Medan et al. (2021), with
   WDS-specific conditioning and the documented adaptations below.
4. Score Gaia candidates within 5 arcsec.
5. Resolve competing component matches with a global one-to-one bipartite
   assignment using pairwise posterior log-odds as edge utilities, following
   the assignment formalism of Budavari & Basu (2016).

Methodological details
----------------------
- WDS is the reference catalog. The quasi-color is

      q = G_Gaia - m_WDS

  and the empirical model is stratified by WDS reference magnitude and WDS
  absolute Galactic latitude.
- MODEL_RADIUS_ARCSEC = 15 arcsec is used only for empirical model calibration;
  ASSIGNMENT_RADIUS_ARCSEC = 5 arcsec defines final candidate eligibility.
- The production quasi-color grid is fixed to [-19, 22] mag with 0.25-mag bins.
  A run aborts if finite modeled quasi-colors fall outside this frozen support.
- The displaced H0 angular marginal is linear, excluding the outer 1 arcsec of
  the 15-arcsec cone to reduce search-boundary distortion.
- The displaced H0 quasi-color marginal is a 2--10 component Gaussian mixture.
  As in Medan's released source, fitting uses (data-model)/sqrt(data) on nonzero
  bins and BIC = N ln(RSS/N) + 3*n_components*ln(N). Gaussian means are
  constrained to the occupied displaced-q interval of each stratum. These bounds
  constrain the Gaussian centers; tails can extend outside that interval.
- The independent angular x quasi-color H0 surface receives a scalar C fitted
  to occupied 2-D displaced cells. Medan's scale S, percentile smoothing,
  H1 subtraction, empirical priors, and Bayes grid are then constructed.
- Percentile smoothing is scipy.ndimage.percentile_filter(..., percentile=60,
  size=5), matching the supplied Medan implementation.
- If WDS magnitude is known but Gaia G is missing, a separately calibrated
  theta-only model is used for the same (m_WDS, |b|) stratum. These theta-only
  models are trained on all RAW/DISPLACED candidates in the stratum, regardless
  of whether Gaia G is available, so the fallback does not inherit conditioning
  on finite Gaia photometry. If WDS magnitude is missing, a |b|-only model is
  formed by combining all six calibrated theta-only WDS-magnitude strata in
  count space; it is disabled unless all six source strata are available.
- The 2-arcmin displaced control is shifted toward the celestial equator, using
  a spherical meridional offset.
- The final global assignment is not the analytic likelihood of Budavari &
  Basu (2016). Their one-to-one assignment formalism is applied to empirical
  Medan-style pairwise posterior log-odds.
- Negative-utility edges (posterior < 0.5) are excluded from the assignment
  graph because a zero-utility orphan option is always available; this does not
  change the maximum-weight solution.
- The reported pairwise posterior is an empirical association probability under
  H1 versus random-field H0. It is not a marginal posterior over all possible
  global catalog assignments.

Gaia retrieval
--------------
Two TAP_UPLOAD backends are retained:
- ARI Heidelberg (default production backend)
- ESA Gaia Archive

The 15-arcsec model queries retrieve only source_id, position and Gaia G; the
5-arcsec catalog query retains the full Gaia source row.

Outputs
-------
- One final row per WDS component, containing both the globally assigned Gaia
  source (if any) and the highest-posterior individual eligible candidate.
- A scored-candidate table containing every final candidate edge within 5 arcsec.
- Scalar model diagnostics and compressed model arrays for reproducibility.
- Per-start Gaussian fit-audit CSV at <model-summary>.fit_audit.csv.

Deterministic Gaussian initialization
-------------------------------------
For every Gaussian-component count n=2..10, seven fixed initial conditions are
tried: the original Medan-style coincident start, midpoint weighted-quantile and
uniform starts, and deterministic half-/double-width variants. No random-number
generator or seed is used. Within each n, the converged fit with minimum weighted
least-squares cost is retained; Medan's unweighted RSS-based BIC then selects n.
The optimizer budget is max_nfev=10000. Per-start fit diagnostics are saved at
<model-summary>.fit_audit.csv.

Assignment margins are computed for every retained connected component.
assigned_edge_margin is the objective loss on forbidding that selected real edge.
assignment_group_margin is the best different real mapping gap within the retained
nonnegative-utility graph, including orphan alternatives. Dummy permutations do
not count. Neither margin is a probability. Group sizes and objectives are
repeated per target. A missing group margin indicates a target outside the retained
graph; a missing edge margin indicates that no real edge was assigned.

Input selection and validation
-----------------------------
The module-level test switch selects the example-data path when True and the
production WDS path when False. --components can override either default.
The parser accepts fixed-format HHMMSS.ss and signed DDMMSS.s coordinates, fails
on malformed/out-of-range records, and permits the valid direction (0, 0).
Both retrieval services use the same downstream calibration and assignment.
"""

import argparse
import os
import numpy as np
import pandas as pd
import time
import warnings
import re

from astroquery.gaia import Gaia
from astroquery.utils.tap.core import TapPlus
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.ndimage import percentile_filter
from scipy.optimize import curve_fit, linear_sum_assignment, least_squares, OptimizeWarning



# -----------------------------------------------------------------------------
# Production configuration
# -----------------------------------------------------------------------------
DEFAULT_GAIA_BACKEND = "ari" # "ari" or "esa"
FIT_MAX_NFEV = 10000
# Seven fixed Gaussian starts; no random-number generator is used.
DETERMINISTIC_START_SPECS = [
    ("original", "original", None),
    ("quantile_mid_w1", "quantile_mid", 1.0),
    ("uniform_mid_w1", "uniform_mid", 1.0),
    ("quantile_mid_w0p5", "quantile_mid", 0.5),
    ("quantile_mid_w2", "quantile_mid", 2.0),
    ("uniform_mid_w0p5", "uniform_mid", 0.5),
    ("uniform_mid_w2", "uniform_mid", 2.0),
]
FIT_AUDIT = []
ARI_TAP_URL = "https://gaia.ari.uni-heidelberg.de/tap"
ARI_BATCH_SIZE = 10000
ESA_BATCH_SIZE = 10000

# Enable testing mode
test = False

DEFAULT_COMPONENTS_FILE = "./data/wds_components.summ.txt"
DEFAULT_OUTPUT_FILE = "./outputs/components_crossmatch_bayesian.csv"
DEFAULT_CANDIDATE_OUTPUT_FILE = "./outputs/components_crossmatch_candidates_scored.csv"
DEFAULT_MODEL_SUMMARY_FILE = "./outputs/components_crossmatch_model_summary.csv"
DEFAULT_MODEL_ARRAYS_FILE = "./outputs/components_crossmatch_model_arrays.npz"

if test:
    DEFAULT_COMPONENTS_FILE = "./example_data/wds_components.summ.extended.txt"
    DEFAULT_OUTPUT_FILE = "./example_outputs/components_crossmatch_bayesian.csv"
    DEFAULT_CANDIDATE_OUTPUT_FILE = "./example_outputs/components_crossmatch_candidates_scored.csv"
    DEFAULT_MODEL_SUMMARY_FILE = "./example_outputs/components_crossmatch_model_summary.csv"
    DEFAULT_MODEL_ARRAYS_FILE = "./example_outputs/components_crossmatch_model_arrays.npz"

gaia_epoch = 2016.0
wds_epoch = 2000.0

# Medan's released code stores all candidate associations out to 15 arcsec for
# constructing the empirical distributions, while a 5-arcsec condition is used
# for the best-match/candidate stage. Keep those roles separate here.
MODEL_RADIUS_ARCSEC = 15.0
ASSIGNMENT_RADIUS_ARCSEC = 5.0
model_radius_deg = MODEL_RADIUS_ARCSEC / 3600.0
assignment_radius_deg = ASSIGNMENT_RADIUS_ARCSEC / 3600.0

displacement_deg = 2.0 / 60.0  # 2 arcmin, as in Medan et al. (2021)

# Empirical Bayesian conditioning.
# The |b| cuts are retained from Medan et al. (2021). WDS is the reference
# catalog here, so the magnitude strata are defined using m_WDS rather than
# candidate Gaia G. The boundaries below are rounded approximately equal-count
# quantiles from the WDS reference-magnitude population.
b_bins = [(0.0, 19.5), (19.5, 41.8), (41.8, 90.1)]
wds_mag_strata = [
    (-np.inf, 10.0),
    (10.0, 11.0),
    (11.0, 12.0),
    (12.0, 13.0),
    (13.0, 15.5),
    (15.5, np.inf),
]

# Angular grid for the 15-arcsec MODEL sample. A fixed width makes the
# smoothing scale explicit and avoids silently changing resolution when radii
# are changed. 0.2 arcsec is also the scale used throughout diagnostic plotting
# in the released Medan implementation.
DIST_BIN_WIDTH = 0.20
dist_bins = np.arange(
    0.0,
    MODEL_RADIUS_ARCSEC + DIST_BIN_WIDTH * 0.5,
    DIST_BIN_WIDTH
)

# Medan excludes the high-separation edge of the displaced angular histogram
# before fitting its linear marginal because the search boundary can distort
# the edge. Their exact source uses a grid-specific hard-coded trimming rule.
# Our TAP cone search has a different edge geometry, so we retain the same
# principle but exclude only the outer 1 arcsec. THIS IS AN EXPLICIT ADAPTATION.
ANGULAR_H0_EDGE_TRIM_ARCSEC = 1.0

# Quasi-color q = G_Gaia - m_WDS. The production support was frozen after the
# full-catalog RAW+DISPLACED distribution was inspected.
DMAG_MIN = -19.0
DMAG_MAX = 22.0
DMAG_BIN_WIDTH = 0.25
_N_DMAG_BINS = int(round((DMAG_MAX - DMAG_MIN) / DMAG_BIN_WIDTH))
dmag_bins = DMAG_MIN + np.arange(_N_DMAG_BINS + 1, dtype=float) * DMAG_BIN_WIDTH

# -----------------------------------------------------------------------------
# Medan et al. (2021) smoothing
# -----------------------------------------------------------------------------
def medan_smooth(data):
    """Apply the percentile smoothing used in the supplied Medan code."""
    return percentile_filter(
        data,
        percentile=60,
        size=5
    )


# -----------------------------------------------------------------------------
# WDS parsing
# -----------------------------------------------------------------------------
def parse_wds_components(components_file):
    """Parse the WDS component file into a vectorized pandas DataFrame."""
    print("Parsing WDS file (Vectorized)...")

    with open(components_file, "r", encoding="latin-1") as f:
        lines = [line.rstrip("\n\r") for line in f]
    if not lines:
        raise ValueError("WDS input is empty.")
    for number, line in enumerate(lines, 1):
        # The WDS coordinate fields are fixed-width, but their printed decimal
        # precision can vary and trailing component-type columns may be blank.
        # Require the coordinate fields themselves, then validate their numeric
        # structure and ranges without imposing a particular number of decimals.
        if len(line) < 130:
            raise ValueError(
                f"Malformed WDS record at input line {number}: expected at least 130 characters, got {len(line)}."
            )

        ra = line[112:121].strip()
        dec = line[121:130].strip()
        if (re.fullmatch(r"\d{6}(?:\.\d*)?", ra) is None
                or re.fullmatch(r"[+-]\d{6}(?:\.\d*)?", dec) is None):
            raise ValueError(
                f"Malformed WDS coordinate at input line {number}: RA={ra!r}, Dec={dec!r}"
            )

        hh, mm, ss = int(ra[:2]), int(ra[2:4]), float(ra[4:])
        dd, dm, ds = int(dec[1:3]), int(dec[3:5]), float(dec[5:])

        # WDS fixed-width coordinates can contain a rounded seconds field of
        # exactly 60.0 (e.g. -03 56 60.0). Treat that as the normal sexagesimal
        # carry into the next minute rather than rejecting an otherwise valid
        # coordinate. The arithmetic conversion below already performs that carry.
        if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss <= 60
                and 0 <= dd <= 90 and 0 <= dm < 60 and 0 <= ds <= 60):
            raise ValueError(
                f"Out-of-range WDS coordinate at input line {number}: RA={ra!r}, Dec={dec!r}"
            )

        ra_deg_check = (hh + mm / 60.0 + ss / 3600.0) * 15.0
        dec_abs_check = dd + dm / 60.0 + ds / 3600.0
        if not (ra_deg_check <= 360.0 and dec_abs_check <= 90.0):
            raise ValueError(
                f"Out-of-range WDS coordinate at input line {number}: RA={ra!r}, Dec={dec!r}"
            )

    df = pd.DataFrame({"raw": lines})
    df = df[df["raw"].str.len() >= 130].copy()

    # 1. Identifiers
    df["wds_id"] = df["raw"].str.slice(0, 10).str.strip()
    discoverer_full = df["raw"].str.slice(10, 22).str.strip()
    df["component"] = discoverer_full.str.extract(r"([A-Za-z]+)$").fillna("A")
    df["discoverer"] = discoverer_full.str.replace(r"([A-Za-z]+)$", "", regex=True).str.strip()

    # 2. Observations & relative parameters
    df["first_obs"] = pd.to_numeric(df["raw"].str.slice(23, 27), errors="coerce")
    df["last_obs"] = pd.to_numeric(df["raw"].str.slice(28, 32), errors="coerce")
    df["observations"] = pd.to_numeric(df["raw"].str.slice(32, 37), errors="coerce").fillna(0).astype(int)
    df["position_angle"] = pd.to_numeric(df["raw"].str.slice(37, 44), errors="coerce")
    df["separation"] = pd.to_numeric(df["raw"].str.slice(44, 53), errors="coerce")

    # 3. Magnitudes & spectra
    df["mag_primary"] = pd.to_numeric(df["raw"].str.slice(53, 63), errors="coerce").fillna(1234.0)
    df["mag_secondary"] = pd.to_numeric(df["raw"].str.slice(63, 70), errors="coerce").fillna(1234.0)
    df["spectral_type"] = df["raw"].str.slice(70, 80).str.strip()

    # 4. Proper motions
    df["pmra_primary"] = pd.to_numeric(df["raw"].str.slice(80, 84), errors="coerce").fillna(0.0)
    df["pmdec_primary"] = pd.to_numeric(df["raw"].str.slice(84, 88), errors="coerce").fillna(0.0)
    df["pmra_secondary"] = pd.to_numeric(df["raw"].str.slice(89, 93), errors="coerce").fillna(0.0)
    df["pmdec_secondary"] = pd.to_numeric(df["raw"].str.slice(93, 97), errors="coerce").fillna(0.0)

    # 5. Notes
    df["notes"] = df["raw"].str.slice(106, 112).str.strip()

    # 6. RA components and full string
    df["ra_hh"] = pd.to_numeric(df["raw"].str.slice(112, 114), errors="coerce").fillna(0).astype(int)
    df["ra_mm"] = pd.to_numeric(df["raw"].str.slice(114, 116), errors="coerce").fillna(0).astype(int)
    df["ra_ss"] = pd.to_numeric(df["raw"].str.slice(116, 121), errors="coerce").fillna(0.0)
    df["ra"] = df["raw"].str.slice(112, 121).str.strip()

    # 7. Dec components and full string
    df["dec_sign"] = np.where(df["raw"].str.slice(121, 122) == "-", -1, 1)
    df["dec_deg"] = pd.to_numeric(df["raw"].str.slice(122, 124), errors="coerce").fillna(0).astype(int)
    df["dec_mm"] = pd.to_numeric(df["raw"].str.slice(124, 126), errors="coerce").fillna(0).astype(int)
    df["dec_ss"] = pd.to_numeric(df["raw"].str.slice(126, 130), errors="coerce").fillna(0.0)
    df["dec"] = df["raw"].str.slice(121, 130).str.strip()

    # 8. Component type
    df["component_type"] = df["raw"].str.slice(131, 134).str.strip()

    # Coordinate calculation before spherical propagation.
    df["ra_deg_calc"] = ((
        df["ra_hh"] + df["ra_mm"] / 60.0 + df["ra_ss"] / 3600.0
    ) * 15.0) % 360.0
    df["dec_deg_calc"] = df["dec_sign"] * (
        df["dec_deg"] + df["dec_mm"] / 60.0 + df["dec_ss"] / 3600.0
    )

    # Coordinates were validated before conversion. (0, 0) is a valid direction.

    # Unique ID for tracking.
    df["upload_id"] = np.arange(len(df))
    df = df.drop(columns=["raw"])

    return df


# -----------------------------------------------------------------------------
# Spherical propagation
# -----------------------------------------------------------------------------
def propagate_coordinates(wds_df):
    """
    Propagate WDS directions from J2000 to the Gaia reference epoch on the
    celestial sphere.

    This is an explicit adaptation of Medan et al. (2021), who use first-order
    spherical-coordinate propagation equations. Here the same proper-motion
    information is applied as a tangent-plane vector to a unit direction vector,
    then renormalized. This avoids choosing a particular declination in the
    cos(delta) factor and avoids manual pole clipping.

    No distance, parallax, or radial velocity is introduced.
    """
    print("Propagating coordinates on the celestial sphere...")

    dt_years = gaia_epoch - wds_epoch

    # 1. PM inheritance logic: secondary can use its own PM when present;
    # otherwise inherit the primary PM.
    is_sec = (wds_df["component_type"] == "**s").to_numpy()
    has_sec_pm = (
        (wds_df["pmra_secondary"] != 0) |
        (wds_df["pmdec_secondary"] != 0)
    ).to_numpy()
    use_sec_pm = is_sec & has_sec_pm

    pm_ra_cosdec = np.where(
        use_sec_pm,
        wds_df["pmra_secondary"],
        wds_df["pmra_primary"]
    ).astype(float)

    pm_dec = np.where(
        use_sec_pm,
        wds_df["pmdec_secondary"],
        wds_df["pmdec_primary"]
    ).astype(float)

    # 2. WDS 'P' note: arcsec per 100 years -> mas/yr.
    mask_p = wds_df["notes"].str.contains("P", na=False).to_numpy()
    pm_ra_cosdec[mask_p] *= 10.0
    pm_dec[mask_p] *= 10.0

    # 3. Target magnitude used for the Medan-style quasi-color.
    wds_df["wds_mag"] = np.where(
        is_sec,
        wds_df["mag_secondary"],
        wds_df["mag_primary"]
    )

    # 4. Initial direction vector at the WDS epoch.
    alpha = np.radians(wds_df["ra_deg_calc"].to_numpy(float))
    delta = np.radians(wds_df["dec_deg_calc"].to_numpy(float))

    r0 = np.column_stack([
        np.cos(delta) * np.cos(alpha),
        np.cos(delta) * np.sin(alpha),
        np.sin(delta)
    ])

    # Tangent basis vectors.
    e_alpha = np.column_stack([
        -np.sin(alpha),
         np.cos(alpha),
         np.zeros_like(alpha)
    ])

    e_delta = np.column_stack([
        -np.sin(delta) * np.cos(alpha),
        -np.sin(delta) * np.sin(alpha),
         np.cos(delta)
    ])

    # mas -> rad.
    mas_to_rad = np.deg2rad(1.0 / 3_600_000.0)

    tangent_velocity = (
        pm_ra_cosdec[:, None] * e_alpha +
        pm_dec[:, None] * e_delta
    ) * mas_to_rad

    # First-order propagation and renormalization to the unit sphere.
    r1 = r0 + dt_years * tangent_velocity
    r1 /= np.linalg.norm(r1, axis=1)[:, None]

    ra_2016 = np.degrees(np.arctan2(r1[:, 1], r1[:, 0])) % 360.0
    dec_2016 = np.degrees(np.arcsin(np.clip(r1[:, 2], -1.0, 1.0)))

    wds_df["ra_j2016"] = ra_2016
    wds_df["dec_j2016"] = dec_2016

    c_epoch = SkyCoord(
        ra=ra_2016 * u.deg,
        dec=dec_2016 * u.deg,
        frame="icrs"
    )
    wds_df["gal_b"] = c_epoch.galactic.b.deg

    wds_df.drop(columns=["ra_deg_calc", "dec_deg_calc"], inplace=True)


# -----------------------------------------------------------------------------
# Gaia candidate retrieval
# -----------------------------------------------------------------------------
def get_displaced_coordinates(ra_deg, dec_deg):
    """
    Apply the Medan et al. (2021) 2-arcmin displaced-control offset.

    The RELEASED Medan cross_match_dis() code shifts northern targets by
    -2 arcmin in declination and southern targets by +2 arcmin: i.e. TOWARD
    the celestial equator. We reproduce that sign convention, but use a
    spherical meridional offset so the geometry remains valid everywhere.
    """
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)

    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    # Position angle 180 deg = south for northern targets; 0 deg = north for
    # southern targets. At dec=0, choose south; the distinction is immaterial
    # to the local-control construction but makes the convention deterministic.
    position_angle = np.where(dec_deg >= 0.0, 180.0, 0.0) * u.deg
    displaced = coords.directional_offset_by(
        position_angle,
        displacement_deg * u.deg
    )

    return displaced.ra.deg, displaced.dec.deg


def query_gaia_batches_esa(wds_df, is_displaced=False, model_sample=False):
    """
    Query the ESA Gaia Archive with TAP_UPLOAD and return WDS--Gaia candidate pairs.

    Three roles are supported:
      RAW catalog candidates: 5 arcsec, g.* retained for the final product.
      RAW model sample:       15 arcsec, minimal Gaia columns only.
      DISPLACED model sample: 15 arcsec, minimal Gaia columns only.

    Splitting RAW-model and RAW-catalog retrieval preserves the exact Bayesian
    training sample while avoiding transfer of every Gaia DR3 column for the
    5--15 arcsec background-dominated annulus.
    """
    df_upload = wds_df[
        ["upload_id", "wds_id", "component", "ra_j2016", "dec_j2016", "wds_mag", "gal_b"]
    ].copy()
    df_upload.rename(
        columns={"ra_j2016": "ra_target", "dec_j2016": "dec_target"},
        inplace=True
    )

    if is_displaced:
        ra_disp, dec_disp = get_displaced_coordinates(
            df_upload["ra_target"].to_numpy(float),
            df_upload["dec_target"].to_numpy(float)
        )
        df_upload["ra_target"] = ra_disp
        df_upload["dec_target"] = dec_disp

    upload_table = Table.from_pandas(df_upload)

    if is_displaced:
        radius_deg = model_radius_deg
        full_gaia_columns = False
        mode_str = "DISPLACED MODEL (15 arcsec, Background)"
    elif model_sample:
        radius_deg = model_radius_deg
        full_gaia_columns = False
        mode_str = "RAW MODEL (15 arcsec, True+Background)"
    else:
        radius_deg = assignment_radius_deg
        full_gaia_columns = True
        mode_str = "RAW CATALOG (5 arcsec, full Gaia output)"

    batch_size = ESA_BATCH_SIZE
    all_results = []
    print(f"\n--- Starting Gaia TAP Queries from ESA: {mode_str} ---")

    gaia_select = (
        "g.*"
        if full_gaia_columns
        else "g.source_id, g.ra, g.dec, g.phot_g_mean_mag"
    )

    if not full_gaia_columns:
        print("Model query: requesting minimal Gaia columns only.")

    for i in range(0, len(upload_table), batch_size):
        batch_table = upload_table[i:i + batch_size]

        adql_query = f"""
        SELECT
            u.upload_id, u.wds_id, u.component, u.wds_mag, u.gal_b,
            {gaia_select},
            DISTANCE(
                POINT('ICRS', u.ra_target, u.dec_target),
                POINT('ICRS', g.ra, g.dec)
            ) * 3600.0 AS dist_arcsec
        FROM TAP_UPLOAD.my_table AS u
        JOIN gaiadr3.gaia_source AS g
        ON 1 = CONTAINS(
            POINT('ICRS', u.ra_target, u.dec_target),
            CIRCLE('ICRS', g.ra, g.dec, {radius_deg})
        )
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                job = Gaia.launch_job_async(
                    query=adql_query,
                    upload_resource=batch_table,
                    upload_table_name="my_table"
                )
                batch_results = job.get_results()
                all_results.append(batch_results)
                print(
                    f"Batch {(i // batch_size) + 1} / "
                    f"{int(np.ceil(len(upload_table) / batch_size))} "
                    f"matched {len(batch_results)} sources."
                )
                break
            except Exception as exc:
                print(
                    f"  Error: {exc}. Retrying {attempt + 1}/{max_retries} in 10 seconds..."
                )
                time.sleep(10)
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Failed to reach Gaia TAP server after max retries."
                    ) from exc

    if not all_results:
        return Table()

    stacked_table = vstack(all_results, metadata_conflicts="silent")
    stacked_table["row_id"] = np.arange(len(stacked_table))
    return stacked_table



def query_gaia_batches_ari(wds_df, is_displaced=False, model_sample=False):
    """
    Query the ARI Heidelberg Gaia TAP mirror using standard TAP_UPLOAD.

    ARI explicitly supports user-table uploads and positional crossmatches.  The
    uploaded WDS table is the small side of the JOIN, while Gaia is the POINT and
    each uploaded WDS position defines the CIRCLE, matching ARI's documented
    optimization guidance.

    Model samples use ``gaiadr3.gaia_source_lite`` because only source_id, ra,
    dec and phot_g_mean_mag are needed.  The final <=5-arcsec candidate query uses
    the full ``gaiadr3.gaia_source`` table so the output can retain every Gaia
    source column, exactly as with the ESA backend.
    """
    df_upload = wds_df[
        ["upload_id", "wds_id", "component", "ra_j2016", "dec_j2016", "wds_mag", "gal_b"]
    ].copy()
    df_upload.rename(
        columns={"ra_j2016": "ra_target", "dec_j2016": "dec_target"},
        inplace=True
    )

    if is_displaced:
        ra_disp, dec_disp = get_displaced_coordinates(
            df_upload["ra_target"].to_numpy(float),
            df_upload["dec_target"].to_numpy(float)
        )
        df_upload["ra_target"] = ra_disp
        df_upload["dec_target"] = dec_disp

    upload_table = Table.from_pandas(df_upload)

    if is_displaced:
        radius_deg = model_radius_deg
        full_gaia_columns = False
        mode_str = "DISPLACED MODEL (15 arcsec, Background)"
    elif model_sample:
        radius_deg = model_radius_deg
        full_gaia_columns = False
        mode_str = "RAW MODEL (15 arcsec, True+Background)"
    else:
        radius_deg = assignment_radius_deg
        full_gaia_columns = True
        mode_str = "RAW CATALOG (5 arcsec, full Gaia output)"

    tap = TapPlus(url=ARI_TAP_URL)
    batch_size = ARI_BATCH_SIZE
    all_results = []

    gaia_table = (
        "gaiadr3.gaia_source"
        if full_gaia_columns
        else "gaiadr3.gaia_source_lite"
    )
    gaia_select = (
        "g.*"
        if full_gaia_columns
        else "g.source_id, g.ra, g.dec, g.phot_g_mean_mag"
    )

    print(f"\n--- Starting Gaia TAP Queries from ARI Heidelberg: {mode_str} ---")
    print(f"ARI table: {gaia_table}; TAP_UPLOAD batch size: {batch_size}")
    if not full_gaia_columns:
        print("Model query: requesting minimal Gaia columns from gaia_source_lite.")

    n_batches = int(np.ceil(len(upload_table) / batch_size))
    for i in range(0, len(upload_table), batch_size):
        batch_table = upload_table[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        # ARI recommends: smallest table first in JOIN; Gaia sources as POINT;
        # uploaded/input positions as CIRCLE centers.
        adql_query = f"""
        SELECT
            u.upload_id, u.wds_id, u.component, u.wds_mag, u.gal_b,
            {gaia_select},
            DISTANCE(
                POINT('', g.ra, g.dec),
                POINT('', u.ra_target, u.dec_target)
            ) * 3600.0 AS dist_arcsec
        FROM TAP_UPLOAD.my_table AS u
        JOIN {gaia_table} AS g
        ON CONTAINS(
            POINT('', g.ra, g.dec),
            CIRCLE('', u.ra_target, u.dec_target, {radius_deg:.12f})
        ) = 1
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                job = tap.launch_job_async(
                    query=adql_query,
                    upload_resource=batch_table,
                    upload_table_name="my_table"
                )
                batch_results = job.get_results()
                all_results.append(batch_results)
                print(
                    f"Batch {batch_num} / {n_batches} "
                    f"matched {len(batch_results)} sources."
                )
                break
            except Exception as exc:
                print(
                    f"  ARI error on Batch {batch_num}: {exc}. "
                    f"Retrying {attempt + 1}/{max_retries} in 10 seconds..."
                )
                time.sleep(10)
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"ARI Gaia TAP failed on batch {batch_num} after "
                        f"{max_retries} attempts."
                    ) from exc

    if not all_results:
        return Table()

    stacked_table = vstack(all_results, metadata_conflicts="silent")
    stacked_table["row_id"] = np.arange(len(stacked_table))
    return stacked_table

def query_gaia_selected_backend(wds_df, backend, is_displaced=False, model_sample=False):
    """Dispatch one Gaia query stage to ARI Heidelberg or the ESA Gaia Archive."""
    backend = str(backend).strip().lower()

    if backend == "ari":
        return query_gaia_batches_ari(
            wds_df,
            is_displaced=is_displaced,
            model_sample=model_sample,
        )
    if backend == "esa":
        return query_gaia_batches_esa(
            wds_df,
            is_displaced=is_displaced,
            model_sample=model_sample,
        )

    raise ValueError(
        f"Unknown Gaia backend {backend!r}. Choose 'ari' or 'esa'."
    )


# -----------------------------------------------------------------------------
# Quasi-color grid configuration
# -----------------------------------------------------------------------------
def _finite_quasicolors(table):
    """Return finite q = G_Gaia - m_WDS values from a candidate table."""
    df = table.to_pandas() if isinstance(table, Table) else table

    wds_mag = pd.to_numeric(df["wds_mag"], errors="coerce").to_numpy(float)
    gaia_g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(float)

    valid = (
        np.isfinite(wds_mag) &
        (wds_mag != 1234.0) &
        np.isfinite(gaia_g)
    )

    return gaia_g[valid] - wds_mag[valid]


def validate_quasicolor_support(raw_table, disp_table):
    """Verify that all finite modeled quasi-colors lie inside the frozen grid."""
    q_raw = _finite_quasicolors(raw_table)
    q_disp = _finite_quasicolors(disp_table)

    if len(q_raw) == 0 or len(q_disp) == 0:
        raise RuntimeError(
            "RAW and DISPLACED model samples must both contain finite WDS and Gaia G magnitudes."
        )

    q_all = np.concatenate([q_raw, q_disp])
    observed_min = float(np.min(q_all))
    observed_max = float(np.max(q_all))

    if observed_min < DMAG_MIN or observed_max > DMAG_MAX:
        raise RuntimeError(
            "Observed quasi-color values lie outside the frozen production support: "
            f"observed=[{observed_min:.3f}, {observed_max:.3f}], "
            f"fixed=[{DMAG_MIN:.3f}, {DMAG_MAX:.3f}]."
        )

    print("\nValidated frozen quasi-color grid:")
    print("  q = G_Gaia - m_WDS")
    print(f"  RAW finite q values:       {len(q_raw)}")
    print(f"  DISPLACED finite q values: {len(q_disp)}")
    print(f"  Observed q range:          [{observed_min:.3f}, {observed_max:.3f}] mag")
    print(f"  Production q support:      [{DMAG_MIN:.3f}, {DMAG_MAX:.3f}] mag")
    print(f"  Bin width:                 {DMAG_BIN_WIDTH:.3f} mag")
    print(f"  Number of q bins:          {len(dmag_bins) - 1}")


# -----------------------------------------------------------------------------
# Medan scale-factor calculation
# -----------------------------------------------------------------------------
def broken_line_model(x, a, b, c, d):
    """Model used in the Medan et al. scale-factor fit."""
    one = a * x + b
    two = c * x + d
    return np.minimum(one, two)


def find_optimal_scale_factor(Htrue, Hmod):
    """
    Determine Medan's displaced-background scale S from the negative-residual
    elbow and retain the full diagnostic curve for publication/auditing.

    Returns None if the elbow is not identifiable, otherwise a dictionary with
    scale_S, the sampled S grid, negative-residual sums, and fitted broken-line
    parameters.
    """
    scales = np.arange(0.0, 5.0, 0.05)
    sums_neg = np.zeros_like(scales)

    Htrue_smooth = medan_smooth(Htrue)

    for i, scale in enumerate(scales):
        Hsub_smooth = Htrue_smooth - medan_smooth(Hmod * scale)
        sums_neg[i] = np.sum(Hsub_smooth[Hsub_smooth < 0.0])

    if np.allclose(sums_neg, sums_neg[0]):
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                broken_line_model,
                scales,
                sums_neg,
                p0=(-1.0, 1.0, -10000.0, 1.0),
                maxfev=20000
            )
    except Exception:
        return None

    denominator = popt[0] - popt[2]
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return None

    scale = (popt[3] - popt[1]) / denominator
    if not np.isfinite(scale) or scale < 0.0 or scale >= 5.0:
        return None

    return {
        "scale_S": float(scale),
        "scale_grid": scales.copy(),
        "negative_residual_sum": sums_neg.copy(),
        "broken_line_params": np.asarray(popt, dtype=float),
    }


# -----------------------------------------------------------------------------
# Empirical Bayesian model construction
# -----------------------------------------------------------------------------
def _medan_quasicolor_model_from_vector(x, params):
    """Evaluate a flat (A, mu, sigma) Gaussian parameter vector."""
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    model = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amplitude, mean, sigma = params[i:i + 3]
        model += amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return model


def _medan_quasicolor_weighted_residual(params, x, data):
    """Residual (data-model)/sqrt(data), as in Medan's n_gauss1d()."""
    return (data - _medan_quasicolor_model_from_vector(x, params)) / np.sqrt(data)


def _medan_quasicolor_weighted_residual_jacobian(params, x, data):
    """Exact Jacobian of the weighted Medan residual."""
    x = np.asarray(x, dtype=float)
    data = np.asarray(data, dtype=float)
    p = np.asarray(params, dtype=float)
    inv_sqrt = 1.0 / np.sqrt(data)
    J = np.empty((len(x), len(p)), dtype=float)
    for j, i in enumerate(range(0, len(p), 3)):
        amplitude, mean, sigma = p[i:i + 3]
        dx = x - mean
        e = np.exp(-0.5 * (dx / sigma) ** 2)
        g = amplitude * e
        J[:, 3 * j] = -e * inv_sqrt
        J[:, 3 * j + 1] = -(g * dx / sigma**2) * inv_sqrt
        J[:, 3 * j + 2] = -(g * dx**2 / sigma**3) * inv_sqrt
    return J


def _weighted_quantile_means(x, y, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    cdf = (np.cumsum(y) - 0.5 * y) / np.sum(y)
    return np.interp(probabilities, cdf, x)


def _deterministic_gaussian_start(spec, n_components, x_fit, y_fit,
                                  q_fit_min, q_fit_max):
    """Construct one fixed, data-derived initialization vector."""
    name, placement, width_factor = spec
    mode_center = float(x_fit[int(np.argmax(y_fit))])

    if placement == "original":
        return np.tile([1000.0, mode_center, 1.0], n_components)

    probs = (np.arange(n_components, dtype=float) + 0.5) / n_components
    if placement == "quantile_mid":
        means = _weighted_quantile_means(x_fit, y_fit, probs)
    elif placement == "uniform_mid":
        means = q_fit_min + probs * (q_fit_max - q_fit_min)
    else:
        raise ValueError(f"Unknown deterministic placement rule: {placement}")

    nominal_width = max(
        (q_fit_max - q_fit_min) / (2.0 * n_components),
        DMAG_BIN_WIDTH,
    )
    width = max(float(width_factor) * nominal_width, DMAG_BIN_WIDTH)
    sigmas = np.full(n_components, width, dtype=float)
    amplitudes = np.maximum(np.interp(means, x_fit, y_fit), 1.0)
    return np.column_stack([amplitudes, means, sigmas]).ravel()


def fit_medan_quasicolor_model_from_counts(measured, centers):
    """Fit Medan's quasi-color background model deterministically.

    For each component count n=2..10, seven fixed initial configurations are
    tried. The successful solution with minimum weighted least-squares cost is
    retained for that n. The final n is then selected with Medan's unweighted
    RSS-based BIC. This keeps the original fitting/model-selection distinction
    with seven fixed initializations and no random-number generator.
    """
    measured = np.asarray(measured, dtype=float)
    centers = np.asarray(centers, dtype=float)

    positive = np.isfinite(measured) & (measured > 0.0) & np.isfinite(centers)
    x_fit = centers[positive]
    y_fit = measured[positive]
    if len(y_fit) == 0:
        return None

    q_fit_min = float(np.min(x_fit))
    q_fit_max = float(np.max(x_fit))
    if not np.isfinite(q_fit_min) or not np.isfinite(q_fit_max) or q_fit_min >= q_fit_max:
        return None

    best = None
    for n_components in range(2, 11):
        # Require more occupied bins than free Gaussian parameters.
        if len(y_fit) <= 3 * n_components:
            continue

        lower = np.tile([0.0, q_fit_min, 1e-8], n_components)
        upper = np.tile([np.inf, q_fit_max, np.inf], n_components)

        result = None
        winning_name = None
        audit_indices = []
        for spec in DETERMINISTIC_START_SPECS:
            start_name = spec[0]
            start = _deterministic_gaussian_start(
                spec, n_components, x_fit, y_fit, q_fit_min, q_fit_max
            )
            audit = {
                "initialization": "deterministic_7start",
                "n_components": n_components,
                "start_name": start_name,
                "success": False,
            }
            try:
                trial = least_squares(
                    _medan_quasicolor_weighted_residual,
                    start,
                    jac=_medan_quasicolor_weighted_residual_jacobian,
                    args=(x_fit, y_fit),
                    bounds=(lower, upper),
                    x_scale="jac",
                    max_nfev=FIT_MAX_NFEV,
                )
                usable = (
                    trial.success
                    and np.all(np.isfinite(trial.x))
                    and np.isfinite(trial.cost)
                )
                audit.update(
                    success=bool(usable),
                    weighted_cost=float(trial.cost) if np.isfinite(trial.cost) else np.nan,
                    nfev=int(trial.nfev),
                    message=str(trial.message),
                )
                if usable:
                    residual_ss = float(np.sum(
                        (y_fit - _medan_quasicolor_model_from_vector(x_fit, trial.x)) ** 2
                    ))
                    audit["bic"] = float(
                        len(y_fit) * np.log(max(
                            residual_ss / len(y_fit), np.finfo(float).tiny
                        ))
                        + 3 * n_components * np.log(len(y_fit))
                    )
                    if result is None or trial.cost < result.cost:
                        result = trial
                        winning_name = start_name
            except Exception as exc:
                audit["message"] = f"{type(exc).__name__}: {exc}"
            FIT_AUDIT.append(audit)
            audit_indices.append(len(FIT_AUDIT) - 1)

        if result is None:
            continue

        for idx in audit_indices:
            FIT_AUDIT[idx]["selected_for_n"] = (
                FIT_AUDIT[idx].get("start_name") == winning_name
            )

        model_fit = _medan_quasicolor_model_from_vector(x_fit, result.x)
        model_all = _medan_quasicolor_model_from_vector(centers, result.x)
        if not np.all(np.isfinite(model_fit)) or not np.all(np.isfinite(model_all)):
            continue

        residual_ss = float(np.sum((y_fit - model_fit) ** 2))
        n_points = len(y_fit)
        mse = max(residual_ss / n_points, np.finfo(float).tiny)
        bic = n_points * np.log(mse) + (3 * n_components) * np.log(n_points)
        if not np.isfinite(bic):
            continue

        if best is None or bic < best["bic"]:
            model_counts = np.maximum(model_all, 0.0)
            model_total = float(np.sum(model_counts))
            outside_occupied = (centers < q_fit_min) | (centers > q_fit_max)
            mass_outside = (
                float(np.sum(model_counts[outside_occupied]) / model_total)
                if model_total > 0.0 else np.nan
            )
            best = {
                "counts": model_counts,
                "bic": float(bic),
                "n_components": int(n_components),
                "params": np.asarray(result.x, dtype=float),
                "residual_ss": residual_ss,
                "n_positive_bins": int(n_points),
                "optimizer_cost_weighted": float(result.cost),
                "optimizer_nfev": int(result.nfev),
                "selected_start": winning_name,
                "model_mass_outside_occupied_range": mass_outside,
            }

    return best

def fit_linear_scale_through_origin(model_values, data_values):
    """Fit Medan's scalar C with curve_fit(scale, model, observed)."""
    x = np.asarray(model_values, dtype=float)
    y = np.asarray(data_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0 or np.allclose(x, 0.0):
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                lambda model, C: C * model,
                x,
                y,
                p0=(1.0,),
                maxfev=20000
            )
    except Exception:
        return None

    C = float(popt[0])
    if not np.isfinite(C) or C <= 0.0:
        return None
    return C


def fit_displaced_h0_model(disp_dist, disp_dmag):
    """
    Build the modeled displaced-field frequency surface H0.

    This follows Medan's RELEASED back_mod_2d_gauss() construction:
      1. histogram the displaced sample in (theta, q);
      2. marginalize that measured 2-D histogram into Hang(theta), Hmag(q);
      3. fit a straight line to Hang after removing the outer search-edge bins;
      4. fit 2--10 Gaussian sums to positive Hmag bins with Pearson-like weights
         and select n with BIC = N ln(RSS/N) + 3n ln N;
      5. multiply the two fitted marginals;
      6. fit a scalar C to positive observed Hdis cells.

    Deliberate adaptations relative to Medan's released implementation are:
      * the exact angular edge trim: their source hard-codes a trimming expression
        tied to their query/grid setup, whereas this code excludes the outer
        ANGULAR_H0_EDGE_TRIM_ARCSEC;
      * Gaussian means are constrained to the occupied displaced-q interval of
        the stratum. These bounds constrain the centers, not the tails; Gaussian
        tails can extend outside the occupied interval.
    """
    disp_dist = np.asarray(disp_dist, dtype=float)
    disp_dmag = np.asarray(disp_dmag, dtype=float)

    valid = np.isfinite(disp_dist) & np.isfinite(disp_dmag)
    disp_dist = disp_dist[valid]
    disp_dmag = disp_dmag[valid]
    if len(disp_dist) == 0:
        return None

    Hdis, _, _ = np.histogram2d(
        disp_dist,
        disp_dmag,
        bins=[dist_bins, dmag_bins]
    )
    Hdis = Hdis.astype(float)

    dist_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    dmag_centers = 0.5 * (dmag_bins[:-1] + dmag_bins[1:])

    Hang = np.sum(Hdis, axis=1)
    Hmag = np.sum(Hdis, axis=0)

    # Medan removes the outer angular edge before fitting the line. With an
    # exact TAP cone, only the region near the hard 15-arcsec boundary needs to
    # be excluded. Require at least three bins for the linear fit.
    fit_max = MODEL_RADIUS_ARCSEC - ANGULAR_H0_EDGE_TRIM_ARCSEC
    angular_fit_mask = (
        np.isfinite(Hang) &
        np.isfinite(dist_centers) &
        (dist_centers <= fit_max)
    )
    if np.count_nonzero(angular_fit_mask) < 3:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt_ang, _ = curve_fit(
                lambda x, m, b: m * x + b,
                dist_centers[angular_fit_mask],
                Hang[angular_fit_mask],
                p0=(100000.0, 0.01),
                maxfev=20000
            )
    except Exception:
        return None

    angular_model = popt_ang[0] * dist_centers + popt_ang[1]
    if not np.all(np.isfinite(angular_model)):
        return None

    # A frequency model cannot be negative. Medan's released fit is unconstrained;
    # with sparse strata its intercept can fluctuate slightly below zero near
    # theta=0. Retain their linear fit but floor only those unphysical predicted
    # counts at zero. Record how many bins required this numerical/physical guard.
    angular_negative_bins = int(np.count_nonzero(angular_model < 0.0))
    angular_model = np.maximum(angular_model, 0.0)
    if np.sum(angular_model) <= 0.0:
        return None

    mag_fit = fit_medan_quasicolor_model_from_counts(Hmag, dmag_centers)
    if mag_fit is None:
        return None
    mag_model = mag_fit["counts"]

    Hbase = np.outer(angular_model, mag_model)

    # Released source fits C using only cells for which the observed displaced
    # histogram is nonzero.
    positive_cells = Hdis > 0.0
    C = fit_linear_scale_through_origin(
        Hbase[positive_cells],
        Hdis[positive_cells]
    )
    if C is None:
        return None

    Hmod = Hbase * C
    if not np.all(np.isfinite(Hmod)) or np.any(Hmod < 0.0):
        return None

    n_displaced_hist = float(np.sum(Hdis))
    if n_displaced_hist <= 0.0:
        return None
    hmod_total_over_ndisplaced = float(np.sum(Hmod) / n_displaced_hist)

    return {
        "Hdis_data": Hdis,
        "Hmod": Hmod,
        "angular_slope": float(popt_ang[0]),
        "angular_intercept": float(popt_ang[1]),
        "angular_fit_max_arcsec": float(fit_max),
        "angular_negative_bins_floored": angular_negative_bins,
        "background_C": float(C),
        "quasicolor_n_components": mag_fit["n_components"],
        "quasicolor_bic": mag_fit["bic"],
        "quasicolor_residual_ss": mag_fit["residual_ss"],
        "quasicolor_n_positive_bins": mag_fit["n_positive_bins"],
        "quasicolor_fit_params": mag_fit["params"],
        "quasicolor_model_mass_outside_occupied_range": mag_fit["model_mass_outside_occupied_range"],
        "Hmod_total_over_Ndisplaced": hmod_total_over_ndisplaced,
    }

def build_bayesian_model_2d(raw_valid, disp_valid):
    """
    Build one Medan-style 2-D posterior model in (theta, G_Gaia - m_WDS).
    """
    raw_dist = np.asarray(raw_valid["dist_arcsec"], dtype=float)
    disp_dist = np.asarray(disp_valid["dist_arcsec"], dtype=float)

    raw_dmag = (
        np.asarray(raw_valid["phot_g_mean_mag"], dtype=float) -
        np.asarray(raw_valid["wds_mag"], dtype=float)
    )
    disp_dmag = (
        np.asarray(disp_valid["phot_g_mean_mag"], dtype=float) -
        np.asarray(disp_valid["wds_mag"], dtype=float)
    )

    h0 = fit_displaced_h0_model(disp_dist, disp_dmag)
    if h0 is None:
        return None

    Htrue, _, _ = np.histogram2d(
        raw_dist,
        raw_dmag,
        bins=[dist_bins, dmag_bins]
    )

    scale_fit = find_optimal_scale_factor(Htrue, h0["Hmod"])
    if scale_fit is None:
        return None
    scale = scale_fit["scale_S"]

    # Same construction as the supplied Medan code.
    Htrue_smooth = medan_smooth(Htrue)
    Hdis_smooth = medan_smooth(h0["Hmod"] * scale)
    Hsub_smooth = Htrue_smooth - Hdis_smooth
    Hsub_smooth[Hsub_smooth < 0.0] = 0.0

    true_total = np.sum(Htrue_smooth)
    signal_total = np.sum(Hsub_smooth)
    background_total = np.sum(Hdis_smooth)

    if true_total <= 0.0 or signal_total <= 0.0 or background_total <= 0.0:
        return None

    P1 = signal_total / true_total
    if not np.isfinite(P1) or P1 < 0.0 or P1 > 1.0:
        return None
    P0 = 1.0 - P1

    f1 = Hsub_smooth / signal_total
    f0 = Hdis_smooth / background_total

    denominator = (f1 * P1) + (f0 * P0)
    # A cell with f1=f0=0 contains no support under either empirical hypothesis.
    # Its posterior is undefined, not zero. Keep it as NaN so scoring leaves any
    # candidate landing there explicitly unmodeled rather than turning 0/0 into
    # an artificial probability that is later clipped to 1e-12.
    bayes_2d = np.full_like(f1, np.nan, dtype=float)
    np.divide(
        f1 * P1,
        denominator,
        out=bayes_2d,
        where=denominator > 0.0
    )

    return {
        "mode": "2D",
        "f1": f1,
        "f0": f0,
        "bayes_2d": bayes_2d,
        "undefined_bayes_cells": int(np.count_nonzero(~np.isfinite(bayes_2d))),
        "prior_H1": float(P1),
        "prior_H0": float(P0),
        "scale_S": scale,
        "dist_bins": dist_bins,
        "mag_bins": dmag_bins,
        "N_true": len(raw_valid),
        "N_displaced": len(disp_valid),
        "N_signal": float(signal_total),
        "quasicolor_n_components": h0["quasicolor_n_components"],
        "quasicolor_bic": h0["quasicolor_bic"],
        "quasicolor_residual_ss": h0["quasicolor_residual_ss"],
        "quasicolor_n_positive_bins": h0["quasicolor_n_positive_bins"],
        "quasicolor_fit_params": h0["quasicolor_fit_params"],
        "quasicolor_model_mass_outside_occupied_range": h0["quasicolor_model_mass_outside_occupied_range"],
        "Hmod_total_over_Ndisplaced": h0["Hmod_total_over_Ndisplaced"],
        "angular_slope": h0["angular_slope"],
        "angular_intercept": h0["angular_intercept"],
        "angular_fit_max_arcsec": h0["angular_fit_max_arcsec"],
        "angular_negative_bins_floored": h0["angular_negative_bins_floored"],
        "background_C": h0["background_C"],
        "scale_fit_grid": scale_fit["scale_grid"],
        "scale_fit_negative_residual_sum": scale_fit["negative_residual_sum"],
        "scale_fit_broken_line_params": scale_fit["broken_line_params"],
        "Hdis_data": h0["Hdis_data"],
        "Hmod_background": h0["Hmod"],
        # Retain count-space arrays so the b-only missing-magnitude model can
        # combine already calibrated 2-D strata without re-fitting a 1-D S.
        "Htrue_smooth": Htrue_smooth,
        "Hdis_smooth": Hdis_smooth,
        "Hsub_smooth": Hsub_smooth,
    }



def build_bayesian_model_theta_only(raw_valid, disp_valid):
    """
    Build a 1-D Medan-style posterior model in angular separation theta.

    This model is calibrated independently of Gaia G. The RAW and DISPLACED
    training samples are selected only by WDS reference-magnitude and |b|
    strata; Gaia photometric availability is not part of the selection. This fallback
    assumes transfer to the missing-G population at fixed theta, m_WDS and |b|;
    its probability calibration depends on that transfer assumption.

    The displaced H0(theta) frequency model is the same edge-trimmed linear
    angular model used inside the 2-D construction. Because it is already fit
    directly to the 1-D displaced counts, no product-normalization constant C is
    required. Medan's scale-S elbow, percentile smoothing, H1 subtraction, and
    empirical-prior construction are then applied directly in one dimension.
    """
    raw_dist = np.asarray(raw_valid["dist_arcsec"], dtype=float)
    disp_dist = np.asarray(disp_valid["dist_arcsec"], dtype=float)

    raw_dist = raw_dist[np.isfinite(raw_dist)]
    disp_dist = disp_dist[np.isfinite(disp_dist)]
    if len(raw_dist) == 0 or len(disp_dist) == 0:
        return None

    Htrue, _ = np.histogram(raw_dist, bins=dist_bins)
    Hdis, _ = np.histogram(disp_dist, bins=dist_bins)
    Htrue = Htrue.astype(float)
    Hdis = Hdis.astype(float)

    dist_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    fit_max = MODEL_RADIUS_ARCSEC - ANGULAR_H0_EDGE_TRIM_ARCSEC
    angular_fit_mask = (
        np.isfinite(Hdis) &
        np.isfinite(dist_centers) &
        (dist_centers <= fit_max)
    )
    if np.count_nonzero(angular_fit_mask) < 3:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt_ang, _ = curve_fit(
                lambda x, m, b: m * x + b,
                dist_centers[angular_fit_mask],
                Hdis[angular_fit_mask],
                p0=(100000.0, 0.01),
                maxfev=20000,
            )
    except Exception:
        return None

    Hmod = popt_ang[0] * dist_centers + popt_ang[1]
    if not np.all(np.isfinite(Hmod)):
        return None

    angular_negative_bins = int(np.count_nonzero(Hmod < 0.0))
    Hmod = np.maximum(Hmod, 0.0)
    if np.sum(Hmod) <= 0.0:
        return None

    n_displaced_hist = float(np.sum(Hdis))
    if n_displaced_hist <= 0.0:
        return None
    hmod_total_over_ndisplaced = float(np.sum(Hmod) / n_displaced_hist)

    scale_fit = find_optimal_scale_factor(Htrue, Hmod)
    if scale_fit is None:
        return None
    scale = scale_fit["scale_S"]

    Htrue_smooth = medan_smooth(Htrue)
    Hdis_smooth = medan_smooth(Hmod * scale)
    Hsub_smooth = Htrue_smooth - Hdis_smooth
    Hsub_smooth[Hsub_smooth < 0.0] = 0.0

    true_total = float(np.sum(Htrue_smooth))
    signal_total = float(np.sum(Hsub_smooth))
    background_total = float(np.sum(Hdis_smooth))
    if true_total <= 0.0 or signal_total <= 0.0 or background_total <= 0.0:
        return None

    P1 = signal_total / true_total
    if not np.isfinite(P1) or P1 < 0.0 or P1 > 1.0:
        return None
    P0 = 1.0 - P1

    f1 = Hsub_smooth / signal_total
    f0 = Hdis_smooth / background_total
    denominator = (f1 * P1) + (f0 * P0)
    bayes_1d = np.full_like(f1, np.nan, dtype=float)
    np.divide(
        f1 * P1,
        denominator,
        out=bayes_1d,
        where=denominator > 0.0,
    )

    return {
        "mode": "1D_theta_magb_all_G",
        "f1": f1,
        "f0": f0,
        "bayes_1d": bayes_1d,
        "undefined_bayes_cells": int(np.count_nonzero(~np.isfinite(bayes_1d))),
        "prior_H1": float(P1),
        "prior_H0": float(P0),
        "scale_S": float(scale),
        "dist_bins": dist_bins,
        "N_true": int(len(raw_dist)),
        "N_displaced": int(len(disp_dist)),
        "N_signal": float(signal_total),
        "angular_slope": float(popt_ang[0]),
        "angular_intercept": float(popt_ang[1]),
        "angular_fit_max_arcsec": float(fit_max),
        "angular_negative_bins_floored": angular_negative_bins,
        "Hmod_total_over_Ndisplaced": hmod_total_over_ndisplaced,
        "scale_fit_grid": scale_fit["scale_grid"],
        "scale_fit_negative_residual_sum": scale_fit["negative_residual_sum"],
        "scale_fit_broken_line_params": scale_fit["broken_line_params"],
        "Hdis_data": Hdis,
        "Hmod_background": Hmod,
        "Htrue_smooth": Htrue_smooth,
        "Hdis_smooth": Hdis_smooth,
        "Hsub_smooth": Hsub_smooth,
    }

def get_b_key(b_abs):
    """Return the WDS |b| stratum containing b_abs."""
    if not np.isfinite(b_abs):
        return None

    for b_min, b_max in b_bins:
        if b_min <= b_abs < b_max:
            return (b_min, b_max)
    return None


def get_wds_mag_key(wds_mag):
    """Return the WDS reference-magnitude stratum containing wds_mag."""
    if not np.isfinite(wds_mag) or wds_mag == 1234.0:
        return None

    for mag_min, mag_max in wds_mag_strata:
        if mag_min <= wds_mag < mag_max:
            return (mag_min, mag_max)
    return None


def build_b_only_astrometric_models(models_theta):
    """
    Build one theta-only model per WDS |b| stratum for missing m_WDS.

    Within each latitude bin, combine the six independently calibrated
    theta-only WDS-magnitude strata in count space. Because those source models
    were themselves trained without conditioning on Gaia G availability, this
    mixture also does not inherit a finite-G selection condition.

    The remaining explicit assumption is that, at fixed |b|, the reference-
    magnitude mixture represented by targets with measured m_WDS is applicable
    to targets whose WDS magnitude is missing. The model is disabled unless all
    six calibrated magnitude strata are available.
    """
    models_1d_b = {}

    for b_key in b_bins:
        signal_theta = np.zeros(len(dist_bins) - 1, dtype=float)
        background_theta = np.zeros(len(dist_bins) - 1, dtype=float)
        true_total = 0.0
        source_scales = []
        n_source_strata = 0

        for mag_key in wds_mag_strata:
            model = models_theta.get((*b_key, *mag_key))
            if model is None:
                continue

            signal_theta += model["Hsub_smooth"]
            background_theta += model["Hdis_smooth"]
            true_total += float(np.sum(model["Htrue_smooth"]))
            source_scales.append(model["scale_S"])
            n_source_strata += 1

        signal_total = float(np.sum(signal_theta))
        background_total = float(np.sum(background_theta))

        if n_source_strata != len(wds_mag_strata):
            models_1d_b[b_key] = None
            print(
                f"  Unmodeled astrometry-only latitude stratum: |b|={b_key} "
                f"({n_source_strata}/{len(wds_mag_strata)} calibrated theta-only "
                "magnitude strata; all are required for the missing-m_WDS mixture)"
            )
            continue

        if true_total <= 0.0 or signal_total <= 0.0 or background_total <= 0.0:
            models_1d_b[b_key] = None
            continue

        P1 = signal_total / true_total
        if not np.isfinite(P1) or P1 < 0.0 or P1 > 1.0:
            models_1d_b[b_key] = None
            continue
        P0 = 1.0 - P1

        f1_dist = signal_theta / signal_total
        f0_dist = background_theta / background_total
        denominator = (f1_dist * P1) + (f0_dist * P0)
        bayes_1d = np.full_like(f1_dist, np.nan, dtype=float)
        np.divide(
            f1_dist * P1,
            denominator,
            out=bayes_1d,
            where=denominator > 0.0,
        )

        models_1d_b[b_key] = {
            "mode": "1D_b_theta_mixture",
            "f1": f1_dist,
            "f0": f0_dist,
            "bayes_1d": bayes_1d,
            "undefined_bayes_cells": int(np.count_nonzero(~np.isfinite(bayes_1d))),
            "prior_H1": float(P1),
            "prior_H0": float(P0),
            # No single S exists: this is a count-space mixture of six
            # independently calibrated theta-only magnitude strata.
            "scale_S": np.nan,
            "source_scale_min": float(np.min(source_scales)),
            "source_scale_max": float(np.max(source_scales)),
            "source_scale_median": float(np.median(source_scales)),
            "N_source_strata": int(n_source_strata),
            "dist_bins": dist_bins,
            "N_signal": float(signal_total),
        }

    return models_1d_b

def build_empirical_bayesian_models(raw_table, disp_table):
    """
    Build the empirical Bayesian calibration bundle.

    2-D models (18):
        condition on (WDS magnitude stratum, WDS |b| stratum), require finite
        Gaia G, and evaluate (theta, G_Gaia - m_WDS).

    Theta-only models (18):
        use the same (m_WDS, |b|) strata but do NOT condition on Gaia G being
        finite. They are independently calibrated in theta and are used when
        m_WDS is known but Gaia G is missing.

    |b|-only models (3):
        used when m_WDS itself is unavailable; each is a count-space mixture of
        all six calibrated theta-only magnitude strata at that |b|.
    """
    print("\nBuilding Medan-style empirical Bayesian models...")

    raw_df = raw_table.to_pandas() if isinstance(raw_table, Table) else raw_table.copy()
    disp_df = disp_table.to_pandas() if isinstance(disp_table, Table) else disp_table.copy()

    raw_wds_mag = pd.to_numeric(raw_df["wds_mag"], errors="coerce").to_numpy(float)
    disp_wds_mag = pd.to_numeric(disp_df["wds_mag"], errors="coerce").to_numpy(float)
    raw_g = pd.to_numeric(raw_df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    disp_g = pd.to_numeric(disp_df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    raw_b = np.abs(pd.to_numeric(raw_df["gal_b"], errors="coerce").to_numpy(float))
    disp_b = np.abs(pd.to_numeric(disp_df["gal_b"], errors="coerce").to_numpy(float))

    models_2d = {}
    models_theta = {}

    for b_min, b_max in b_bins:
        for mag_min, mag_max in wds_mag_strata:
            key = (b_min, b_max, mag_min, mag_max)

            # Common WDS conditioning for the 2-D and theta-only models.
            raw_base_mask = (
                (raw_b >= b_min) &
                (raw_b < b_max) &
                np.isfinite(raw_wds_mag) &
                (raw_wds_mag != 1234.0) &
                (raw_wds_mag >= mag_min) &
                (raw_wds_mag < mag_max)
            )
            disp_base_mask = (
                (disp_b >= b_min) &
                (disp_b < b_max) &
                np.isfinite(disp_wds_mag) &
                (disp_wds_mag != 1234.0) &
                (disp_wds_mag >= mag_min) &
                (disp_wds_mag < mag_max)
            )

            # Full 2-D q model necessarily requires finite Gaia G.
            raw_2d = raw_df.loc[raw_base_mask & np.isfinite(raw_g)].copy()
            disp_2d = disp_df.loc[disp_base_mask & np.isfinite(disp_g)].copy()

            model_2d = None
            if len(raw_2d) > 0 and len(disp_2d) > 0:
                audit_start = len(FIT_AUDIT)
                model_2d = build_bayesian_model_2d(raw_2d, disp_2d)
                for audit in FIT_AUDIT[audit_start:]:
                    audit.update(b_min=b_min, b_max=b_max, mag_min=mag_min, mag_max=mag_max)

            if model_2d is None:
                print(
                    "  Unmodeled 2-D stratum: "
                    f"|b|=[{b_min},{b_max}), "
                    f"m_WDS=[{mag_min},{mag_max}) "
                    f"(raw={len(raw_2d)}, disp={len(disp_2d)}; "
                    "Medan scale/background model not identifiable)"
                )
            models_2d[key] = model_2d

            # Independent theta-only calibration. Gaia G availability is not part
            # of the training selection because theta itself does not require G.
            raw_theta = raw_df.loc[raw_base_mask].copy()
            disp_theta = disp_df.loc[disp_base_mask].copy()

            model_theta = None
            if len(raw_theta) > 0 and len(disp_theta) > 0:
                model_theta = build_bayesian_model_theta_only(raw_theta, disp_theta)

            if model_theta is None:
                print(
                    "  Unmodeled theta-only stratum: "
                    f"|b|=[{b_min},{b_max}), "
                    f"m_WDS=[{mag_min},{mag_max}) "
                    f"(raw={len(raw_theta)}, disp={len(disp_theta)}; "
                    "1-D scale/background model not identifiable)"
                )
            models_theta[key] = model_theta

    valid_2d = sum(model is not None for model in models_2d.values())
    valid_theta = sum(model is not None for model in models_theta.values())
    print(f"Generated {valid_2d}/18 valid 2-D probability strata.")
    print(f"Generated {valid_theta}/18 valid theta-only probability strata.")

    models_1d_b = build_b_only_astrometric_models(models_theta)
    valid_1d = sum(model is not None for model in models_1d_b.values())
    print(f"Generated {valid_1d}/3 valid astrometry-only latitude strata.")

    return {
        "models_2d": models_2d,
        "models_theta": models_theta,
        "models_1d_b": models_1d_b,
    }


# -----------------------------------------------------------------------------
# Candidate scoring
# -----------------------------------------------------------------------------
def get_hist_bin(value, bins):
    """Return a valid histogram-bin index or None when outside the support."""
    if not np.isfinite(value):
        return None

    idx = np.digitize(value, bins) - 1

    # np.histogram includes the exact rightmost edge in the final bin whereas
    # np.digitize normally places it one index beyond; mirror histogram behavior.
    if idx == len(bins) - 1 and np.isclose(value, bins[-1]):
        idx = len(bins) - 2

    if idx < 0 or idx >= len(bins) - 1:
        return None

    return int(idx)


def score_candidates(raw_table, model_bundle):
    """
    Evaluate every RAW candidate with a Medan-style pairwise posterior.

    Cases
    -----
    1. m_WDS and Gaia G available:
       use the full 2-D (theta, G-m_WDS) model in the known (m_WDS, |b|) stratum.
    2. m_WDS available but Gaia G missing:
       use the independently calibrated theta-only model for the same
       (m_WDS, |b|) stratum. Its training sample does not condition on finite G.
    3. m_WDS missing:
       use the |b|-only complete-six-stratum theta-model mixture, if available.

    Finite-G quasi-colors must lie inside the frozen production support.
    Cells with zero support under both H1 and H0 remain unmodeled (NaN).
    """
    raw_df = raw_table.to_pandas().copy() if isinstance(raw_table, Table) else raw_table.copy()

    models_2d = model_bundle["models_2d"]
    models_theta = model_bundle["models_theta"]
    models_1d_b = model_bundle["models_1d_b"]

    raw_df["quasi_color"] = np.nan
    raw_df["posterior_prob"] = np.nan
    raw_df["log_odds"] = np.nan
    raw_df["bayes_model"] = "unmodeled"
    raw_df["model_prior_H1"] = np.nan
    raw_df["model_prior_H0"] = np.nan
    raw_df["scale_S"] = np.nan
    raw_df["assignment_eligible"] = (
        pd.to_numeric(raw_df["dist_arcsec"], errors="coerce") <= ASSIGNMENT_RADIUS_ARCSEC
    )

    for idx, row in raw_df.iterrows():
        b_abs = abs(float(row["gal_b"])) if np.isfinite(row["gal_b"]) else np.nan
        b_key = get_b_key(b_abs)
        if b_key is None:
            continue

        dist_idx = get_hist_bin(float(row["dist_arcsec"]), dist_bins)
        if dist_idx is None:
            continue

        try:
            wds_mag = float(row.get("wds_mag", np.nan))
        except (TypeError, ValueError):
            wds_mag = np.nan

        try:
            g_mag = float(row.get("phot_g_mean_mag", np.nan))
        except (TypeError, ValueError):
            g_mag = np.nan

        has_wds_mag = np.isfinite(wds_mag) and wds_mag != 1234.0
        has_gaia_g = np.isfinite(g_mag)

        if has_wds_mag:
            mag_key = get_wds_mag_key(wds_mag)
            if mag_key is None:
                continue

            if has_gaia_g:
                model_2d = models_2d.get((*b_key, *mag_key))
                if model_2d is None:
                    continue

                q = g_mag - wds_mag
                raw_df.at[idx, "quasi_color"] = q

                mag_idx = get_hist_bin(q, dmag_bins)
                if mag_idx is None:
                    # The fixed production support is an explicit model boundary.
                    # Never hide a support error with a theta-only fallback.
                    raise RuntimeError(
                        f"Candidate q={q:.6f} lies outside the configured quasi-color "
                        f"support [{dmag_bins[0]}, {dmag_bins[-1]}]."
                    )

                posterior = model_2d["bayes_2d"][dist_idx, mag_idx]
                prior_H1 = model_2d["prior_H1"]
                prior_H0 = model_2d["prior_H0"]
                scale_S = model_2d["scale_S"]
                model_name = "2D_WDSref"
            else:
                model_theta = models_theta.get((*b_key, *mag_key))
                if model_theta is None:
                    continue

                posterior = model_theta["bayes_1d"][dist_idx]
                prior_H1 = model_theta["prior_H1"]
                prior_H0 = model_theta["prior_H0"]
                scale_S = model_theta["scale_S"]
                model_name = "1D_magb_missing_G_theta_calibrated"

        else:
            # Without m_WDS the reference-magnitude stratum is unknown. Use the
            # complete six-stratum |b|-only mixture, if it was constructible.
            model_1d = models_1d_b.get(b_key)
            if model_1d is None:
                continue

            posterior = model_1d["bayes_1d"][dist_idx]
            prior_H1 = model_1d["prior_H1"]
            prior_H0 = model_1d["prior_H0"]
            scale_S = model_1d["scale_S"]
            model_name = (
                "1D_b_missing_WDS_mag"
                if has_gaia_g else
                "1D_b_missing_WDS_and_G"
            )

        # A NaN posterior denotes a bin with no empirical support under either
        # H1 or H0. Leave the candidate explicitly unmodeled.
        if not np.isfinite(posterior):
            continue

        posterior = float(posterior)
        utility_posterior = float(np.clip(posterior, 1e-12, 1.0 - 1e-12))
        log_odds = np.log(utility_posterior) - np.log1p(-utility_posterior)

        # Keep the empirical posterior itself unchanged (including exact 0 or 1).
        # Only the assignment utility is numerically regularized to avoid infinite
        # log-odds at the boundaries.
        raw_df.at[idx, "posterior_prob"] = posterior
        raw_df.at[idx, "log_odds"] = log_odds
        raw_df.at[idx, "bayes_model"] = model_name
        raw_df.at[idx, "model_prior_H1"] = prior_H1
        raw_df.at[idx, "model_prior_H0"] = prior_H0
        raw_df.at[idx, "scale_S"] = scale_S

    return raw_df


# -----------------------------------------------------------------------------
# Global bipartite assignment
# -----------------------------------------------------------------------------
def build_graph_components(scored_df):
    """
    Find connected components of the WDS-component/Gaia candidate graph.

    Only non-negative-utility edges (log_odds >= 0, equivalently posterior >= 0.5)
    are needed for the assignment graph. Negative-utility edges can never improve
    the objective because every WDS component and Gaia source has a zero-cost dummy
    orphan option. Removing them therefore leaves the optimum unchanged while
    preventing irrelevant weak edges from joining otherwise independent components.

    Returns both the connected components and a per-WDS edge lookup so the full
    scored DataFrame never has to be rescanned inside the component loop.
    """
    w_to_g = {}
    g_to_w = {}
    edges_by_w = {}

    for row in scored_df.itertuples(index=False):
        if (
            not np.isfinite(row.posterior_prob)
            or not bool(row.assignment_eligible)
            or not np.isfinite(row.log_odds)
            or float(row.log_odds) < 0.0
        ):
            continue

        w = int(row.upload_id)
        g = int(row.source_id)
        row_id = int(row.row_id)
        log_odds = float(row.log_odds)

        w_to_g.setdefault(w, set()).add(g)
        g_to_w.setdefault(g, set()).add(w)
        edges_by_w.setdefault(w, []).append((g, log_odds, row_id))

    visited_w = set()
    visited_g = set()
    components = []

    for start_w in sorted(w_to_g):
        if start_w in visited_w:
            continue

        stack = [("w", start_w)]
        component_w = set()
        component_g = set()

        while stack:
            node_type, node = stack.pop()

            if node_type == "w":
                if node in visited_w:
                    continue
                visited_w.add(node)
                component_w.add(node)

                for g in w_to_g.get(node, ()):
                    if g not in visited_g:
                        stack.append(("g", g))
            else:
                if node in visited_g:
                    continue
                visited_g.add(node)
                component_g.add(node)

                for w in g_to_w.get(node, ()):
                    if w not in visited_w:
                        stack.append(("w", w))

        components.append((component_w, component_g))

    return components, edges_by_w


def assignment_margins(cost, M, N, pair_to_row, row_ind, col_ind):
    """Return edge exclusion margins and the runner-up gap on retained real edges.

    Dummy permutations never define a different association. Alternatives may
    orphan a target. Margins use clipped pairwise log-odds, not probabilities.
    The graph excludes negative edges, so these are margins within that retained
    graph, not runner-up gaps over discarded negative-utility associations.
    """
    optimum = -float(cost[row_ind, col_ind].sum())
    selected = {(int(r), int(c)) for r, c in zip(row_ind, col_ind)
                if r < M and c < N and (r, c) in pair_to_row}
    edge_margins = {}
    for r, c in sorted(selected):
        saved = cost[r, c]
        cost[r, c] = np.inf
        try:
            ar, ac = linear_sum_assignment(cost)
            alternative = -float(cost[ar, ac].sum())
        finally:
            cost[r, c] = saved
        gap = optimum - alternative
        if gap < -1e-8 * max(1.0, abs(optimum)):
            raise RuntimeError("Alternative assignment exceeds reported optimum.")
        edge_margins[pair_to_row[(r, c)]] = max(0.0, gap)
    group_gap = min(edge_margins.values(), default=np.inf)
    # A strict superset of the selected real edges can differ only by adding
    # zero-utility edges between currently orphaned nodes. Exclusion alone would
    # miss that kind of tie (including an all-orphan optimum).
    used_w = {r for r, _ in selected}
    used_g = {c for _, c in selected}
    if any(r not in used_w and c not in used_g and cost[r, c] == 0.0
           for r, c in pair_to_row):
        group_gap = 0.0
    return optimum, group_gap, edge_margins


def optimize_global_assignments(scored_df, wds_df):
    """
    Solve the global one-to-one WDS-component/Gaia assignment problem.

    Candidate edges are indexed once. Each connected component is solved
    independently with assignment margins, and final output uses vectorized joins.
    """
    print("\nResolving global one-to-one assignments (Budavari & Basu)...")
    t0 = time.time()

    scored_df = scored_df.copy()
    wds_df = wds_df.copy()

    # Require integer identifiers for graph construction and output joins.
    for col in ("upload_id", "source_id", "row_id"):
        scored_df[col] = pd.to_numeric(scored_df[col], errors="raise").astype(np.int64)

    if scored_df["assignment_eligible"].dtype != bool:
        scored_df["assignment_eligible"] = (
            scored_df["assignment_eligible"].astype(str).str.lower().eq("true")
        )

    # One O(E) lookup used everywhere below. row_id is unique in the scored file.
    scored_by_row = scored_df.set_index("row_id", drop=False)

    components, edges_by_w = build_graph_components(scored_df)
    if components:
        # Select one actual component by padded matrix size, breaking size ties
        # by edge count. All reported counts belong to that same component.
        largest_id, (largest_w_nodes, largest_g_nodes) = max(
            enumerate(components),
            key=lambda item: (
                len(item[1][0]) + len(item[1][1]),
                sum(len(edges_by_w[w]) for w in item[1][0]),
            ),
        )
        largest_w = len(largest_w_nodes)
        largest_g = len(largest_g_nodes)
        largest_edges = sum(len(edges_by_w[w]) for w in largest_w_nodes)
        print(
            f"Assignment graph: {len(components)} connected components; "
            f"largest by padded matrix size is group {largest_id}: "
            f"{largest_w} WDS / {largest_g} Gaia / {largest_edges} edges "
            f"(padded matrix size {largest_w + largest_g})."
        )
    else:
        print("Assignment graph contains no non-negative-utility candidate edges.")

    assigned_source = {}
    assigned_row = {}
    group_diagnostics = {}
    edge_margin_by_w = {}

    # Highest-posterior individual eligible candidate is retained regardless of
    # whether posterior < 0.5, so it remains useful as a diagnostic suggestion.
    valid_candidates = scored_df[
        np.isfinite(scored_df["posterior_prob"]) &
        scored_df["assignment_eligible"].astype(bool)
    ]

    if not valid_candidates.empty:
        best_idx = valid_candidates.groupby("upload_id", sort=False)["posterior_prob"].idxmax()
        best_summary = valid_candidates.loc[
            best_idx,
            ["upload_id", "source_id", "posterior_prob", "dist_arcsec", "quasi_color", "bayes_model"]
        ].copy()
        best_summary.rename(columns={
            "source_id": "best_source_id",
            "posterior_prob": "best_posterior_prob",
            "dist_arcsec": "best_dist_arcsec",
            "quasi_color": "best_quasi_color",
            "bayes_model": "best_bayes_model",
        }, inplace=True)
        best_summary["best_source_id"] = best_summary["best_source_id"].astype("Int64")
    else:
        best_summary = pd.DataFrame(columns=[
            "upload_id", "best_source_id", "best_posterior_prob", "best_dist_arcsec",
            "best_quasi_color", "best_bayes_model"
        ])
        best_summary["best_source_id"] = best_summary["best_source_id"].astype("Int64")

    # Solve each independent connected component using only pre-indexed edges.
    for group_id, (component_w, component_g) in enumerate(components):
        w_nodes = sorted(component_w)
        g_nodes = sorted(component_g)
        M = len(w_nodes)
        N = len(g_nodes)

        size = M + N
        cost = np.zeros((size, size), dtype=float)
        cost[:M, :N] = 1e6

        w_index = {w: i for i, w in enumerate(w_nodes)}
        g_index = {g: j for j, g in enumerate(g_nodes)}
        pair_to_row = {}

        for w in w_nodes:
            i = w_index[w]
            for g, log_odds, row_id in edges_by_w.get(w, ()):
                j = g_index.get(g)
                if j is None:
                    continue
                cost[i, j] = -log_odds
                pair_to_row[(i, j)] = row_id

        row_ind, col_ind = linear_sum_assignment(cost)
        optimum, group_gap, edge_gaps = assignment_margins(
            cost, M, N, pair_to_row, row_ind, col_ind)
        for w in w_nodes:
            group_diagnostics[w] = {
                "assignment_group_id": group_id,
                "assignment_group_n_wds": M, "assignment_group_n_gaia": N,
                "assignment_group_n_edges": len(pair_to_row),
                "assignment_group_objective": optimum,
                "assignment_group_margin": group_gap,
            }

        for r, c in zip(row_ind, col_ind):
            if r >= M or c >= N:
                continue
            if cost[r, c] >= 1e5:
                continue

            row_id = pair_to_row[(r, c)]
            row = scored_by_row.loc[row_id]
            w = int(row["upload_id"])
            g = int(row["source_id"])
            assigned_source[w] = g
            assigned_row[w] = row_id
            edge_margin_by_w[w] = edge_gaps.get(row_id, np.nan)

    print(
        f"Hungarian/component stage finished in {time.time() - t0:.1f} s; "
        f"assembling final catalog..."
    )

    # -------------------------------------------------------------------------
    # Build one output row per WDS component with vectorized joins.
    # -------------------------------------------------------------------------
    wds_cols_ordered = [
        "wds_id", "discoverer", "component", "first_obs", "last_obs",
        "observations", "position_angle", "separation", "mag_primary",
        "mag_secondary", "spectral_type", "pmra_primary", "pmdec_primary",
        "pmra_secondary", "pmdec_secondary", "notes", "ra_hh", "ra_mm",
        "ra_ss", "ra", "dec_sign", "dec_deg", "dec_mm", "dec_ss", "dec",
        "component_type", "upload_id", "ra_j2016", "dec_j2016", "wds_mag", "gal_b"
    ]

    final_df = wds_df[wds_cols_ordered].copy()
    final_df = final_df.merge(best_summary, on="upload_id", how="left", sort=False)
    final_df["best_bayes_model"] = final_df["best_bayes_model"].fillna("none")
    final_df["best_source_id"] = final_df["best_source_id"].astype("Int64")

    wds_side_candidate_columns = {
        "upload_id", "wds_id", "component", "wds_mag", "gal_b", "row_id"
    }
    gaia_columns = [
        col for col in scored_df.columns
        if col not in wds_side_candidate_columns
    ]

    if assigned_row:
        assigned_ids = list(assigned_row.values())
        assigned_rows_df = scored_by_row.loc[assigned_ids, ["upload_id"] + gaia_columns].copy()

        # IMPORTANT: convert integer-valued Gaia columns to Pandas nullable Int64
        # *before* the left merge.  A left merge from an int64 column onto the full
        # WDS table introduces missing values for unassigned components; Pandas then
        # promotes ordinary int64 to float64.  Gaia source_id values exceed 2**53, so
        # that promotion irreversibly rounds many 19-digit IDs.  Converting only
        # after the merge is too late.  Nullable Int64 preserves exact integers while
        # still allowing <NA> for unassigned rows.
        integer_gaia_columns = []
        for col in gaia_columns:
            if pd.api.types.is_integer_dtype(scored_df[col].dtype):
                assigned_rows_df[col] = assigned_rows_df[col].astype("Int64")
                integer_gaia_columns.append(col)

        assigned_rows_df.rename(
            columns={col: f"assigned_{col}" for col in gaia_columns},
            inplace=True
        )
        final_df = final_df.merge(assigned_rows_df, on="upload_id", how="left", sort=False)

        # Defensive dtype assertion after the merge.  This must never route the
        # values through float64; doing so would destroy exact Gaia identifiers.
        for col in integer_gaia_columns:
            out_col = f"assigned_{col}"
            final_df[out_col] = final_df[out_col].astype("Int64")
    else:
        for col in gaia_columns:
            out_col = f"assigned_{col}"
            if pd.api.types.is_integer_dtype(scored_df[col].dtype):
                final_df[out_col] = pd.Series(pd.array([pd.NA] * len(final_df), dtype="Int64"))
            else:
                final_df[out_col] = np.nan

    assigned_w = set(assigned_row)
    has_best = final_df["best_source_id"].notna()
    is_assigned = final_df["upload_id"].isin(assigned_w)
    final_df["assignment_status"] = np.select(
        [is_assigned, has_best],
        ["assigned_global", "not_assigned_global"],
        default="no_modeled_candidate"
    )
    for col in ("assignment_group_id", "assignment_group_n_wds",
                "assignment_group_n_gaia", "assignment_group_n_edges",
                "assignment_group_objective", "assignment_group_margin"):
        values = {w: d[col] for w, d in group_diagnostics.items()}
        final_df[col] = final_df["upload_id"].map(values)
        if col in ("assignment_group_id", "assignment_group_n_wds",
                   "assignment_group_n_gaia", "assignment_group_n_edges"):
            final_df[col] = final_df[col].astype("Int64")
    final_df["assigned_edge_margin"] = final_df["upload_id"].map(edge_margin_by_w)
    print(f"Final catalog assembly finished in {time.time() - t0:.1f} s total.")
    return final_df


# -----------------------------------------------------------------------------
# Model diagnostics / reproducibility products
# -----------------------------------------------------------------------------
def save_model_diagnostics(model_bundle, model_summary_path, model_arrays_path):
    """Save per-stratum scalar diagnostics and full model arrays."""
    models_2d = model_bundle["models_2d"]
    models_theta = model_bundle["models_theta"]
    models_1d_b = model_bundle["models_1d_b"]

    summary_rows = []
    arrays = {
        "dist_bins": np.asarray(dist_bins, dtype=float),
        "dmag_bins": np.asarray(dmag_bins, dtype=float),
    }

    for key, model in models_2d.items():
        b_min, b_max, mag_min, mag_max = key
        row = {
            "model_type": "2D",
            "b_min": b_min,
            "b_max": b_max,
            "mag_min": mag_min,
            "mag_max": mag_max,
            "valid": model is not None,
        }

        tag = (
            f"b_{b_min:g}_{b_max:g}__m_"
            f"{str(mag_min).replace('-', 'neg').replace('.', 'p')}_"
            f"{str(mag_max).replace('-', 'neg').replace('.', 'p')}"
        )

        if model is not None:
            row.update({
                "N_true": model["N_true"],
                "N_displaced": model["N_displaced"],
                "N_signal": model["N_signal"],
                "prior_H1": model["prior_H1"],
                "prior_H0": model["prior_H0"],
                "scale_S": model["scale_S"],
                "background_C": model["background_C"],
                "angular_slope": model["angular_slope"],
                "angular_intercept": model["angular_intercept"],
                "angular_fit_max_arcsec": model["angular_fit_max_arcsec"],
                "angular_negative_bins_floored": model["angular_negative_bins_floored"],
                "quasicolor_n_components": model["quasicolor_n_components"],
                "quasicolor_bic": model["quasicolor_bic"],
                "quasicolor_residual_ss": model["quasicolor_residual_ss"],
                "quasicolor_n_positive_bins": model["quasicolor_n_positive_bins"],
                "quasicolor_model_mass_outside_occupied_range": model["quasicolor_model_mass_outside_occupied_range"],
                "Hmod_total_over_Ndisplaced": model["Hmod_total_over_Ndisplaced"],
                "undefined_bayes_cells": model["undefined_bayes_cells"],
            })
            arrays[f"{tag}__bayes_2d"] = model["bayes_2d"]
            arrays[f"{tag}__Htrue_smooth"] = model["Htrue_smooth"]
            arrays[f"{tag}__Hdis_smooth"] = model["Hdis_smooth"]
            arrays[f"{tag}__Hsub_smooth"] = model["Hsub_smooth"]
            arrays[f"{tag}__Hdis_data"] = model["Hdis_data"]
            arrays[f"{tag}__Hmod_background"] = model["Hmod_background"]
            arrays[f"{tag}__quasicolor_fit_params"] = model["quasicolor_fit_params"]
            arrays[f"{tag}__scale_grid"] = model["scale_fit_grid"]
            arrays[f"{tag}__negative_residual_sum"] = model["scale_fit_negative_residual_sum"]
            arrays[f"{tag}__broken_line_params"] = model["scale_fit_broken_line_params"]

        summary_rows.append(row)

    for key, model in models_theta.items():
        b_min, b_max, mag_min, mag_max = key
        row = {
            "model_type": "1D_theta_magb_all_G",
            "b_min": b_min,
            "b_max": b_max,
            "mag_min": mag_min,
            "mag_max": mag_max,
            "valid": model is not None,
        }

        tag = (
            f"b_{b_min:g}_{b_max:g}__m_"
            f"{str(mag_min).replace('-', 'neg').replace('.', 'p')}_"
            f"{str(mag_max).replace('-', 'neg').replace('.', 'p')}__theta_all_G"
        )

        if model is not None:
            row.update({
                "N_true": model["N_true"],
                "N_displaced": model["N_displaced"],
                "N_signal": model["N_signal"],
                "prior_H1": model["prior_H1"],
                "prior_H0": model["prior_H0"],
                "scale_S": model["scale_S"],
                "angular_slope": model["angular_slope"],
                "angular_intercept": model["angular_intercept"],
                "angular_fit_max_arcsec": model["angular_fit_max_arcsec"],
                "angular_negative_bins_floored": model["angular_negative_bins_floored"],
                "Hmod_total_over_Ndisplaced": model["Hmod_total_over_Ndisplaced"],
                "undefined_bayes_cells": model["undefined_bayes_cells"],
            })
            arrays[f"{tag}__bayes_1d"] = model["bayes_1d"]
            arrays[f"{tag}__Htrue_smooth"] = model["Htrue_smooth"]
            arrays[f"{tag}__Hdis_smooth"] = model["Hdis_smooth"]
            arrays[f"{tag}__Hsub_smooth"] = model["Hsub_smooth"]
            arrays[f"{tag}__Hdis_data"] = model["Hdis_data"]
            arrays[f"{tag}__Hmod_background"] = model["Hmod_background"]
            arrays[f"{tag}__scale_grid"] = model["scale_fit_grid"]
            arrays[f"{tag}__negative_residual_sum"] = model["scale_fit_negative_residual_sum"]
            arrays[f"{tag}__broken_line_params"] = model["scale_fit_broken_line_params"]

        summary_rows.append(row)

    for b_key, model in models_1d_b.items():
        b_min, b_max = b_key
        row = {
            "model_type": "1D_missing_WDS_mag",
            "b_min": b_min,
            "b_max": b_max,
            "mag_min": np.nan,
            "mag_max": np.nan,
            "valid": model is not None,
        }
        if model is not None:
            row.update({
                "N_signal": model["N_signal"],
                "prior_H1": model["prior_H1"],
                "prior_H0": model["prior_H0"],
                "scale_S": np.nan,
                "source_scale_min": model["source_scale_min"],
                "source_scale_max": model["source_scale_max"],
                "source_scale_median": model["source_scale_median"],
                "N_source_strata": model["N_source_strata"],
                "undefined_bayes_cells": model["undefined_bayes_cells"],
            })
            tag = f"b_{b_min:g}_{b_max:g}__missing_wds_mag"
            arrays[f"{tag}__bayes_1d"] = model["bayes_1d"]
        summary_rows.append(row)

    for path in (model_summary_path, model_arrays_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(model_summary_path, index=False)
    np.savez_compressed(model_arrays_path, **arrays)

    print(f"Saved model summary to: {model_summary_path}")
    print(f"Saved model arrays to:  {model_arrays_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _check_candidate_pair_uniqueness(scored_candidates):
    """Abort rather than silently repair unexpected duplicate WDS--Gaia edges."""
    duplicated = scored_candidates.duplicated(
        ["upload_id", "source_id"], keep=False
    )
    if duplicated.any():
        n_rows = int(duplicated.sum())
        n_pairs = int(
            scored_candidates.loc[duplicated, ["upload_id", "source_id"]]
            .drop_duplicates()
            .shape[0]
        )
        raise RuntimeError(
            "Gaia retrieval/scoring produced duplicated WDS--Gaia candidate pairs: "
            f"{n_pairs} duplicated pairs involving {n_rows} rows. "
            "The production pipeline does not silently deduplicate them."
        )



def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Empirical-Bayesian WDS component crossmatch to Gaia DR3."
    )
    parser.add_argument(
        "--backend",
        choices=("ari", "esa"),
        default=DEFAULT_GAIA_BACKEND,
        help="Gaia TAP backend (default: ari).",
    )
    parser.add_argument(
        "--components",
        default=DEFAULT_COMPONENTS_FILE,
        help="Input WDS component summary file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Final one-row-per-WDS-component output CSV.",
    )
    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATE_OUTPUT_FILE,
        help="Scored 5-arcsec candidate-edge CSV.",
    )
    parser.add_argument(
        "--model-summary",
        default=DEFAULT_MODEL_SUMMARY_FILE,
        help="Per-stratum model diagnostic CSV.",
    )
    parser.add_argument(
        "--model-arrays",
        default=DEFAULT_MODEL_ARRAYS_FILE,
        help="Compressed model-array NPZ.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    FIT_AUDIT.clear()

    # 1. Parse and prepare WDS components.
    wds_df = parse_wds_components(args.components)
    propagate_coordinates(wds_df)

    _ensure_parent(args.output)
    _ensure_parent(args.candidates)
    _ensure_parent(args.model_summary)
    _ensure_parent(args.model_arrays)

    print(f"Using Gaia query backend: {args.backend}")

    # 2. RAW MODEL sample to 15 arcsec, minimal Gaia columns.
    raw_model_results = query_gaia_selected_backend(
        wds_df,
        backend=args.backend,
        is_displaced=False,
        model_sample=True,
    )

    # 3. DISPLACED MODEL sample to 15 arcsec, minimal Gaia columns.
    disp_results = query_gaia_selected_backend(
        wds_df,
        backend=args.backend,
        is_displaced=True,
        model_sample=True,
    )

    # 4. RAW CATALOG candidates to 5 arcsec, retaining all Gaia columns.
    raw_results = query_gaia_selected_backend(
        wds_df,
        backend=args.backend,
        is_displaced=False,
        model_sample=False,
    )

    if len(raw_model_results) == 0:
        raise RuntimeError("No Gaia candidates were returned by the 15-arcsec RAW model query.")
    if len(disp_results) == 0:
        raise RuntimeError("No Gaia candidates were returned by the displaced model query.")
    if len(raw_results) == 0:
        raise RuntimeError("No Gaia candidates were returned by the 5-arcsec catalog query.")

    # 5. Verify the frozen q support against both 15-arcsec model samples.
    validate_quasicolor_support(raw_model_results, disp_results)

    # 6. Build the empirical Medan-style Bayesian distributions from 15 arcsec.
    model_bundle = build_empirical_bayesian_models(raw_model_results, disp_results)

    pd.DataFrame(FIT_AUDIT).to_csv(args.model_summary + '.fit_audit.csv', index=False)

    # 7. Save the calibrated model for reproducibility/auditing.
    save_model_diagnostics(
        model_bundle,
        model_summary_path=args.model_summary,
        model_arrays_path=args.model_arrays,
    )

    # 8. Score the full Gaia records returned by the 5-arcsec catalog query.
    scored_candidates = score_candidates(raw_results, model_bundle)

    # Duplicate candidate edges are unexpected with the fixed production query
    # path. Abort rather than silently selecting one representation of the pair.
    _check_candidate_pair_uniqueness(scored_candidates)

    # 9. Save every scored final-candidate edge before assignment.
    scored_candidates.to_csv(args.candidates, index=False)

    # 10. Global one-to-one assignment.
    final_matches = optimize_global_assignments(scored_candidates, wds_df)

    # 11. Save final output: one row per WDS component.
    final_matches.to_csv(args.output, index=False)

    n_assigned = int(
        np.sum(np.asarray(final_matches["assignment_status"]) == "assigned_global")
    )
    n_eligible_scored = int(
        np.sum(
            scored_candidates["assignment_eligible"].astype(bool)
            & np.isfinite(scored_candidates["posterior_prob"])
        )
    )

    print("\nCrossmatch complete!")
    print(f"WDS components: {len(wds_df)}")
    print(
        f"Scored candidates within {ASSIGNMENT_RADIUS_ARCSEC:.1f} arcsec: "
        f"{n_eligible_scored}"
    )
    print(f"Globally assigned components: {n_assigned}")
    print(f"Saved final catalog to: {args.output}")
    print(f"Saved scored candidates to: {args.candidates}")


if __name__ == "__main__":
    main()