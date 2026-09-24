"""
Combines WDS Summary pairs with the component-level WDS -> Gaia DR3 crossmatch.

The crossmatch is assumed to use the published schema: the globally selected
Gaia counterpart and all of its Gaia columns are stored with an ``assigned_``
prefix, while ``best_*`` columns are retained only as pairwise diagnostics.

The output retains every WDS Summary field parsed by this script and adds:
- Gaia DR3 properties for the globally assigned counterparts,
- Bayesian association and global-assignment diagnostics,
- pair-level parallax/proper-motion/radial-velocity consistency diagnostics,
- projected separation and a simple WD/MS classification,
- El-Badry et al. (2021) chance-alignment cross-reference when available, and
- per-component WDS--Gaia proper-motion differences.

Inputs:
- WDS Summary file (wdsweb_summ2.txt)
- Bayesian component-level crossmatch output (components_crossmatch_bayesian.csv)
- El-Badry et al. (2021) catalog FITS file (all_columns_catalog.fits) [optional]
"""

import math
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# Suppress openpyxl styling warnings if any
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Enable testing mode
test = False

summary_file = './data/wdsweb_summ2.txt'
crossmatch_file = './outputs/components_crossmatch_bayesian.csv'
output_file = './outputs/wds_summary_with_gaia.csv'
if test:
    crossmatch_file = './example_outputs/components_crossmatch_bayesian.csv'
    output_file = './example_outputs/wds_summary_with_gaia.csv'
# Configuration Paths
elbadry_file = './data/all_columns_catalog.fits'


def parse_summary_line(line):
    """
    Parses a line from the WDS summary file according to wdsweb_format.txt.
    Returns a dictionary of the extracted fields.
    """
    if len(line) < 50:
        return None
    
    try:
        data = {
            'wds_id': line[0:10].strip(),
            'discoverer': line[10:17].strip(),
            'components': line[17:22].strip(),
            'date_first': line[23:27].strip(),
            'date_last': line[28:32].strip(),
            'num_obs': line[33:37].strip(),
            'pa_first': line[38:41].strip(),
            'pa_last': line[42:45].strip(),
            'sep_first': line[46:51].strip(),
            'sep_last': line[52:57].strip(),
            'mag_pri': line[58:63].strip(),
            'mag_sec': line[64:69].strip(),
            'spectral_type': line[70:79].strip(),
            'pm_pri_ra': line[80:84].strip(),
            'pm_pri_dec': line[84:88].strip(),
            'pm_sec_ra': line[89:93].strip(),
            'pm_sec_dec': line[93:97].strip(),
            'dm_number': line[98:106].strip(),
            'notes': line[107:111].strip() if len(line) > 107 else '',
            'coord_precise': line[112:130].strip() if len(line) > 112 else '',
        }
        return data
    except Exception:
        return None


def extract_component_letters(components_str):
    """
    Extracts component letters from a string like 'AB', 'AC', 'Aa,Ab', 'AB-C'.
    Returns a tuple (primary, secondary).
    """
    if not components_str:
        return ('A', 'B')
    
    components_str = components_str.strip()
    
    if ',' in components_str:
        parts = components_str.split(',')
        if len(parts) >= 2:
            return (parts[0].strip(), parts[1].strip())
    
    if '-' in components_str:
        parts = components_str.split('-')
        if len(parts) >= 2:
            return (parts[0].strip(), parts[1].strip())
    
    if len(components_str) >= 2:
        return (components_str[0], components_str[1])
    
    return (components_str, 'B')


def build_crossmatch_lookup(crossmatch_df):
    """Build component lookups while preserving repeated component labels."""
    lookup = {}
    wds_ids = set()

    for row in crossmatch_df.to_dict(orient='records'):
        wds_id = str(row.get('wds_id', '')).strip()
        comp = str(row.get('component', '')).strip()
        disc = str(row.get('discoverer', '')).replace(' ', '').strip()

        wds_ids.add(wds_id)
        lookup.setdefault((wds_id, comp), []).append(row)
        if disc:
            lookup.setdefault((wds_id, disc, comp), []).append(row)

    return lookup, wds_ids


def _component_candidates(wds_id, discoverer, comp_letter, lookup):
    """Return every crossmatch row compatible with one WDS component label."""
    disc_clean = discoverer.replace(' ', '').strip()
    if disc_clean:
        rows = lookup.get((wds_id, disc_clean, comp_letter), [])
        if rows:
            return rows
    return lookup.get((wds_id, comp_letter), [])


def _row_identity(row):
    if not row:
        return None
    upload_id = _get_val(row, 'upload_id')
    return ('upload_id', upload_id) if upload_id is not None else ('object', id(row))


def find_component(wds_id, discoverer, comp_letter, lookup,
                   preferred_type=None, exclude=None):
    """Retrieve one distinct component row, optionally preferring **p or **s."""
    candidates = _component_candidates(wds_id, discoverer, comp_letter, lookup)
    exclude = set() if exclude is None else set(exclude)

    available = [r for r in candidates if _row_identity(r) not in exclude]
    if preferred_type:
        for row in available:
            if _norm_str(_get_val(row, 'component_type')) == preferred_type:
                return row

    return available[0] if available else {}


def find_pair_components(wds_id, discoverer, comp1, comp2, lookup):
    """
    Resolve the two WDS endpoints as distinct component rows.

    The primary endpoint preferentially uses a ``**p`` row and the secondary
    endpoint a ``**s`` row. This matters for rare hierarchical entries such as
    ``Bb,Bb``, where two different component rows share the same literal label.
    """
    c1 = find_component(
        wds_id, discoverer, comp1, lookup, preferred_type='**p'
    )

    excluded = {_row_identity(c1)} if c1 else set()
    c2 = find_component(
        wds_id, discoverer, comp2, lookup,
        preferred_type='**s', exclude=excluded
    )

    return c1, c2


def load_elbadry_pairs(fits_path):
    """
    Loads El-Badry 2021 catalog into an O(1) lookup dictionary of pair chances.
    """
    if not fits_path or not Path(fits_path).exists():
        return {}
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            pairs = {}
            for row in data:
                id1, id2 = int(row['source_id1']), int(row['source_id2'])
                r_chance = float(row['R_chance_align'])
                pairs[(min(id1, id2), max(id1, id2))] = r_chance
            return pairs
    except Exception as e:
        print(f"Warning: Could not load El-Badry catalog: {e}")
        return {}


def get_elbadry_info(gaia_id1, gaia_id2, elbadry_pairs):
    """ Fetch chance alignment probability if pair exists in EB21. """
    if not elbadry_pairs or not gaia_id1 or not gaia_id2:
        return False, ''
    try:
        id1, id2 = int(gaia_id1), int(gaia_id2)
        key = (min(id1, id2), max(id1, id2))
        if key in elbadry_pairs:
            return True, elbadry_pairs[key]
        return False, ''
    except (ValueError, TypeError):
        return False, ''


def _norm_str(value):
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


def _get_val(comp_data, *keys):
    """Safely retrieves the first existing non-null key from a dict."""
    if not comp_data:
        return None
    for k in keys:
        if k in comp_data and comp_data[k] is not None and not pd.isna(comp_data[k]):
            val = comp_data[k]
            if str(val).strip() != '' and str(val).strip().lower() != 'nan':
                return val
    return None


def _to_float(comp_data, *keys):
    """Extracts and casts to float across multiple candidate column names."""
    val = _get_val(comp_data, *keys)
    if val is None:
        return None
    try:
        return float(str(val).strip())
    except Exception:
        return None


def _gaia_val(comp_data, base_key):
    """
    Read a Gaia value from the globally assigned counterpart.

    Pair-level physical diagnostics must use only ``assigned_*`` values so that
    an unassigned ``best_*`` candidate cannot leak into the published summary.
    """
    if not comp_data:
        return None
    return _get_val(comp_data, f'assigned_{base_key}')


def _gaia_float(comp_data, base_key):
    val = _gaia_val(comp_data, base_key)
    if val is None:
        return None
    try:
        return float(str(val).strip())
    except Exception:
        return None


def _best_val(comp_data, base_key):
    """Read one of the pairwise ``best_*`` diagnostic fields."""
    if not comp_data:
        return None
    return _get_val(comp_data, f'best_{base_key}')


def _has_gaia_match(comp_data):
    """True only for a globally assigned Gaia counterpart."""
    if not comp_data:
        return False

    source_id = _get_val(comp_data, 'assigned_source_id')
    status = _norm_str(_get_val(comp_data, 'assignment_status'))
    return status == 'assigned_global' and source_id is not None


def _assignment_status(comp_data):
    if not comp_data:
        return ''
    return _norm_str(_get_val(comp_data, 'assignment_status'))


def _wds_pm_used(comp_data):
    """
    Reconstruct the WDS proper-motion vector used by the crossmatch.

    For secondary-component rows (component_type == '**s'), use the secondary
    WDS proper motion when either secondary PM component is non-zero; otherwise
    inherit the primary PM. For all other rows use the primary PM. WDS rows with
    a 'P' note are converted from centiarcsec/yr to mas/yr by multiplying by 10,
    matching the crossmatch. A final (0, 0) vector is treated as
    unavailable for the WDS--Gaia PM-difference diagnostic.
    """
    if not comp_data:
        return None, None

    pmra_primary = _to_float(comp_data, 'pmra_primary')
    pmdec_primary = _to_float(comp_data, 'pmdec_primary')
    pmra_secondary = _to_float(comp_data, 'pmra_secondary')
    pmdec_secondary = _to_float(comp_data, 'pmdec_secondary')

    # The crossmatch parser stores absent PM values as zero.
    pmra_primary = 0.0 if pmra_primary is None else pmra_primary
    pmdec_primary = 0.0 if pmdec_primary is None else pmdec_primary
    pmra_secondary = 0.0 if pmra_secondary is None else pmra_secondary
    pmdec_secondary = 0.0 if pmdec_secondary is None else pmdec_secondary

    component_type = _norm_str(_get_val(comp_data, 'component_type'))
    has_secondary_pm = (pmra_secondary != 0.0) or (pmdec_secondary != 0.0)
    use_secondary_pm = (component_type == '**s') and has_secondary_pm

    if use_secondary_pm:
        pmra, pmdec = pmra_secondary, pmdec_secondary
    else:
        pmra, pmdec = pmra_primary, pmdec_primary

    notes = _norm_str(_get_val(comp_data, 'notes'))
    if 'P' in notes:
        pmra *= 10.0
        pmdec *= 10.0

    if pmra == 0.0 and pmdec == 0.0:
        return None, None

    return float(pmra), float(pmdec)


def generate_summary_with_gaia(summary_path, crossmatch_path, output_path, elbadry_path=None):
    """ Core routine to merge WDS pairs with their Gaia properties. """
    print(f"Reading crossmatch from: {crossmatch_path}")
    # Keep all input columns as strings. Gaia source_id values are 64-bit integers and
    # must never pass through float64, which cannot exactly represent 19-digit IDs.
    crossmatch_df = pd.read_csv(crossmatch_path, dtype=str, keep_default_na=False)
    print(f"  - {len(crossmatch_df)} records in crossmatch")

    required_columns = {
        'wds_id', 'component', 'assignment_status', 'assigned_source_id'
    }
    missing_columns = sorted(required_columns.difference(crossmatch_df.columns))
    if missing_columns:
        raise ValueError(
            "Crossmatch file is missing required columns: "
            + ", ".join(missing_columns)
        )

    crossmatch_lookup, available_wds_ids = build_crossmatch_lookup(crossmatch_df)
    print(f"  - {len(available_wds_ids)} unique WDS systems in crossmatch")
    
    elbadry_pairs = {}
    if elbadry_path and Path(elbadry_path).exists():
        print(f"Loading El-Badry 2021 catalog from: {elbadry_path}")
        elbadry_pairs = load_elbadry_pairs(elbadry_path)
        print(f"  - {len(elbadry_pairs)} pairs loaded.")
    
    print(f"Reading summary from: {summary_path}")
    results = []
    
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    start_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('00'): 
            start_line = i
            break

    pairs_processed = 0

    print("Combining pairs and calculating metrics...")
    for line in lines[start_line:]:
        if not line.strip() or line.startswith('<') or line.startswith(' '):
            continue
            
        wds_id = line[0:10].strip()
        if wds_id not in available_wds_ids:
            continue
            
        summary_data = parse_summary_line(line)
        if not summary_data or not summary_data['wds_id']:
            continue
            
        pairs_processed += 1
        comp1, comp2 = extract_component_letters(summary_data['components'])
            
        c1, c2 = find_pair_components(
            wds_id, summary_data['discoverer'], comp1, comp2, crossmatch_lookup
        )
        
        has_match_1 = _has_gaia_match(c1)
        has_match_2 = _has_gaia_match(c2)

        gaia_id_1 = _norm_str(_gaia_val(c1, 'source_id'))
        gaia_id_2 = _norm_str(_gaia_val(c2, 'source_id'))

        if has_match_1 and has_match_2:
            completeness_flag = 'FULL'
        elif has_match_1 or has_match_2:
            completeness_flag = 'PARTIAL'
        else:
            completeness_flag = 'NONE'

        assignment_status_1 = _assignment_status(c1)
        assignment_status_2 = _assignment_status(c2)

        best_gaia_id_1 = _norm_str(_best_val(c1, 'source_id'))
        best_gaia_id_2 = _norm_str(_best_val(c2, 'source_id'))
        best_posterior_1 = _norm_str(_best_val(c1, 'posterior_prob'))
        best_posterior_2 = _norm_str(_best_val(c2, 'posterior_prob'))
        best_dist_1 = _norm_str(_best_val(c1, 'dist_arcsec'))
        best_dist_2 = _norm_str(_best_val(c2, 'dist_arcsec'))
        best_model_1 = _norm_str(_best_val(c1, 'bayes_model'))
        best_model_2 = _norm_str(_best_val(c2, 'bayes_model'))

        best_equals_assigned_1 = (
            best_gaia_id_1 == gaia_id_1 if best_gaia_id_1 and gaia_id_1 else ''
        )
        best_equals_assigned_2 = (
            best_gaia_id_2 == gaia_id_2 if best_gaia_id_2 and gaia_id_2 else ''
        )

        # Robust Gaia Coordinate Resolution (Checks Gaia coordinates first, avoids raw WDS string strings)
        ra1 = _gaia_float(c1, 'ra')
        dec1 = _gaia_float(c1, 'dec')
        ra2 = _gaia_float(c2, 'ra')
        dec2 = _gaia_float(c2, 'dec')

        gaia_sep_arcsec = ''
        gaia_pa_deg = ''
        if None not in (ra1, dec1, ra2, dec2):
            sc1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
            sc2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
            gaia_sep_arcsec = float(sc1.separation(sc2).arcsec)
            gaia_pa_deg = float(sc1.position_angle(sc2).to(u.deg).value)

        # Primary parallax
        plx1 = _gaia_float(c1, 'parallax')
        plx2 = _gaia_float(c2, 'parallax')
        eplx1 = _gaia_float(c1, 'parallax_error')
        eplx2 = _gaia_float(c2, 'parallax_error')
        ruwe1 = _gaia_float(c1, 'ruwe')
        ruwe2 = _gaia_float(c2, 'ruwe')

        # Projected separation in AU
        projected_sep_au = ''
        if gaia_sep_arcsec != '' and plx1 is not None and plx1 > 0:
            projected_sep_au = float(gaia_sep_arcsec * 1000.0 / plx1)

        # Delta Radial Velocity
        rv1 = _gaia_float(c1, 'radial_velocity')
        rv2 = _gaia_float(c2, 'radial_velocity')
        erv1 = _gaia_float(c1, 'radial_velocity_error')
        erv2 = _gaia_float(c2, 'radial_velocity_error')

        delta_rv, e_delta_rv, sigma_delta_rv = '', '', ''
        if rv1 is not None and rv2 is not None:
            delta_rv = abs(float(rv2 - rv1))
            if erv1 is not None and erv2 is not None:
                e = float((erv1 ** 2 + erv2 ** 2) ** 0.5)
                if e > 0:
                    e_delta_rv = e
                    sigma_delta_rv = float(delta_rv / e)

        # El-Badry Criterion 2: Parallax consistency with RUWE error inflation
        inflation_1, inflation_2 = 1.0, 1.0
        if plx1 is not None and plx1 > 0 and ruwe1 is not None and ruwe1 > 1.0:
            f_max_1 = 3.6921 * (plx1 / 10.0) ** 0.065
            inflation_1 = 1.0 + (f_max_1 - 1.0) * (1.0 - math.exp(-2.77 * (ruwe1 - 1.0)))

        if plx2 is not None and plx2 > 0 and ruwe2 is not None and ruwe2 > 1.0:
            f_max_2 = 3.6921 * (plx2 / 10.0) ** 0.065
            inflation_2 = 1.0 + (f_max_2 - 1.0) * (1.0 - math.exp(-2.77 * (ruwe2 - 1.0)))

        delta_parallax, e_delta_parallax, sigma_delta_parallax = '', '', ''
        if plx1 is not None and plx2 is not None:
            delta_parallax = abs(float(plx1 - plx2))
            if eplx1 is not None and eplx2 is not None:
                e_plx = float(((eplx1 * inflation_1) ** 2 + (eplx2 * inflation_2) ** 2) ** 0.5)
                if e_plx > 0:
                    e_delta_parallax = e_plx
                    sigma_delta_parallax = float(delta_parallax / e_plx)

        # El-Badry Criterion 3: Proper motion consistency with Keplerian orbit
        pmra1 = _gaia_float(c1, 'pmra')
        pmra2 = _gaia_float(c2, 'pmra')
        pmdec1 = _gaia_float(c1, 'pmdec')
        pmdec2 = _gaia_float(c2, 'pmdec')
        epmra1 = _gaia_float(c1, 'pmra_error')
        epmra2 = _gaia_float(c2, 'pmra_error')
        epmdec1 = _gaia_float(c1, 'pmdec_error')
        epmdec2 = _gaia_float(c2, 'pmdec_error')

        # Per-component WDS--Gaia proper-motion differences. These use the exact
        # WDS PM vector adopted during coordinate propagation in the crossmatch,
        # not the pairwise Gaia--Gaia proper-motion difference calculated below.
        wds_pmra1, wds_pmdec1 = _wds_pm_used(c1)
        wds_pmra2, wds_pmdec2 = _wds_pm_used(c2)

        delta_pmra_wds_gaia_1 = ''
        delta_pmdec_wds_gaia_1 = ''
        delta_pm_wds_gaia_1 = ''
        if None not in (wds_pmra1, wds_pmdec1, pmra1, pmdec1):
            delta_pmra_wds_gaia_1 = float(pmra1 - wds_pmra1)
            delta_pmdec_wds_gaia_1 = float(pmdec1 - wds_pmdec1)
            delta_pm_wds_gaia_1 = float(
                (delta_pmra_wds_gaia_1 ** 2 + delta_pmdec_wds_gaia_1 ** 2) ** 0.5
            )

        delta_pmra_wds_gaia_2 = ''
        delta_pmdec_wds_gaia_2 = ''
        delta_pm_wds_gaia_2 = ''
        if None not in (wds_pmra2, wds_pmdec2, pmra2, pmdec2):
            delta_pmra_wds_gaia_2 = float(pmra2 - wds_pmra2)
            delta_pmdec_wds_gaia_2 = float(pmdec2 - wds_pmdec2)
            delta_pm_wds_gaia_2 = float(
                (delta_pmra_wds_gaia_2 ** 2 + delta_pmdec_wds_gaia_2 ** 2) ** 0.5
            )

        delta_pm, delta_pm_orbital, e_delta_pm, delta_pm_norm = '', '', '', ''
        if None not in (pmra1, pmra2, pmdec1, pmdec2):
            dmu_ra, dmu_dec = float(pmra1 - pmra2), float(pmdec1 - pmdec2)
            delta_pm = float((dmu_ra ** 2 + dmu_dec ** 2) ** 0.5)
            
            if plx1 is not None and plx1 > 0 and gaia_sep_arcsec != '' and gaia_sep_arcsec > 0:
                delta_pm_orbital = float(0.44 * (plx1 ** 1.5) * (gaia_sep_arcsec ** -0.5))
            
            if None not in (epmra1, epmra2, epmdec1, epmdec2) and delta_pm > 0:
                var_ra = epmra1 ** 2 + epmra2 ** 2
                var_dec = epmdec1 ** 2 + epmdec2 ** 2
                e_delta_pm = float((1.0 / delta_pm) * ((var_ra * (dmu_ra ** 2) + var_dec * (dmu_dec ** 2)) ** 0.5))
            
            if delta_pm_orbital != '' and e_delta_pm != '':
                denom = delta_pm_orbital + 2.0 * e_delta_pm
                if denom > 0:
                    delta_pm_norm = float(delta_pm / denom)

        # Stellar Classification (WD vs MS)
        G1 = _gaia_float(c1, 'phot_g_mean_mag')
        G2 = _gaia_float(c2, 'phot_g_mean_mag')
        bp_rp1 = _gaia_float(c1, 'bp_rp')
        bp_rp2 = _gaia_float(c2, 'bp_rp')
        
        col1 = None not in (bp_rp1, G1, plx1) and plx1 > 0
        col2 = None not in (bp_rp2, G2, plx2) and plx2 > 0
        
        binary_type = '????' 
        if col1 or col2:
            Mg1 = G1 + 5.0 * math.log10(plx1 / 100.0) if col1 else None
            Mg2 = G2 + 5.0 * math.log10(plx2 / 100.0) if col2 else None
            
            wd1 = (Mg1 > (3.25 * bp_rp1 + 9.625)) if col1 else False
            wd2 = (Mg2 > (3.25 * bp_rp2 + 9.625)) if col2 else False
            
            if col1 and col2:
                if wd1 and wd2: binary_type = 'WDWD'
                elif wd1 != wd2: binary_type = 'WDMS'
                else: binary_type = 'MSMS'
            elif col1 and not col2:
                binary_type = 'WD??' if wd1 else 'MS??'
            elif col2 and not col1:
                binary_type = 'WD??' if wd2 else 'MS??'

        # Fetch El-Badry Cross-reference
        in_elbadry, r_chance_align = get_elbadry_info(gaia_id_1, gaia_id_2, elbadry_pairs)

        # Start from the complete parsed WDS Summary row, then append derived and
        # Gaia quantities. This guarantees that every WDS Summary-file column
        # parsed above is retained in the output catalog.
        result = dict(summary_data)
        result.update({
            'wds_id': summary_data['wds_id'],
            'discoverer': summary_data['discoverer'],
            'components': summary_data['components'],
            'date_first': summary_data['date_first'],
            'date_last': summary_data['date_last'],
            'num_obs': summary_data['num_obs'],
            'pa_first': summary_data['pa_first'],
            'pa_last': summary_data['pa_last'],
            'gaia_pa_deg': gaia_pa_deg,
            'sep_first': summary_data['sep_first'],
            'sep_last': summary_data['sep_last'],
            'gaia_sep_arcsec': gaia_sep_arcsec,
            'projected_sep_au': projected_sep_au,
            'spectral_type': summary_data['spectral_type'],
            'dm_number': summary_data['dm_number'],
            'notes': summary_data['notes'],
            'completeness_flag': completeness_flag,
            'assignment_status_1': assignment_status_1,
            'assignment_status_2': assignment_status_2,
            'best_gaia_id_1': best_gaia_id_1,
            'best_gaia_id_2': best_gaia_id_2,
            'best_posterior_prob_1': best_posterior_1,
            'best_posterior_prob_2': best_posterior_2,
            'best_dist_arcsec_1': best_dist_1,
            'best_dist_arcsec_2': best_dist_2,
            'best_bayes_model_1': best_model_1,
            'best_bayes_model_2': best_model_2,
            'best_equals_assigned_1': best_equals_assigned_1,
            'best_equals_assigned_2': best_equals_assigned_2,
            'binary_type': binary_type,
            'in_elbadry_2021': in_elbadry,
            'R_chance_align': r_chance_align,
            
            'comp1_letter': comp1,
            'comp2_letter': comp2,
            
            'gaia_id_1': gaia_id_1,
            'gaia_id_2': gaia_id_2,
            # Pairwise-association values for the globally assigned counterparts.
            'posterior_prob_1': _norm_str(_gaia_val(c1, 'posterior_prob')),
            'posterior_prob_2': _norm_str(_gaia_val(c2, 'posterior_prob')),
            'dist_arcsec_1': _norm_str(_gaia_val(c1, 'dist_arcsec')),
            'dist_arcsec_2': _norm_str(_gaia_val(c2, 'dist_arcsec')),
            'bayes_model_1': _norm_str(_gaia_val(c1, 'bayes_model')),
            'bayes_model_2': _norm_str(_gaia_val(c2, 'bayes_model')),
            
            'wds_ra_1': _norm_str(_get_val(c1, 'ra')),
            'wds_dec_1': _norm_str(_get_val(c1, 'dec')),
            'wds_ra_2': _norm_str(_get_val(c2, 'ra')),
            'wds_dec_2': _norm_str(_get_val(c2, 'dec')),
            
            'mag_wds_1': _norm_str(_get_val(c1, 'wds_mag')),
            'mag_wds_2': _norm_str(_get_val(c2, 'wds_mag')),
            'mag_gaia_1': _norm_str(_gaia_val(c1, 'phot_g_mean_mag')),
            'mag_gaia_2': _norm_str(_gaia_val(c2, 'phot_g_mean_mag')),
            'bp_rp_1': _norm_str(_gaia_val(c1, 'bp_rp')),
            'bp_rp_2': _norm_str(_gaia_val(c2, 'bp_rp')),
            
            'ra_1': ra1 if ra1 is not None else '',
            'ra_2': ra2 if ra2 is not None else '',
            'dec_1': dec1 if dec1 is not None else '',
            'dec_2': dec2 if dec2 is not None else '',
            
            'parallax_1': _norm_str(_gaia_val(c1, 'parallax')),
            'parallax_2': _norm_str(_gaia_val(c2, 'parallax')),
            'parallax_error_1': _norm_str(_gaia_val(c1, 'parallax_error')),
            'parallax_error_2': _norm_str(_gaia_val(c2, 'parallax_error')),
            'delta_parallax': delta_parallax,
            'e_delta_parallax': e_delta_parallax,
            'sigma_delta_parallax': sigma_delta_parallax,
            
            'pmra_1': _norm_str(_gaia_val(c1, 'pmra')),
            'pmra_2': _norm_str(_gaia_val(c2, 'pmra')),
            'pmra_error_1': _norm_str(_gaia_val(c1, 'pmra_error')),
            'pmra_error_2': _norm_str(_gaia_val(c2, 'pmra_error')),
            'pmdec_1': _norm_str(_gaia_val(c1, 'pmdec')),
            'pmdec_2': _norm_str(_gaia_val(c2, 'pmdec')),
            'pmdec_error_1': _norm_str(_gaia_val(c1, 'pmdec_error')),
            'pmdec_error_2': _norm_str(_gaia_val(c2, 'pmdec_error')),

            # WDS PM values actually used for the two component rows and their
            # vector differences from the globally assigned Gaia DR3 PMs.
            'wds_pmra_used_1': wds_pmra1 if wds_pmra1 is not None else '',
            'wds_pmdec_used_1': wds_pmdec1 if wds_pmdec1 is not None else '',
            'wds_pmra_used_2': wds_pmra2 if wds_pmra2 is not None else '',
            'wds_pmdec_used_2': wds_pmdec2 if wds_pmdec2 is not None else '',
            'delta_pmra_wds_gaia_1': delta_pmra_wds_gaia_1,
            'delta_pmdec_wds_gaia_1': delta_pmdec_wds_gaia_1,
            'delta_pm_wds_gaia_1': delta_pm_wds_gaia_1,
            'delta_pmra_wds_gaia_2': delta_pmra_wds_gaia_2,
            'delta_pmdec_wds_gaia_2': delta_pmdec_wds_gaia_2,
            'delta_pm_wds_gaia_2': delta_pm_wds_gaia_2,
            
            # Existing Gaia--Gaia component-to-component PM diagnostics.
            'delta_pm': delta_pm,
            'delta_pm_orbital': delta_pm_orbital,
            'e_delta_pm': e_delta_pm,
            'delta_pm_norm': delta_pm_norm,
            
            'ruwe_1': _norm_str(_gaia_val(c1, 'ruwe')),
            'ruwe_2': _norm_str(_gaia_val(c2, 'ruwe')),
            'non_single_star_1': _norm_str(_gaia_val(c1, 'non_single_star')),
            'non_single_star_2': _norm_str(_gaia_val(c2, 'non_single_star')),
            
            'radial_velocity_1': _norm_str(_gaia_val(c1, 'radial_velocity')),
            'radial_velocity_2': _norm_str(_gaia_val(c2, 'radial_velocity')),
            'radial_velocity_error_1': _norm_str(_gaia_val(c1, 'radial_velocity_error')),
            'radial_velocity_error_2': _norm_str(_gaia_val(c2, 'radial_velocity_error')),
            'delta_rv': delta_rv,
            'e_delta_rv': e_delta_rv,
            'sigma_delta_rv': sigma_delta_rv,
            
            'visibility_periods_1': _norm_str(_gaia_val(c1, 'visibility_periods_used')),
            'visibility_periods_2': _norm_str(_gaia_val(c2, 'visibility_periods_used'))
        })
        results.append(result)
    
    print(f"\nSummary:")
    print(f"  - Pairs processed from summary: {pairs_processed}")
    print(f"  - FULL (both components matched): {sum(1 for r in results if r['completeness_flag'] == 'FULL')}")
    print(f"  - PARTIAL (one component matched): {sum(1 for r in results if r['completeness_flag'] == 'PARTIAL')}")
    print(f"  - NONE: {sum(1 for r in results if r['completeness_flag'] == 'NONE')}")
    
    result_df = pd.DataFrame(results)
    
    # Enforce correct types
    text_cols = [
        'wds_id', 'discoverer', 'components', 'spectral_type', 'dm_number', 
        'notes', 'coord_precise', 'completeness_flag', 'binary_type', 'comp1_letter', 'comp2_letter',
        'assignment_status_1', 'assignment_status_2',
        'gaia_id_1', 'gaia_id_2', 'best_gaia_id_1', 'best_gaia_id_2',
        'best_bayes_model_1', 'best_bayes_model_2', 'bayes_model_1', 'bayes_model_2',
        'wds_ra_1', 'wds_dec_1', 'wds_ra_2', 'wds_dec_2', 'in_elbadry_2021'
    ]
    
    for col in result_df.columns:
        if col not in text_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
    # Save CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, float_format='%.15g')
    print(f"\nSaved CSV: {output_path}")

    # Save Excel
    excel_path = str(output_path).replace('.csv', '.xlsx')
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='WDS_Gaia')
            worksheet = writer.sheets['WDS_Gaia']
            
            for col_idx, col_name in enumerate(result_df.columns, start=1):
                col_letter = worksheet.cell(row=1, column=col_idx).column_letter
                
                if col_name in text_cols:
                    for cell in worksheet[col_letter]:
                        cell.number_format = '@'
                else:
                    for idx, cell in enumerate(worksheet[col_letter]):
                        if idx > 0 and cell.value not in (None, ''):
                            cell.number_format = 'General'
                            
                max_length = max((len(str(cell.value)) for cell in worksheet[col_letter] if cell.value is not None), default=0)
                worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)
                
        print(f"Saved Excel: {excel_path}")
    except Exception as e:
        print(f"Excel export failed (CSV is safe). Error: {e}")
        
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combines WDS pairs with globally assigned Gaia DR3 counterparts, retains all parsed WDS Summary columns, and adds pair/component diagnostics.')
    parser.add_argument('--summary', default=summary_file, help='Path to WDS summary file.')
    parser.add_argument('--crossmatch', default=crossmatch_file, help='Path to Bayesian crossmatch CSV.')
    parser.add_argument('--output', default=output_file, help='Path to output CSV.')
    parser.add_argument('--elbadry', default=elbadry_file, help='Path to El-Badry FITS file.')
    parser.add_argument('--skip-elbadry', action='store_true', help='Skip El-Badry chance alignment catalog.')
    args = parser.parse_args()

    elbadry_path = None if args.skip_elbadry else args.elbadry
    generate_summary_with_gaia(args.summary, args.crossmatch, args.output, elbadry_path)