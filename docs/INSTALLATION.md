# Installation

This guide covers installing and verifying `wds-crossmatch`. The project is
used directly from the GitHub repository.

## 1. Requirements

- Python 3.13 (the version used during development and testing)
- `git`
- An internet connection (the pipeline queries the Gaia DR3 TAP service)

Required Python libraries:

| Library | Used for |
| :--- | :--- |
| `numpy`, `pandas`, `scipy` | Numerical work, tables, model fitting and assignment |
| `astropy`, `astroquery` | Coordinates and Gaia TAP queries |
| `openpyxl` | Excel export in `summary.py` |

Tested versions: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, astropy 8.0.1,
astroquery 0.4.11.

## 2. Install

```bash
git clone https://github.com/TehMartinXz/wds-crossmatch.git
cd wds-crossmatch
python3 -m pip install numpy pandas scipy astropy astroquery openpyxl
```

Using `python3 -m pip` guarantees the libraries are installed for the same
interpreter that will run the scripts.

## 3. Verify the installation

Run all commands from the repository root (paths are relative).

Quick check with the small sample (64 components):

```bash
python3 crossmatch.py --components example_data/wds_components.summ.txt
```

This sample is too small to calibrate the statistical model, so it assigns no
components. It only confirms that the installation works.

Full check with the extended sample (3,409 components):

```bash
python3 crossmatch.py --components example_data/wds_components.summ.extended.txt
```

The run takes a few minutes, because it queries Gaia DR3. It should end with:

```text
Crossmatch complete!
WDS components: 3409
Scored candidates within 5.0 arcsec: 2865
Globally assigned components: 1350
```

Results are written to `outputs/`.

## 4. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `CERTIFICATE_VERIFY_FAILED` when querying Gaia | Python from python.org on macOS ships without SSL certificates configured | Run `Install Certificates.command` from `/Applications/Python 3.13/` |
| `FileNotFoundError` for the extended sample | Misspelled file name | Use the exact name `wds_components.summ.extended.txt` (dots, not underscores, around `summ`) |
| `ModuleNotFoundError: No module named 'astroquery'` | `python` points to a different environment (e.g. conda `base`) than the one where libraries were installed | Use `python3` explicitly and install with `python3 -m pip` |
| `WARNING: W50: Invalid unit string 'log(cm.s**-2)'` | astropy does not recognise a unit label used by Gaia | Harmless: it affects only a unit label, not the values |
| `fatal: pathspec ... did not match any files` in git | The terminal is not in the repository root | `cd` to the repository root first |