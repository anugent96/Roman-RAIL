#!/usr/bin/env python3
"""
Cross-match the RAPID/HLTDS SExtractor catalogs and run the RAIL PZFlow
photo-z estimator on the matched field (Roman bands only).

Stage 1 -- catalog matching:
    Each filter directory (F184/, H158/, ...) contains a SExtractor catalog
    named 'awaicgen_output_mosaic_refimsexcat.txt'. Sources are matched on
    sky position (ALPHAWIN_J2000, DELTAWIN_J2000) with a one-to-one
    nearest-neighbor match within --radius. Instrumental MAG_BEST /
    MAGERR_BEST are calibrated to AB with:

        mag_AB = MAG - 2.5*log10(EXPTIME) + 17 + BANDZPT

    The matched catalog is written to matched_photometry.hdf5 with flat
    root-level datasets: objid, RA, Dec (ALPHAWIN/DELTAWIN, deg), and
    roman_<band> / roman_<band>_err per filter (99 where undetected).
    No CSV is written.

Stage 2 -- photo-z estimation:
    The matched HDF5 is run through the RAIL PZFlow estimator using the
    Roman-only trained model (trained_models/pzflow_roman_model.pkl; the
    inform step runs first if the model is missing). Note the PZFlow stage
    has no ref_band config option, so no reference band is passed (the
    companion LePhare script uses roman_Z087).

    Results are saved to pzflow_<YYYYMMDD_HHMMSS>UT.npz (current UT time)
    containing z_grid, pdfs, z_median, z_mode, objid, RA, Dec.

Usage:
    python run_RAPID_pzflow.py [--radius ARCSEC]
"""

import os

# Cap JAX memory to avoid OOM on large datasets
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import argparse
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tables_io
from astropy.coordinates import SkyCoord
from astropy.io import ascii
import astropy.units as u
from rail.core.data import ModelHandle

BASE_DIR = Path(__file__).resolve().parent          # HLTDS_field/
ROOT_DIR = BASE_DIR.parent                          # RAIL Test/
CATALOG_NAME = "awaicgen_output_mosaic_refimsexcat.txt"
DEFAULT_RADIUS = 0.2  # arcsec
MISSING_VALUE = 99.0
MATCHED_H5 = BASE_DIR / "matched_photometry.hdf5"

TRAIN_FILE = ROOT_DIR / "roman_train_simulated_phot_zred.hdf5"
MODEL_FILE = ROOT_DIR / "trained_models" / "pzflow_roman_model.pkl"

BAND_ORDER = ["R062", "Z087", "Y106", "J129", "H158", "F184", "K213"]
BANDS = [f"roman_{b}" for b in BAND_ORDER]
LIMVALS = [27.97, 27.63, 27.60, 27.60, 27.52, 26.95, 25.64]

filter_data = {
    "H158": {"BANDZPT": 15.074, "ZPTMAG": 17.638, "EXPTIME": 302.275},
    "J129": {"BANDZPT": 15.040, "ZPTMAG": 17.638, "EXPTIME": 302.275},
    "F184": {"BANDZPT": 14.622, "ZPTMAG": 18.824, "EXPTIME": 901.175},
    "K213": {"BANDZPT": 14.579, "ZPTMAG": 18.824, "EXPTIME": 901.175},
    "Y106": {"BANDZPT": 15.024, "ZPTMAG": 17.638, "EXPTIME": 302.275},
    "Z087": {"BANDZPT": 14.964, "ZPTMAG": 16.455, "EXPTIME": 101.7},
    "R062": {"BANDZPT": 15.297, "ZPTMAG": 16.954, "EXPTIME": 161.0},
}


# ── Stage 1: catalog matching ─────────────────────────────────────────────────

def find_catalogs(base_dir):
    """Return {filter_name: catalog_path} for every filter directory."""
    catalogs = {}
    for sub in sorted(base_dir.iterdir()):
        cat = sub / CATALOG_NAME
        if sub.is_dir() and cat.is_file():
            catalogs[sub.name] = cat
    return catalogs


def read_catalog(path, filt):
    """Read a SExtractor catalog and return calibrated AB photometry."""
    table = ascii.read(path, format="sextractor")
    df = table["ALPHAWIN_J2000", "DELTAWIN_J2000",
               "MAG_BEST", "MAGERR_BEST"].to_pandas()
    df.columns = ["ra", "dec", "mag_inst", "magerr_inst"]
    # A magnitude of 99 is SExtractor's sentinel for a failed measurement.
    df.loc[df["mag_inst"] >= 99, ["mag_inst", "magerr_inst"]] = np.nan

    zpt = filter_data[filt]
    df["mag"] = (df["mag_inst"] - 2.5 * np.log10(zpt["EXPTIME"])
                 + 17 + zpt["BANDZPT"])
    df["magerr"] = df["mag"] - ((df["mag_inst"] - df["magerr_inst"])
                                - 2.5 * np.log10(zpt["EXPTIME"])
                                + 17 + zpt["BANDZPT"])
    return df[["ra", "dec", "mag", "magerr"]]


def match_one_to_one(master_coords, cat_coords, radius):
    """
    Nearest-neighbor match of catalog sources to master sources, enforced
    one-to-one: if several catalog sources land on the same master source,
    only the closest pair is kept.

    Returns (master_idx, cat_idx) arrays of matched pairs.
    """
    idx, sep, _ = cat_coords.match_to_catalog_sky(master_coords)
    good = sep < radius

    pairs = pd.DataFrame({
        "master": idx[good],
        "cat": np.flatnonzero(good),
        "sep": sep[good].arcsec,
    })
    pairs = pairs.sort_values("sep").drop_duplicates("master", keep="first")
    return pairs["master"].to_numpy(), pairs["cat"].to_numpy()


def build_matched_catalog(radius, h5_path=MATCHED_H5):
    """
    Cross-match all filter catalogs and write the matched photometry to
    an HDF5 file with objid, RA, Dec and roman_<band>(_err) datasets.

    # NOTE: You can change the output to parquet if you need!!
    """
    catalogs = find_catalogs(BASE_DIR)
    if not catalogs:
        raise SystemExit(f"No '{CATALOG_NAME}' files found under {BASE_DIR}")
    unknown = set(catalogs) - set(filter_data)
    if unknown:
        raise SystemExit(f"No zeropoint data for filter(s): {', '.join(unknown)}")

    print(f"Found {len(catalogs)} filter catalogs: {', '.join(catalogs)}")
    data = {}
    for filt, path in catalogs.items():
        data[filt] = read_catalog(path, filt)
        print(f"  {filt}: {len(data[filt])} sources")

    # Seed the master list with the largest catalog so most sources
    # are matched (rather than appended) on later passes.
    filters = sorted(data, key=lambda f: len(data[f]), reverse=True)
    seed = filters[0]
    master = data[seed][["ra", "dec"]].copy().reset_index(drop=True)
    phot = {f: pd.DataFrame(np.nan, index=master.index, columns=["mag", "magerr"])
            for f in filters}
    phot[seed][:] = data[seed][["mag", "magerr"]].to_numpy()
    print(f"  seeding master list with {seed}")

    for filt in filters[1:]:
        cat = data[filt]
        master_coords = SkyCoord(master["ra"].to_numpy() * u.deg,
                                 master["dec"].to_numpy() * u.deg)
        cat_coords = SkyCoord(cat["ra"].to_numpy() * u.deg,
                              cat["dec"].to_numpy() * u.deg)

        m_idx, c_idx = match_one_to_one(master_coords, cat_coords, radius)
        phot[filt].loc[m_idx, ["mag", "magerr"]] = \
            cat.loc[c_idx, ["mag", "magerr"]].to_numpy()

        # Append catalog sources with no master counterpart as new sources.
        unmatched = np.setdiff1d(np.arange(len(cat)), c_idx)
        if len(unmatched) > 0:
            new_index = pd.RangeIndex(len(master), len(master) + len(unmatched))
            master = pd.concat(
                [master, cat.loc[unmatched, ["ra", "dec"]].set_index(new_index)])
            for f in filters:
                pad = pd.DataFrame(np.nan, index=new_index, columns=["mag", "magerr"])
                phot[f] = pd.concat([phot[f], pad])
            phot[filt].loc[new_index, ["mag", "magerr"]] = \
                cat.loc[unmatched, ["mag", "magerr"]].to_numpy()

        print(f"  {filt}: matched {len(m_idx)}, added {len(unmatched)} new "
              f"(master now {len(master)})")

    n_filters = pd.concat([phot[f]["mag"] for f in filters], axis=1) \
                  .notna().sum(axis=1)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("objid", data=np.arange(len(master), dtype=np.int64))
        h5.create_dataset("RA", data=master["ra"].to_numpy(dtype=np.float64))
        h5.create_dataset("Dec", data=master["dec"].to_numpy(dtype=np.float64))
        for band in BAND_ORDER:
            if band not in phot:
                continue
            mag = phot[band]["mag"].fillna(MISSING_VALUE)
            err = phot[band]["magerr"].fillna(MISSING_VALUE)
            h5.create_dataset(f"roman_{band}",
                              data=mag.to_numpy(dtype=np.float64))
            h5.create_dataset(f"roman_{band}_err",
                              data=err.to_numpy(dtype=np.float64))

    in_all = int((n_filters == len(filters)).sum())
    print(f"\nWrote {len(master)} sources to {h5_path}")
    print(f"  detected in all {len(filters)} filters: {in_all}")
    print(f"  match radius: {radius.to_value(u.arcsec)} arcsec")
    return h5_path


# ── Stage 2: PZFlow photo-z estimation ────────────────────────────────────────

def run_pzflow(test_file, outfile):
    from rail.estimation.algos.pzflow_nf import PZFlowInformer, PZFlowEstimator

    training_data = tables_io.read(str(TRAIN_FILE))
    test_data = tables_io.read(str(test_file))
    print(f"Training set : {TRAIN_FILE}")
    print(f"Test set     : {test_file}")

    maglims = dict(zip(BANDS, LIMVALS))
    has_true_z = "redshift" in test_data

    # The PZFlow estimator selects the redshift column from the test table
    # even though posterior() only uses the z grid, never the column values.
    # Add a dummy so test files without redshift (e.g. real fields) work.
    if not has_true_z:
        test_data["redshift"] = np.zeros(len(test_data[BANDS[0]]))

    # NOTE: PZFlow can use errors on photometry, however this uses an ENORMOUS
    # amount of memory with results that are not much better, so we disable this
    cfg = dict(
        zmin=0.0, zmax=6.0, nzbins=601,
        column_names=BANDS,
        mag_limits=maglims,
        include_mag_errors=False,
        num_training_epochs=100,
        chunk_size=20000,
    )

    start_time = time.perf_counter()

    if MODEL_FILE.exists():
        print(f"Loading pzflow model from {MODEL_FILE}...")
        with open(MODEL_FILE, "rb") as f:
            flow = pickle.load(f)
        model_handle = ModelHandle("model", path=str(MODEL_FILE))
        model_handle.set_data(flow)
        print("Model loaded.")
    else:
        print("Informing...")
        informer = PZFlowInformer.make_stage(
            name="inform_PZ", model=str(MODEL_FILE), hdf5_groupname="", **cfg)
        informer.inform(training_data)
        model_handle = informer.get_handle("model")
        print("Informed.")

    print("Now estimating...")
    estimator = PZFlowEstimator.make_stage(
        name="PZFlow", model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_pzflow"}, **cfg)

    results = estimator.estimate(test_data)

    print("Saving results...")
    ensemble = results.data
    z_grid = np.linspace(0.0, 6.0, 601)

    pdfs = ensemble.pdf(z_grid)
    z_median = np.ndarray.flatten(ensemble.ppf(0.5))
    z_mode = np.ndarray.flatten(ensemble.mode(grid=z_grid))

    save_arrays = dict(z_grid=z_grid, pdfs=pdfs,
                       z_median=z_median, z_mode=z_mode)
    if has_true_z:
        save_arrays["z_true"] = test_data["redshift"]
    for key in ("objid", "RA", "Dec"):
        if key in test_data:
            save_arrays[key] = test_data[key]

    np.savez_compressed(outfile, **save_arrays)
    print(f"Saved to {outfile}  ({len(z_median)} galaxies, "
          f"{time.perf_counter() - start_time:.1f}s total)")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-match RAPID SExtractor catalogs and run PZFlow.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help=f"match radius in arcsec (default {DEFAULT_RADIUS})")
    args = parser.parse_args()

    matched = build_matched_catalog(args.radius * u.arcsec)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outfile = BASE_DIR / f"pzflow_{timestamp}UT.npz"
    run_pzflow(matched, outfile)


if __name__ == "__main__":
    main()
