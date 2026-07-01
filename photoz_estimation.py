"""
Unified photo-z estimation script.

Usage:
    python photoz_estimation.py --estimator lephare --data RomanRubin --refband roman_J129 --outfile lephare_results.npz

Supported estimators: flexzboost, pzflow, gpz, bpz, lephare, cmnn, knn
"""

import argparse
import os
import time
import numpy as np
import tables_io
from rail.core.data import ModelHandle
import pickle

# Cap JAX memory to avoid OOM on large datasets
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

parser = argparse.ArgumentParser(description='Run a RAIL photo-z estimator and save results.')
parser.add_argument("--estimator", required=True,
                    choices=['flexzboost', 'pzflow', 'gpz', 'bpz', 'lephare', 'cmnn', 'knn'],
                    help="Which estimator to run")
parser.add_argument("--refband", required=True, help="Reference band, e.g. roman_J129")
parser.add_argument("--outfile", required=True, help="Output .npz file path")
parser.add_argument("--data", required=True, choices=['Roman', 'RomanRubin'],
                    help="Roman = Roman bands only; RomanRubin = Roman + Rubin/LSST")

args = parser.parse_args()

# ── Data loading ──────────────────────────────────────────────────────────────

if args.data == 'RomanRubin':
    trainFile = "roman_rubin_train_simulated_phot_zred.hdf5"
    testFile  = "roman_rubin_open_universe.hdf5"
    bands   = ['lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'lsst_z', 'lsst_y',
               'roman_R062', 'roman_Z087', 'roman_Y106', 'roman_J129', 'roman_H158', 'roman_F184', 'roman_K213']
    limvals = [24.44, 25.60, 25.59, 25.17, 24.47, 23.56,
               27.97, 27.63, 27.60, 27.60, 27.52, 26.95, 25.64]
    lp_filters  = ['lsst/total_u.pb', 'lsst/total_g.pb', 'lsst/total_r.pb',
                   'lsst/total_i.pb', 'lsst/total_z.pb', 'lsst/total_y.pb',
                   'roman/roman_R062.pb', 'roman/roman_Z087.pb', 'roman/roman_Y106.pb',
                   'roman/roman_J129.pb', 'roman/roman_H158.pb', 'roman/roman_F184.pb',
                   'roman/roman_K213.pb']
    bpz_filters = ['DC2LSST_u', 'DC2LSST_g', 'DC2LSST_r', 'DC2LSST_i', 'DC2LSST_z', 'DC2LSST_y',
                   'roman_R062', 'roman_Z087', 'roman_Y106', 'roman_J129', 'roman_H158', 'roman_F184', 'roman_K213']

    flexzboost_model = 'flexzboost_model.pkl'
    pzflow_model = 'pzflow_model.pkl'
    lephare_model = 'lephare_model.pkl'
    gpz_model = 'gpz_model.pkl'
    bpz_model = 'bpz_model.pkl'
    cmnn_model = 'cmnn_model.pkl'
    knn_model = 'knn_model.pkl'

elif args.data == 'Roman':
    trainFile = "smaller_roman_train_simulated_phot_zred.hdf5"
    testFile  = "roman_highz_simulated_test.hdf5"
    bands   = ['roman_R062', 'roman_Z087', 'roman_Y106', 'roman_J129', 'roman_H158', 'roman_F184', 'roman_K213']
    limvals = [27.97, 27.63, 27.60, 27.60, 27.52, 26.95, 25.64]
    lp_filters  = ['roman/roman_R062.pb', 'roman/roman_Z087.pb', 'roman/roman_Y106.pb',
                   'roman/roman_J129.pb', 'roman/roman_H158.pb', 'roman/roman_F184.pb',
                   'roman/roman_K213.pb']
    bpz_filters = ['roman_R062', 'roman_Z087', 'roman_Y106', 'roman_J129', 'roman_H158', 'roman_F184', 'roman_K213']

    flexzboost_model = 'flexzboost_roman_model.pkl'
    pzflow_model = 'pzflow_roman_model.pkl'
    lephare_model = 'lephare_roman_model.pkl'
    gpz_model = 'gpz_roman_model.pkl'
    bpz_model = 'bpz_roman_model.pkl'
    cmnn_model = 'cmnn_roman_model.pkl'
    knn_model = 'knn_roman_model.pkl'

training_data = tables_io.read(trainFile)
test_data     = tables_io.read(testFile)
print(f'Training set : {trainFile}')
print(f'Test set     : {testFile}')

import pandas as pd
err_cols = [c for c in pd.DataFrame(test_data).columns if 'err' in c.lower()]
print(f'Error columns in test data: {err_cols}')

errbands = [f"{band}_err" for band in bands]
maglims  = dict(zip(bands, limvals))

print(f'Estimator    : {args.estimator}')
print(f'Bands        : {bands}')

# ── Estimator-specific setup ──────────────────────────────────────────────────

start_time = time.perf_counter()

if args.estimator == 'flexzboost':
    from rail.estimation.algos.flexzboost import FlexZBoostInformer, FlexZBoostEstimator

    cfg = dict(
        zmin=0.0, zmax=6.0, nzbins=601,
        trainfrac=0.75,
        bumpmin=0.02, bumpmax=0.35, nbump=20,
        sharpmin=0.7, sharpmax=2.1, nsharp=15,
        max_basis=35, basis_system="cosine",
        regression_params={"max_depth": 8, "objective": "reg:squarederror"},
        bands=bands, ref_band=args.refband, err_bands=errbands,
        mag_limits=maglims, nondetect_val=99, include_mag_errors=True,
    )
    if os.path.exists(flexzboost_model):
        print('Found existing flexzboost_model.pkl, skipping inform step.')
        model_handle = ModelHandle('model', path=flexzboost_model)
    else:
        print('Informing...')
        informer  = FlexZBoostInformer.make_stage(
            name='inform_FlexZBoost', model=flexzboost_model, hdf5_groupname="", **cfg)
        informer.inform(training_data)
        print('Informed.')
        model_handle = informer.get_handle('model')
    print('Now estimating...')
    estimator = FlexZBoostEstimator.make_stage(
        name='FlexZBoost', model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_flexzboost"}, **cfg)

elif args.estimator == 'pzflow':
    from rail.estimation.algos.pzflow_nf import PZFlowInformer, PZFlowEstimator

    # NOTE: PZFlow can use errors on photometry, however this uses an ENORMOUS
    # amount of memory with results that are not much better, so we disable this

    cfg = dict(
        zmin=0.0, zmax=6.0, nzbins=601,
        column_names=bands,
        mag_limits=maglims,
        include_mag_errors=False,
        num_training_epochs=100,
        chunk_size=20000,
    )
    model_path = pzflow_model
    if os.path.exists(model_path):
        print(f'Loading pzflow model from {model_path}...')
        with open(model_path, 'rb') as f:
            flow = pickle.load(f)

        model_handle = ModelHandle('model', path=model_path)
        model_handle.set_data(flow)
        print('Model loaded.')
        print('Now estimating...')

    else:
        print('Informing...')
        informer  = PZFlowInformer.make_stage(
            name='inform_PZ', model=pzflow_model, hdf5_groupname="", **cfg)
        informer.inform(training_data)
        model_handle = informer.get_handle('model')
        print('Informed. Now estimating...')

    estimator = PZFlowEstimator.make_stage(
        name='PZFlow', model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_pzflow"}, **cfg)

elif args.estimator == 'gpz':
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

    replace_error_vals = [lim * 0.1 for lim in limvals]
    cfg = dict(
        zmin=0.0, zmax=6.0, nzbins=601,
        bands=bands, err_bands=errbands, ref_band=args.refband,
        mag_limits=maglims, nondetect_val=99.0,
        replace_error_vals=replace_error_vals,
        trainfrac=0.75, n_basis=50, max_iter=200,
    )
    if os.path.exists(gpz_model):
        print('Found existing gpz_model.pkl, skipping inform step.')
        model_handle = ModelHandle('model', path=gpz_model)
    else:
        print('Informing...')
        informer  = GPzInformer.make_stage(
            name='inform_GPz', model=gpz_model, hdf5_groupname="", **cfg)
        informer.inform(training_data)
        print('Informed.')
        model_handle = informer.get_handle('model')
    print('Now estimating...')
    estimator = GPzEstimator.make_stage(
        name='GPz', model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_gpz"}, **cfg)

elif args.estimator == 'bpz':
    import importlib.util
    from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator

    bpz_spec      = importlib.util.find_spec("desc_bpz")
    bpz_data_path = os.path.join(os.path.dirname(bpz_spec.origin), "data_files")
    zp_errors     = [0.01] * len(bands)

    inform_cfg = dict(
        bands=bands, err_bands=errbands, ref_band=args.refband,
        mag_limits=maglims, nondetect_val=99.0,
        data_path=bpz_data_path,
        output_hdfn=True,
    )
    estimate_cfg = dict(
        **inform_cfg,
        zmin=0.0, zmax=6.0, dz=0.01,
        filter_list=bpz_filters,
        zp_errors=zp_errors,
    )
    if os.path.exists(bpz_model):
        print('Found existing bpz_model.pkl, skipping inform step.')
        model_handle = ModelHandle('model', path=bpz_model)
    else:
        print('Informing...')
        informer  = BPZliteInformer.make_stage(
            name='inform_BPZ', model=bpz_model, hdf5_groupname="", **inform_cfg)
        informer.inform(training_data)
        print('Informed.')
        model_handle = informer.get_handle('model')
    print('Now estimating...')
    estimator = BPZliteEstimator.make_stage(
        name='BPZ', model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_bpz"}, **estimate_cfg)

elif args.estimator == 'lephare':
    import sys, os
    import lephare as lp
    from rail.estimation.algos.lephare import LephareInformer, LephareEstimator

    lephare_config = lp.default_cosmos_config.copy()

    if args.data == 'RomanRubin':
        lephare_config.update({
            "FILTER_LIST":  ",".join(lp_filters),
            "FILTER_FILE":  "roman_rubin_filters",
            "FILTER_CALIB": ",".join(["0"] * len(lp_filters)),
            "STAR_LIB":     "LIB_STAR",  "STAR_LIB_IN": "LIB_STAR",  "STAR_LIB_OUT": "ROMAN_RUBIN_STAR_MAG",
            "GAL_LIB":      "LIB_CE",    "GAL_LIB_IN":  "LIB_CE",    "GAL_LIB_OUT":  "ROMAN_RUBIN_GAL_MAG",
            "QSO_LIB":      "LIB_QSO",   "QSO_LIB_IN":  "LIB_QSO",   "QSO_LIB_OUT":  "ROMAN_RUBIN_QSO_MAG",
            "ZPHOTLIB":     "ROMAN_RUBIN_GAL_MAG,ROMAN_RUBIN_STAR_MAG,ROMAN_RUBIN_QSO_MAG",
            "INP_TYPE":     "M",
            "GLB_CONTEXT":  "0",
            "ERR_SCALE":    "0.02",
            "AUTO_ADAPT":   "YES",
        })

    elif args.data == 'Roman':
        lephare_config.update({
            "FILTER_LIST":  ",".join(lp_filters),
            "FILTER_FILE":  "roman_filters",
            "FILTER_CALIB": ",".join(["0"] * len(lp_filters)),
            "STAR_LIB":     "LIB_STAR",  "STAR_LIB_IN": "LIB_STAR",  "STAR_LIB_OUT": "ROMAN_STAR_MAG",
            "GAL_LIB":      "LIB_CE",    "GAL_LIB_IN":  "LIB_CE",    "GAL_LIB_OUT":  "ROMAN_GAL_MAG",
            "QSO_LIB":      "LIB_QSO",   "QSO_LIB_IN":  "LIB_QSO",   "QSO_LIB_OUT":  "ROMAN_QSO_MAG",
            "ZPHOTLIB":     "ROMAN_GAL_MAG,ROMAN_STAR_MAG,ROMAN_QSO_MAG",
            "INP_TYPE":     "M",
            "GLB_CONTEXT":  "0",
            "ERR_SCALE":    "0.02",
            "AUTO_ADAPT":   "YES",
        })
    
    inform_cfg = dict(
        zmin=0.0, zmax=6.0, nzbins=601,
        bands=bands, err_bands=errbands, ref_band=args.refband,
        mag_limits=maglims, nondetect_val=99.0, redshift_col='redshift',
    )
    for key, val in lephare_config.items():
        inform_cfg[f"lephare.{key}"] = val

    estimate_cfg = dict(
        bands=bands, err_bands=errbands, ref_band=args.refband,
        mag_limits=maglims, nondetect_val=99.0,
        posterior_output=11,  # BAY_ZG: Bayesian galaxy redshift posterior
    )
    if os.path.exists(lephare_model):
        print('Found existing lephare_model.pkl, skipping inform step.')
        from rail.core.data import ModelHandle
        model_handle = ModelHandle('model', path=lephare_model)
    else:
        print('Informing...')
        informer = LephareInformer.make_stage(
            name='inform_LePhare', model=lephare_model, hdf5_groupname="", **inform_cfg)
        informer.inform(training_data)
        print('Informed.')
        model_handle = informer.get_handle('model')

    estimator = LephareEstimator.make_stage(
        name='LePhare', model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_lephare"}, **estimate_cfg)

elif args.estimator == 'cmnn':
    from rail.estimation.algos.cmnn import CMNNInformer, CMNNEstimator

    # This is a toy model. It is not recommended that we use this

    cfg = dict(
        zmin=0.0, zmax=6.0, nzbins=601,
        bands=bands, err_bands=errbands, ref_band=args.refband,
        mag_limits=maglims, nondetect_val=99.0,
        nondetect_replace=False,
        redshift_col='redshift',
    )
    if os.path.exists(cmnn_model):
        print('Found existing cmnn_model.pkl, skipping inform step.')
        model_handle = ModelHandle('model', path=cmnn_model)
    else:
        print('Informing...')
        informer  = CMNNInformer.make_stage(
            name='inform_CMNN', model=cmnn_model, hdf5_groupname="", **cfg)
        informer.inform(training_data)
        print('Informed.')
        model_handle = informer.get_handle('model')
    print('Now estimating...')
    estimator = CMNNEstimator.make_stage(
        name='CMNN', model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_cmnn"}, **cfg)

elif args.estimator == 'knn':
    from rail.estimation.algos.k_nearneigh import KNearNeighInformer, KNearNeighEstimator

    # This is a toy model. It is not recommended that we use this

    cfg = dict(
        zmin=0.0, zmax=6.0, nzbins=601,
        bands=bands, err_bands=errbands, ref_band=args.refband,
        mag_limits=maglims, nondetect_val=99.0,
        redshift_col='redshift',
        trainfrac=0.75, seed=0,
        leaf_size=15, nneigh_min=3, nneigh_max=7,
        only_colors=False,
    )
    if os.path.exists(knn_model):
        print('Found existing knn_model.pkl, skipping inform step.')
        model_handle = ModelHandle('model', path=knn_model)
    else:
        print('Informing...')
        informer  = KNearNeighInformer.make_stage(
            name='inform_KNN', model=knn_model, hdf5_groupname="", **cfg)
        informer.inform(training_data)
        print('Informed.')
        model_handle = informer.get_handle('model')
    print('Now estimating...')
    estimator = KNearNeighEstimator.make_stage(
        name='KNN', model=model_handle,
        hdf5_groupname="", aliases={"output": "pz_knn"}, **cfg)

print(f'Training time: {time.perf_counter() - start_time:.1f}s')

# ── Estimate ──────────────────────────────────────────────────────────────────

print("Estimating photo-z's for test set...")

results = estimator.estimate(test_data)

# ── Save ──────────────────────────────────────────────────────────────────────

print("Saving results...")
ensemble = results.data
z_grid   = np.linspace(0.0, 6.0, 601)

pdfs     = ensemble.pdf(z_grid)
z_median = np.ndarray.flatten(ensemble.ppf(0.5))
z_mode   = np.ndarray.flatten(ensemble.mode(grid=z_grid))
z_true   = test_data['redshift']

np.savez_compressed(args.outfile,
                    z_grid=z_grid, pdfs=pdfs,
                    z_median=z_median, z_mode=z_mode,
                    z_true=z_true)

print(f"Saved to {args.outfile}  ({len(z_true)} galaxies, {time.perf_counter() - start_time:.1f}s total)")
