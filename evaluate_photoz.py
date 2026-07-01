"""
Photo-z evaluation script.
Loads .npz output files from any of the photoz_*.py scripts and computes
a suite of point-estimate and PDF-based metrics.

Usage:
    python evaluate_photoz.py --files flexzboost=fzb.npz pzflow=pzf.npz gpz=gpz.npz bpz=bpz.npz
"""

import argparse
import numpy as np
import qp
import qp.metrics as qpm
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

parser = argparse.ArgumentParser(description='Evaluate and compare photo-z outputs.')
parser.add_argument('--files', nargs='+', metavar='NAME=FILE',
                    help='Named .npz files, e.g. flexzboost=fzb_output.npz pzflow=pzf_output.npz')
parser.add_argument('--outdir', default='.', help='Directory to save plots')
args = parser.parse_args()

def samples_to_pdfs(samples, z_grid):
    """Convert per-galaxy posterior samples to a PDF evaluated on z_grid."""
    dz = z_grid[1] - z_grid[0]
    bin_edges = np.append(z_grid - dz / 2, z_grid[-1] + dz / 2)
    pdfs = np.array([np.histogram(s, bins=bin_edges, density=True)[0]
                     for s in samples])
    return pdfs


# Parse name=file pairs
results = {}
for item in args.files:
    name, path = item.split('=')
    data = np.load(path)
    z_grid = data['z_grid']

    if 'pdfs' in data.files:
        pdfs = data['pdfs']
    elif 'samples' in data.files:
        print(f"  [{name}] No pdfs found — estimating from samples...")
        pdfs = samples_to_pdfs(data['samples'], z_grid)
    else:
        raise KeyError(f"{path} contains neither 'pdfs' nor 'samples'")

    # Clip any small negative values produced by the estimator
    pdfs = np.clip(pdfs, 0, None)

    # Remove galaxies where the estimator failed (NaN PDF or NaN point estimates)
    valid = (~np.isnan(pdfs).any(axis=1)
             & ~np.isnan(data['z_median'])
             & ~np.isnan(data['z_mode']))
    n_removed = int((~valid).sum())
    if n_removed:
        print(f"  [{name}] Dropping {n_removed} galaxies with NaN PDFs or point estimates")
    pdfs = pdfs[valid]

    results[name] = {
        'z_true':   data['z_true'][valid],
        'z_median': data['z_median'][valid],
        'z_mode':   data['z_mode'][valid],
        'z_grid':   z_grid,
        'pdfs':     pdfs,
    }
    print(f"Loaded {name}: {path}  ({valid.sum()} galaxies)")


def deltaz(z_phot, z_true):
    return (z_phot - z_true) / (1.0 + z_true)


def point_metrics(z_phot, z_true):
    dz = deltaz(z_phot, z_true)
    bias = np.mean(dz)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    outlier_frac = np.mean(np.abs(dz) > 0.15)
    return bias, nmad, outlier_frac


def pdf_metrics(pdfs, z_grid, z_true):
    """Compute PDF-based metrics using qp."""
    ens = qp.Ensemble(qp.interp, data=dict(xvals=z_grid, yvals=pdfs))

    # PIT values
    pit_obj = qpm.PIT(ens, z_true)
    pit_vals = pit_obj.pit_samps

    # KS statistic on PIT (deviation from uniform)
    from scipy.stats import kstest
    ks_stat, ks_p = kstest(pit_vals, 'uniform')

    # CDE loss (lower = better)
    cde_metric = qpm.CDELossMetric(eval_grid=z_grid)
    cde_loss = cde_metric.evaluate(ens, z_true)

    # KLD between stacked P(z) and true n(z) histogram
    dz = z_grid[1] - z_grid[0]
    stacked_pz = pdfs.mean(axis=0)
    stacked_pz /= stacked_pz.sum() * dz
    true_nz, _ = np.histogram(z_true, bins=len(z_grid),
                               range=(z_grid[0], z_grid[-1]), density=True)
    # KLD requires no zeros
    eps = 1e-10
    kld = np.sum(true_nz * np.log((true_nz + eps) / (stacked_pz + eps))) * dz

    # Wasserstein distance between stacked P(z) and true n(z)
    wass = wasserstein_distance(z_grid, z_grid,
                                u_weights=stacked_pz, v_weights=true_nz)

    return pit_vals, ks_stat, cde_loss, kld, wass


# ── Print summary table ──────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'Method':<15} {'Bias':>8} {'NMAD':>8} {'Outlier%':>10} "
      f"{'PIT KS':>8} {'CDE Loss':>10} {'KLD':>8} {'Wass':>8}")
print("="*80)

pit_data = {}
for name, data in results.items():
    z_true = data['z_true']
    z_phot = data['z_median']
    z_grid = data['z_grid']
    pdfs   = data['pdfs']

    bias, nmad, outlier = point_metrics(z_phot, z_true)
    pit_vals, ks_stat, cde_loss, kld, wass = pdf_metrics(pdfs, z_grid, z_true)
    pit_data[name] = pit_vals

    print(f"{name:<15} {bias:>8.4f} {nmad:>8.4f} {outlier*100:>9.2f}% "
          f"{ks_stat:>8.4f} {cde_loss:>10.4f} {kld:>8.4f} {wass:>8.4f}")

print("="*80)
print("Metrics: Bias & NMAD closer to 0 = better | Outlier% lower = better |")
print("         PIT KS closer to 0 = better (uniform PIT) |")
print("         CDE Loss, KLD, Wasserstein lower = better\n")


# ── Plots ─────────────────────────────────────────────────────────────────────
colors = plt.cm.tab10.colors
name_list = list(results.keys())

fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), sharey=True)
if len(results) == 1:
    axes = [axes]
fig.suptitle('PIT Histograms (uniform = perfect calibration)', fontsize=13)
for ax, (name, data), color in zip(axes, results.items(), colors):
    pit_vals = pit_data[name]
    ax.hist(pit_vals, bins=20, range=(0, 1), color=color, alpha=0.7, density=True)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1)
    ax.set_title(name)
    ax.set_xlabel('PIT value')
axes[0].set_ylabel('Density')
plt.tight_layout()
plt.savefig(f'{args.outdir}/pit_histograms.png', dpi=150)
print(f"Saved: {args.outdir}/pit_histograms.png")

# z_phot vs z_true scatter
fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
if len(results) == 1:
    axes = [axes]
fig.suptitle('Photo-z vs True Redshift', fontsize=13)
for ax, (name, data), color in zip(axes, results.items(), colors):
    z_true = data['z_true']
    z_phot = data['z_median']
    ax.hexbin(z_true, z_phot, gridsize=60, cmap='Blues', mincnt=1)
    zlim = (0, data['z_grid'].max())
    ax.plot(zlim, zlim, 'r--', linewidth=1)
    ax.set_xlim(zlim); ax.set_ylim(zlim)
    ax.set_title(name)
    ax.set_xlabel('z_true')
axes[0].set_ylabel('z_phot (median)')
plt.tight_layout()
plt.savefig(f'{args.outdir}/zphot_vs_ztrue.png', dpi=150)
print(f"Saved: {args.outdir}/zphot_vs_ztrue.png")

# Stacked n(z) comparison — use a shared fine grid for true n(z)
fig, ax = plt.subplots(figsize=(8, 4))
z_true_ref = list(results.values())[0]['z_true']
shared_grid = np.linspace(0, 6, 200)
true_nz, _ = np.histogram(z_true_ref, bins=len(shared_grid),
                           range=(shared_grid[0], shared_grid[-1]), density=True)
ax.fill_between(shared_grid, true_nz, alpha=0.3, color='gray', label='True n(z)')
for (name, data), color in zip(results.items(), colors):
    z_grid = data['z_grid']
    dz = z_grid[1] - z_grid[0]
    stacked = data['pdfs'].mean(axis=0)
    stacked /= stacked.sum() * dz
    ax.plot(z_grid, stacked, label=name, color=color)
ax.set_xlabel('Redshift')
ax.set_ylabel('n(z)')
ax.set_title('Stacked P(z) vs True n(z)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{args.outdir}/nz_comparison.png', dpi=150)
print(f"Saved: {args.outdir}/nz_comparison.png")

# Δz distribution
fig, ax = plt.subplots(figsize=(8, 4))
for (name, data), color in zip(results.items(), colors):
    dz_vals = deltaz(data['z_median'], data['z_true'])
    ax.hist(dz_vals, bins=100, range=(-0.5, 0.5), histtype='step',
            density=True, label=name, color=color, linewidth=1.5)
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel(r'$\Delta z = (z_{phot} - z_{true}) / (1 + z_{true})$')
ax.set_ylabel('Density')
ax.set_title(r'$\Delta z$ Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(f'{args.outdir}/deltaz_distribution.png', dpi=150)
print(f"Saved: {args.outdir}/deltaz_distribution.png")
