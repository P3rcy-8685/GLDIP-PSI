"""
DIP vs GLDIP+Ψ — Systematic Comparison
========================================
Section 4: Comparison of Standard DIP and GLDIP+Ψ

Fixes applied:
  - plot_metrics: x positions now match len(INITS) dynamically
  - plot_sweeps:  no INITS mutation; one DIP curve per init; no KeyError
  - plot_table:   ifc dict always built from IC keys
  - plot_visual:  axes always 2D so indexing works with any nrows
  - run_gldip:    removed broken alpha override
  - degrade:      removed incorrect clip on v (Au can exceed [0,1])
"""

import numpy as np
import scipy.sparse as sp
from scipy.signal import wiener as scipy_wiener
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HYPERPARAMETERS  (identical for DIP and GLDIP — fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
TAU      = 1.0001
MAX_ITER = 5000
ETA0     = 0.2
ETA1     = 0.9
STEP     = 1.0
# GLDIP-only graph Laplacian params
MU0, MU1, MU2 = 0.3, 0.25, 2.5
GL_R, GL_SIGMA = 1, 0.2

BG, FG  = '#f0f2f5', '#2c3e50'
IC      = {'Raw': '#e74c3c', 'Wiener': '#27ae60', 'TV': '#2980b9'}
DIP_C   = '#f39c12'
GLDIP_C = '#8e44ad'
INITS   = ['Raw', 'Wiener']          # add 'TV' here to enable it everywhere


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH LAPLACIAN
# ─────────────────────────────────────────────────────────────────────────────

def build_laplacian(image, R=1, sigma=0.2):
    h, w = image.shape
    N = h * w
    x = image.flatten()
    rows, cols, wvals = [], [], []
    for idx in range(N):
        r0, c0 = divmod(idx, w)
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                if dr == 0 and dc == 0:
                    continue
                if abs(dr) + abs(dc) > R:
                    continue
                r1, c1 = r0 + dr, c0 + dc
                if 0 <= r1 < h and 0 <= c1 < w:
                    j = r1 * w + c1
                    rows.append(idx); cols.append(j)
                    wvals.append(np.exp(-((x[idx] - x[j]) ** 2) / sigma))
    W = sp.csr_matrix((wvals, (rows, cols)), shape=(N, N))
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    return (D - W).tocsr()


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

class IdentityOp:
    def fwd(self, x): return x.copy()
    def adj(self, x): return x.copy()


class BlurOp:
    def __init__(self, h, w, sigma=1.5, ks=5):
        self.h, self.w = h, w
        half = ks // 2
        ax = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        k /= k.sum()
        self.kernel = k
        rows, cols, vals = [], [], []
        n = h * w
        for idx in range(n):
            r0, c0 = divmod(idx, w)
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    r1, c1 = r0 + dr, c0 + dc
                    if 0 <= r1 < h and 0 <= c1 < w:
                        rows.append(idx); cols.append(r1 * w + c1)
                        vals.append(k[dr + half, dc + half])
        H = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self.H, self.HT = H.tocsr(), H.T.tocsr()

    def fwd(self, x): return self.H @ x
    def adj(self, x): return self.HT @ x


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def init_raw(v, op, h, w, nvar):
    return np.clip(v.copy(), 0, 1)


def init_wiener(v, op, h, w, nvar):
    img2d = v.reshape(h, w)
    if isinstance(op, IdentityOp):
        res = scipy_wiener(img2d, mysize=7, noise=nvar)
    else:
        kh, kw = op.kernel.shape
        pad = np.zeros((h, w))
        pad[:kh, :kw] = op.kernel
        pad = np.roll(np.roll(pad, -kh // 2, 0), -kw // 2, 1)
        Hf  = np.fft.fft2(pad)
        Vf  = np.fft.fft2(img2d)
        nsr = max(nvar * 80, 0.005)
        res = np.real(np.fft.ifft2(np.conj(Hf) / (np.abs(Hf) ** 2 + nsr) * Vf))
    return np.clip(res.flatten(), 0, 1)


def init_tv(v, op, h, w, nvar):
    img2d = v.reshape(h, w)
    if isinstance(op, BlurOp):
        kh, kw = op.kernel.shape
        pad = np.zeros((h, w))
        pad[:kh, :kw] = op.kernel
        pad = np.roll(np.roll(pad, -kh // 2, 0), -kw // 2, 1)
        Hf  = np.fft.fft2(pad)
        Vf  = np.fft.fft2(img2d)
        nsr = max(nvar * 150, 0.01)
        img2d = np.clip(np.real(np.fft.ifft2(np.conj(Hf) / (np.abs(Hf) ** 2 + nsr) * Vf)), 0, 1)
        weight = 0.05
    else:
        weight = 0.12
    return np.clip(denoise_tv_chambolle(img2d, weight=weight).flatten(), 0, 1)


# Build once so both sweep and scenario can share the same lookup
INIT_FNS = {'Raw': init_raw, 'Wiener': init_wiener, 'TV': init_tv}


# ─────────────────────────────────────────────────────────────────────────────
# DIP  (β = 0, data fidelity only)
# ─────────────────────────────────────────────────────────────────────────────

def run_dip(op, v, u0, delta):
    u = u0.copy().astype(np.float64)
    loss_h, res_h = [], []
    t_stop = MAX_ITER
    for t in range(MAX_ITER):
        resid  = op.fwd(u) - v
        rn     = np.linalg.norm(resid)
        loss_h.append(0.5 * rn ** 2)
        res_h.append(rn)
        if rn <= TAU * delta:
            t_stop = t
            break
        HTr    = op.adj(resid)
        HTr_sq = float(np.dot(HTr, HTr))
        alpha  = np.clip(ETA0 * rn ** 2 / (HTr_sq + 1e-14), 1e-8, ETA1)
        u      = np.clip(u - STEP * alpha * HTr, 0, 1)
    return u, loss_h, res_h, t_stop


# ─────────────────────────────────────────────────────────────────────────────
# GLDIP  (adds β·L·u regularization)
# FIX: removed the broken `alpha = max(0.5, ...)` override — alpha is now
#      fully adaptive per Eq. (3.4), identical to DIP's formula.
# ─────────────────────────────────────────────────────────────────────────────

def run_gldip(op, v, u0, L, delta):
    u = u0.copy().astype(np.float64)
    loss_h, res_h = [], []
    t_stop = MAX_ITER
    for t in range(MAX_ITER):
        resid  = op.fwd(u) - v
        rn     = np.linalg.norm(resid)
        loss_h.append(0.5 * rn ** 2)
        res_h.append(rn)
        if rn <= TAU * delta:
            t_stop = t
            break
        HTr    = op.adj(resid)
        HTr_sq = float(np.dot(HTr, HTr))
        alpha  = np.clip(ETA0 * rn ** 2 / (HTr_sq + 1e-14), 1e-8, ETA1)   # FIX: no override
        Lu     = L @ u
        Lu_n   = np.linalg.norm(Lu)
        beta   = min(MU0 * rn ** 2 / Lu_n, MU1 / Lu_n, MU2) if Lu_n > 1e-12 else 0.0
        u      = np.clip(u - STEP * (alpha * HTr + beta * Lu), 0, 1)
    return u, loss_h, res_h, t_stop


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def mets(gt, est):
    g = np.clip(gt, 0, 1)
    e = np.clip(est, 0, 1)
    return dict(
        PSNR=psnr(g, e, data_range=1.0),
        SSIM=ssim(g, e, data_range=1.0),
        MSE =float(np.mean((g - e) ** 2)),
        MAE =float(np.mean(np.abs(g - e))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEGRADE IMAGE
# FIX: removed np.clip on v — Au can legitimately exceed [0,1] for general
#      operators; clipping corrupts the forward model.
# ─────────────────────────────────────────────────────────────────────────────

def degrade(img, op, noise_level, seed=0):
    rng   = np.random.RandomState(seed)
    n     = img.flatten().shape[0]
    nv    = rng.randn(n)
    nv    = nv / np.linalg.norm(nv) * noise_level
    clean = op.fwd(img.flatten())
    v     = clean + np.linalg.norm(clean) * nv   # FIX: no clip
    delta = noise_level
    return v, delta


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SCENARIO: run all inits × DIP / GLDIP
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(img, op, noise_level, seed=0):
    h, w  = img.shape
    nvar  = noise_level ** 2
    v, delta = degrade(img, op, noise_level, seed)
    results  = {}
    for name in INITS:
        fn = INIT_FNS[name]
        u0 = fn(v, op, h, w, nvar)
        L  = build_laplacian(u0.reshape(h, w), R=GL_R, sigma=GL_SIGMA)

        ud, lhd, rhd, tsd = run_dip  (op, v, u0,    delta)
        ug, lhg, rhg, tsg = run_gldip(op, v, u0, L, delta)

        results[name] = dict(
            u0=u0.reshape(h, w), ud=ud.reshape(h, w), ug=ug.reshape(h, w),
            m_init =mets(img, u0.reshape(h, w)),
            m_dip  =mets(img, ud.reshape(h, w)),
            m_gldip=mets(img, ug.reshape(h, w)),
            loss_dip=lhd, res_dip=rhd, ts_dip=tsd,
            loss_gl =lhg, res_gl =rhg, ts_gl =tsg,
        )
        print(f"  [{name:6s}] Init={results[name]['m_init']['PSNR']:.2f} | "
              f"DIP={results[name]['m_dip']['PSNR']:.2f} | "
              f"GLDIP={results[name]['m_gldip']['PSNR']:.2f} | "
              f"Δ={results[name]['m_gldip']['PSNR']-results[name]['m_dip']['PSNR']:+.2f}dB")
    return v.reshape(h, w), results


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Visual grid: GT | Degraded | Init | DIP | GLDIP
# FIX: axes forced to 2D with np.atleast_2d so single-row case doesn't crash
# ─────────────────────────────────────────────────────────────────────────────

def plot_visual(img, v, results, task_label, save_path):
    nrows = len(INITS)
    fig, axes_raw = plt.subplots(nrows, 5, figsize=(18, 4 * nrows), facecolor=BG,
                                  squeeze=False)   # FIX: squeeze=False → always 2D
    fig.suptitle(f'DIP vs GLDIP+Ψ — {task_label}\nVisual Comparison per Initialization',
                 fontsize=13, fontweight='bold', color=FG, y=1.01)

    col_titles = ['Ground Truth', 'Degraded Input', 'Init (Ψ)', 'DIP', 'GLDIP+Ψ']
    col_colors = ['#27ae60', '#7f8c8d', '#bdc3c7', DIP_C, GLDIP_C]

    for ci, (ct, cc) in enumerate(zip(col_titles, col_colors)):
        axes_raw[0, ci].set_title(ct, fontsize=10, color=cc, fontweight='bold', pad=4)

    for ri, iname in enumerate(INITS):
        r      = results[iname]
        panels = [img, v, r['u0'], r['ud'], r['ug']]
        labels = [
            None, None,
            f"PSNR {r['m_init']['PSNR']:.1f}  SSIM {r['m_init']['SSIM']:.3f}",
            f"PSNR {r['m_dip']['PSNR']:.1f}   SSIM {r['m_dip']['SSIM']:.3f}",
            f"PSNR {r['m_gldip']['PSNR']:.1f}  SSIM {r['m_gldip']['SSIM']:.3f}",
        ]
        label_colors = [None, None, '#7f8c8d', DIP_C, GLDIP_C]

        for ci, (im, lbl, lc) in enumerate(zip(panels, labels, label_colors)):
            ax = axes_raw[ri, ci]
            ax.imshow(np.clip(im, 0, 1), cmap='gray', vmin=0, vmax=1,
                      interpolation='lanczos')
            ax.axis('off')
            bc = IC[iname] if ci == 0 else col_colors[ci]
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(2); s.set_edgecolor(bc)
            if ci == 0:
                ax.set_ylabel(f'Ψ = {iname}', fontsize=10, color=IC[iname],
                              fontweight='bold', rotation=90, labelpad=6)
            if lbl:
                ax.text(0.5, -0.08, lbl, transform=ax.transAxes,
                        ha='center', va='top', fontsize=8, color=lc, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Metric bar charts
# FIX: x positions derived from len(INITS), not hardcoded 3
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics(results_dn, results_db, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor=BG)
    fig.suptitle('DIP vs GLDIP+Ψ — Metric Comparison\nDenoising (top) | Deblurring (bottom)',
                 fontsize=13, fontweight='bold', color=FG)

    metric_info = [
        ('PSNR', 'PSNR (dB) ↑', True),
        ('SSIM', 'SSIM ↑',      True),
        ('MSE',  'MSE ↓',       False),
        ('MAE',  'MAE ↓',       False),
    ]

    for row, (task, res) in enumerate([('Denoising', results_dn), ('Deblurring', results_db)]):
        for col, (mn, ml, higher_better) in enumerate(metric_info):
            ax = axes[row, col]
            ax.set_facecolor(BG)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            x  = np.arange(len(INITS))   # FIX: dynamic length
            bw = 0.28
            v_init  = [res[n]['m_init'][mn]  for n in INITS]
            v_dip   = [res[n]['m_dip'][mn]   for n in INITS]
            v_gldip = [res[n]['m_gldip'][mn] for n in INITS]

            ax.bar(x - bw, v_init,  bw, color='#bdc3c7', edgecolor='white',
                   lw=0.5, label='Init', alpha=0.7)
            ax.bar(x,      v_dip,   bw, color=DIP_C,    edgecolor='white',
                   lw=0.5, label='DIP',  alpha=0.9)
            ax.bar(x + bw, v_gldip, bw, color=GLDIP_C,  edgecolor='white',
                   lw=0.5, label='GLDIP+Ψ', alpha=0.9)

            for xi, (vd, vg) in enumerate(zip(v_dip, v_gldip)):
                diff  = vg - vd
                color = '#1e8449' if (diff > 0) == higher_better else '#c0392b'
                ax.text(xi + bw / 2, max(vd, vg), f'{diff:+.3f}',
                        ha='center', va='bottom', fontsize=7.5,
                        color=color, fontweight='bold')
            for xi, vd in enumerate(v_dip):
                ax.text(xi,      vd, f'{vd:.3f}', ha='center', va='bottom',
                        fontsize=6.5, color='#7f8c8d')
            for xi, vg in enumerate(v_gldip):
                ax.text(xi + bw, vg, f'{vg:.3f}', ha='center', va='bottom',
                        fontsize=6.5, color=GLDIP_C, fontweight='bold')

            ax.set_title(f'{task}: {ml}', fontsize=9, color=FG, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(INITS, fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Convergence curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(results_dn, results_db, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle('DIP vs GLDIP+Ψ — Convergence Histories\n'
                 'Dashed = DIP   Solid = GLDIP+Ψ   Dotted vertical = early stop',
                 fontsize=12, fontweight='bold', color=FG)

    for ri, (task, res) in enumerate([('Denoising', results_dn), ('Deblurring', results_db)]):
        ax_l, ax_r = axes[ri]
        for iname in INITS:
            r = res[iname]; c = IC[iname]
            ax_l.semilogy(r['loss_dip'], color=c, lw=1.8, ls='--', alpha=0.7,
                          label=f'{iname} DIP  (t={r["ts_dip"]})')
            ax_r.plot    (r['res_dip'],  color=c, lw=1.8, ls='--', alpha=0.7)
            ax_l.semilogy(r['loss_gl'],  color=c, lw=2.2, ls='-',
                          label=f'{iname} GLDIP (t={r["ts_gl"]})')
            ax_r.plot    (r['res_gl'],   color=c, lw=2.2, ls='-')
            for ax_, hd, hg in [(ax_l, r['loss_dip'], r['loss_gl']),
                                 (ax_r, r['res_dip'],  r['res_gl'])]:
                if r['ts_dip'] < len(hd):
                    ax_.axvline(r['ts_dip'], color=c, ls=':', lw=0.9, alpha=0.5)
                if r['ts_gl'] < len(hg):
                    ax_.axvline(r['ts_gl'],  color=c, ls=':', lw=0.9, alpha=0.8)

        for ax in [ax_l, ax_r]:
            ax.set_facecolor(BG)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=8)

        ax_l.set_title(f'{task}: Loss ½||Hf−vᵟ||²', fontsize=10, color=FG, fontweight='bold')
        ax_l.set_xlabel('Iteration'); ax_l.set_ylabel('Loss (log scale)')
        ax_l.legend(fontsize=7.5, ncol=2)

        ax_r.set_title(f'{task}: Residual ||Hf−vᵟ||', fontsize=10, color=FG, fontweight='bold')
        ax_r.set_xlabel('Iteration'); ax_r.set_ylabel('Residual norm')
        ax_r.legend(handles=[
            plt.Line2D([0], [0], ls='--', color='gray', lw=1.8, label='DIP'),
            plt.Line2D([0], [0], ls='-',  color='gray', lw=2.2, label='GLDIP+Ψ'),
        ], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=145, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Parameter sweeps
# FIX: no INITS mutation; one DIP curve per init (not just Raw);
#      sweep dict always has all INITS keys so no KeyError
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweeps(img, save_path):
    h, w = img.shape

    noise_levels = [0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25]
    blur_sigmas  = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    fixed_noise  = 0.1

    def _sweep(op_fn, xvals, noise_fn):
        """Run DIP + GLDIP for every init over xvals. op_fn(x) → operator."""
        data = {n: {'dip': [], 'gldip': []} for n in INITS}
        for x in xvals:
            op   = op_fn(x)
            nl   = noise_fn(x)
            nvar = nl ** 2
            v, delta = degrade(img, op, nl)
            for iname in INITS:
                u0 = INIT_FNS[iname](v, op, h, w, nvar)
                L  = build_laplacian(u0.reshape(h, w))
                ud, *_ = run_dip  (op, v, u0,    delta)
                ug, *_ = run_gldip(op, v, u0, L, delta)
                data[iname]['dip'].append(
                    psnr(img, np.clip(ud.reshape(h, w), 0, 1), data_range=1.0))
                data[iname]['gldip'].append(
                    psnr(img, np.clip(ug.reshape(h, w), 0, 1), data_range=1.0))
        return data

    print("\n  Noise sweep (denoising)...")
    sweep_dn   = _sweep(lambda nl: IdentityOp(),
                        noise_levels, lambda nl: nl)
    print("  Noise sweep (deblurring)...")
    sweep_db   = _sweep(lambda nl: BlurOp(h, w, sigma=1.5),
                        noise_levels, lambda nl: nl)
    print("  Blur-sigma sweep...")
    sweep_blur = _sweep(lambda bs: BlurOp(h, w, sigma=bs),
                        blur_sigmas,  lambda bs: fixed_noise)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=BG)
    fig.suptitle('DIP vs GLDIP+Ψ — Parameter Sweeps (PSNR dB)\n'
                 'Solid = GLDIP+Ψ   Dashed = DIP',
                 fontsize=12, fontweight='bold', color=FG)

    configs = [
        (axes[0], noise_levels, sweep_dn,   'Noise level σ',
         'Denoising: PSNR vs noise level'),
        (axes[1], noise_levels, sweep_db,   'Noise level σ',
         'Deblurring: PSNR vs noise level'),
        (axes[2], blur_sigmas,  sweep_blur, 'Blur σ',
         f'Deblurring: PSNR vs blur σ  (noise={fixed_noise})'),
    ]

    for ax, xvals, sweep, xlabel, title in configs:
        ax.set_facecolor(BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for iname in INITS:
            c  = IC[iname]
            vd = sweep[iname]['dip']
            vg = sweep[iname]['gldip']
            ax.plot(xvals, vd, color=c, ls='--', lw=1.8, alpha=0.7,
                    label=f'{iname} DIP')
            ax.plot(xvals, vg, color=c, ls='-',  lw=2.2,
                    label=f'{iname} GLDIP')
            ax.fill_between(xvals, vd, vg, color=c, alpha=0.08)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('PSNR (dB)', fontsize=10)
        ax.set_title(title, fontsize=10, color=FG, fontweight='bold')
        ax.tick_params(labelsize=9)

    # Unified legend
    legend_handles = []
    for iname in INITS:
        legend_handles.append(
            plt.Line2D([0], [0], color=IC[iname], lw=2.2, ls='-',
                       label=f'{iname} GLDIP'))
        legend_handles.append(
            plt.Line2D([0], [0], color=IC[iname], lw=1.8, ls='--', alpha=0.7,
                       label=f'{iname} DIP'))
    axes[2].legend(handles=legend_handles, fontsize=7.5, ncol=2, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=145, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Full metrics table
# FIX: ifc built dynamically from IC keys so it never has a missing key
# ─────────────────────────────────────────────────────────────────────────────

def plot_table(results_dn, results_db, save_path):
    fig, ax = plt.subplots(figsize=(24, 8), facecolor=BG)
    ax.axis('off')
    fig.suptitle('DIP vs GLDIP+Ψ — Complete Metrics Table',
                 fontsize=13, fontweight='bold', color=FG, y=1.02)

    hdrs = ['Task', 'Init (Ψ)',
            'PSNR Init', 'PSNR DIP', 'PSNR GLDIP', 'Δ (GLDIP−DIP)',
            'SSIM Init', 'SSIM DIP', 'SSIM GLDIP', 'Δ SSIM',
            'MSE DIP', 'MSE GLDIP',
            'MAE DIP', 'MAE GLDIP',
            'DIP t*', 'GLDIP t*']

    rows = []
    for tn, res in [('Denoising', results_dn), ('Deblurring', results_db)]:
        for iname in INITS:
            r = res[iname]
            b, d, g = r['m_init'], r['m_dip'], r['m_gldip']
            rows.append([
                tn, iname,
                f"{b['PSNR']:.2f}", f"{d['PSNR']:.2f}", f"{g['PSNR']:.2f}",
                f"{g['PSNR'] - d['PSNR']:+.2f}",
                f"{b['SSIM']:.4f}", f"{d['SSIM']:.4f}", f"{g['SSIM']:.4f}",
                f"{g['SSIM'] - d['SSIM']:+.4f}",
                f"{d['MSE']:.5f}", f"{g['MSE']:.5f}",
                f"{d['MAE']:.5f}", f"{g['MAE']:.5f}",
                str(r['ts_dip']), str(r['ts_gl']),
            ])

    tbl = ax.table(cellText=rows, colLabels=hdrs, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2.8)

    for j in range(len(hdrs)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # FIX: build ifc from whatever keys are actually in IC — no hardcoded list
    ifc = {k: v.replace(')', ', 0.15)').replace('rgb', 'rgba')
           for k, v in IC.items()}
    # simpler: just use light tints manually keyed from IC
    tints = {'Raw': '#fde8e8', 'Wiener': '#e8fde8', 'TV': '#e8eeff'}

    for i, row in enumerate(rows):
        iname = row[1]
        dpsnr = float(row[5])
        dssim = float(row[9])
        for j in range(len(hdrs)):
            cell = tbl[(i + 1, j)]
            if j == 1:
                cell.set_facecolor(tints.get(iname, '#f5f5f5'))
                cell.set_text_props(color=IC[iname], fontweight='bold')
            elif j == 5:
                cell.set_facecolor('#d5f5e3' if dpsnr >= 0 else '#fadbd8')
                cell.set_text_props(fontweight='bold',
                                    color='#1e8449' if dpsnr >= 0 else '#c0392b')
            elif j == 9:
                cell.set_facecolor('#d5f5e3' if dssim >= 0 else '#fadbd8')
                cell.set_text_props(fontweight='bold',
                                    color='#1e8449' if dssim >= 0 else '#c0392b')
            elif j == 4:
                cell.set_text_props(fontweight='bold', color=GLDIP_C)
            elif j == 3:
                cell.set_text_props(color=DIP_C)
            elif i % 2 == 0 and j > 1:
                cell.set_facecolor('#f5f5f5')

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    SZ = 64

    img = None
    for candidate in ['ok.jpg', 'ok.png']:
        if os.path.exists(candidate):
            img = np.array(
                Image.open(candidate).convert('L')
                     .resize((SZ, SZ), Image.Resampling.BILINEAR),
                dtype=np.float32) / 255.0
            print(f"Loaded: {candidate}")
            break
    if img is None:
        raw = rgb2gray(skdata.astronaut())
        h0, w0 = raw.shape; s = min(h0, w0)
        raw = raw[(h0-s)//2:(h0+s)//2, (w0-s)//2:(w0+s)//2]
        img = resize(raw, (SZ, SZ), anti_aliasing=True)
        print("Using fallback: astronaut")

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    h, w = img.shape

    print("\n" + "█"*55)
    print("  DIP vs GLDIP+Ψ — Section 4 Experiments")
    print("█"*55)

    print("\n[Scenario 1] Denoising  σ=0.10")
    op_dn = IdentityOp()
    v_dn, res_dn = run_scenario(img, op_dn, noise_level=0.10)

    print("\n[Scenario 2] Deblurring  blur_σ=1.5  noise=0.10")
    op_db = BlurOp(h, w, sigma=1.5)
    v_db, res_db = run_scenario(img, op_db, noise_level=0.10)

    print("\n[Scenario 3] Heavy blur  blur_σ=2.5  noise=0.08")
    op_hb = BlurOp(h, w, sigma=2.5)
    v_hb, res_hb = run_scenario(img, op_hb, noise_level=0.08)

    print("\n[Scenario 4] High noise denoising  σ=0.20")
    op_hn = IdentityOp()
    v_hn, res_hn = run_scenario(img, op_hn, noise_level=0.20)

    print("\nGenerating plots...")

    plot_visual(img, v_dn, res_dn,
                'Denoising  (H=I, σ=0.10)',
                'outputs/dip_vs_gldip_denoise_visual.png')
    plot_visual(img, v_db, res_db,
                'Deblurring  (H=GaussBlur σ=1.5, noise=0.10)',
                'outputs/dip_vs_gldip_deblur_visual.png')
    plot_visual(img, v_hb, res_hb,
                'Heavy blur  (H=GaussBlur σ=2.5, noise=0.08)',
                'outputs/dip_vs_gldip_heavyblur_visual.png')
    plot_visual(img, v_hn, res_hn,
                'High-noise denoising  (H=I, σ=0.20)',
                'outputs/dip_vs_gldip_highnoise_visual.png')

    plot_metrics(res_dn, res_db,   'outputs/dip_vs_gldip_metrics.png')
    plot_curves (res_dn, res_db,   'outputs/dip_vs_gldip_curves.png')
    plot_sweeps (img,               'outputs/dip_vs_gldip_sweep.png')
    plot_table  (res_dn, res_db,   'outputs/dip_vs_gldip_table.png')

    print("\n" + "="*55)
    print("  All outputs saved to outputs/")
    print("="*55)
