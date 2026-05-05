import numpy as np
import scipy.sparse as sp
from scipy.signal import wiener as scipy_wiener
from scipy.linalg import qr
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
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
TAU       = 1.0001
MAX_ITER  = 3000
ETA0      = 0.2
ETA1      = 0.9
STEP      = 1.0
MU0, MU1, MU2 = 0.3, 0.25, 2.5
GL_R, GL_SIGMA = 1, 0.2

BG, FG    = '#f0f2f5', '#2c3e50'
IC        = {'Raw': '#e74c3c', 'Wiener': '#27ae60', 'TV': '#2980b9'}
DIP_C     = '#f39c12'
GLDIP_C   = '#8e44ad'
INITS     = ['Raw', 'Wiener', 'TV']


# ─────────────────────────────────────────────────────────────────────────────
# ALIGNED OPERATOR  A_a  (diagonal)
#   A_a[i,i] = i - q/2,   q = n // 2
#   Note: some diagonal entries are zero (at i = q/2) and negative (i < q/2).
#   We add a small regularizer ε to avoid exact zeros.
# ─────────────────────────────────────────────────────────────────────────────

class AlignedOp:
    """
    A_a = diag(d),  d[i] = i - q/2  for i = 0,...,n-1
    Adjoint = A_a^T = A_a  (diagonal, so self-adjoint)
    """
    def __init__(self, n, eps=0.5):
        self.n   = n
        q        = n // 2
        self.d   = np.array([i - q / 2.0 for i in range(n)], dtype=np.float64)
        # shift by eps so no entry is exactly zero
        self.d  += eps * np.sign(self.d + 1e-12)
        self.d2  = self.d ** 2           # for pseudo-inverse
        self.name = 'Aligned  $A_a$'

    def fwd(self, x):
        return self.d * x

    def adj(self, x):
        return self.d * x               # diagonal → self-adjoint

    def pseudo_inv(self, y, lam=1e-3):
        """Tikhonov pseudo-inverse: (A^T A + λI)^{-1} A^T y"""
        return (self.d * y) / (self.d2 + lam)


# ─────────────────────────────────────────────────────────────────────────────
# NON-ALIGNED OPERATOR  A_na = H A_a H^T
#   H is a uniformly random orthogonal matrix (Haar measure via QR of Gaussian)
# ─────────────────────────────────────────────────────────────────────────────

class NonAlignedOp:
    """
    A_na = H A_a H^T
    fwd:  x → H diag(d) H^T x
    adj:  y → H diag(d) H^T y   (same, since A_na is symmetric)
    """
    def __init__(self, n, eps=0.5, seed=42):
        self.n    = n
        # Build aligned diag as in AlignedOp
        q         = n // 2
        d         = np.array([i - q / 2.0 for i in range(n)], dtype=np.float64)
        d        += eps * np.sign(d + 1e-12)
        self.d    = d
        self.d2   = d ** 2

        # Sample H uniformly from O(n) via QR of Gaussian matrix
        rng       = np.random.RandomState(seed)
        G         = rng.randn(n, n)
        Q, R_mat  = qr(G)
        # Fix signs so that the decomposition is unique (Haar-distributed)
        signs     = np.sign(np.diag(R_mat))
        self.H    = Q * signs[np.newaxis, :]   # H = Q * sign(diag(R))
        self.HT   = self.H.T
        self.name = 'Non-aligned  $A_{na}$'

    def fwd(self, x):
        # A_na x = H diag(d) H^T x
        return self.H @ (self.d * (self.HT @ x))

    def adj(self, y):
        # A_na is symmetric → adj = fwd
        return self.fwd(y)

    def pseudo_inv(self, y, lam=1e-3):
        """Tikhonov: (A_na^T A_na + λI)^{-1} A_na^T y
           = H (d^2 + λ)^{-1} d H^T y
        """
        HTy = self.HT @ y
        return self.H @ ((self.d * HTy) / (self.d2 + lam))


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH LAPLACIAN (spatial, built from the estimated image u0)
# ─────────────────────────────────────────────────────────────────────────────

def build_laplacian(image, R=1, sigma=0.2):
    h, w = image.shape
    N    = h * w
    x    = image.flatten()
    rows, cols, wvals = [], [], []
    for idx in range(N):
        r0, c0 = divmod(idx, w)
        for dr in range(-R, R+1):
            for dc in range(-R, R+1):
                if dr == 0 and dc == 0: continue
                if abs(dr) + abs(dc) > R: continue
                r1, c1 = r0+dr, c0+dc
                if 0 <= r1 < h and 0 <= c1 < w:
                    j = r1*w + c1
                    rows.append(idx); cols.append(j)
                    wvals.append(np.exp(-((x[idx]-x[j])**2) / sigma))
    W = sp.csr_matrix((wvals, (rows, cols)), shape=(N, N))
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    return (D - W).tocsr()


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATIONS Ψ
# ─────────────────────────────────────────────────────────────────────────────

def init_raw(v, op, h, w, nvar):
    """Pseudo-inverse of A applied to v (Tikhonov)"""
    u = op.pseudo_inv(v, lam=1e-2)
    return np.clip(u, 0, 1)

def init_wiener(v, op, h, w, nvar):
    """
    Wiener-style estimate in the operator's eigenbasis:
      û = (A^T A) / (A^T A + NSR) · A^T v   (element-wise in eigen-domain)
    For aligned: eigenbasis = standard basis → element-wise
    For non-aligned: eigenbasis = columns of H → rotate, filter, rotate back
    """
    if isinstance(op, AlignedOp):
        nsr  = max(nvar * 10, 1e-3)
        denom = op.d2 + nsr
        u    = (op.d2 / denom) * op.pseudo_inv(v, lam=nsr)
    else:
        # Rotate to eigenbasis, apply Wiener filter, rotate back
        HTv  = op.HT @ v
        nsr  = max(nvar * 10, 1e-3)
        filtered = (op.d2 / (op.d2 + nsr)) * (op.d / (op.d2 + 1e-14)) * HTv
        u    = op.H @ filtered
    # Reshape to image and TV-smooth to remove ringing
    img2d = np.clip(u.reshape(h, w), 0, 1)
    img2d = denoise_tv_chambolle(img2d, weight=0.05)
    return np.clip(img2d.flatten(), 0, 1)

def init_tv(v, op, h, w, nvar):
    """Pseudo-inverse followed by TV denoising"""
    u     = op.pseudo_inv(v, lam=5e-3)
    img2d = np.clip(u.reshape(h, w), 0, 1)
    img2d = denoise_tv_chambolle(img2d, weight=0.10)
    return np.clip(img2d.flatten(), 0, 1)

INIT_FNS = {'Raw': init_raw, 'Wiener': init_wiener, 'TV': init_tv}


# ─────────────────────────────────────────────────────────────────────────────
# DIP (β = 0)
# ─────────────────────────────────────────────────────────────────────────────

def run_dip(op, v, u0, delta):
    u = u0.copy().astype(np.float64)
    loss_h, res_h = [], []
    t_stop = MAX_ITER
    for t in range(MAX_ITER):
        resid  = op.fwd(u) - v
        rn     = np.linalg.norm(resid)
        loss_h.append(0.5 * rn**2)
        res_h.append(rn)
        if rn <= TAU * delta:
            t_stop = t; break
        ATr    = op.adj(resid)
        ATr_sq = float(np.dot(ATr, ATr))
        alpha  = np.clip(ETA0 * rn**2 / (ATr_sq + 1e-14), 1e-8, ETA1)
        u      = np.clip(u - STEP * alpha * ATr, 0, 1)
    return u, loss_h, res_h, t_stop


# ─────────────────────────────────────────────────────────────────────────────
# GLDIP (adds β · L_{u0} · u)
# ─────────────────────────────────────────────────────────────────────────────

def run_gldip(op, v, u0, L, delta):
    u = u0.copy().astype(np.float64)
    loss_h, res_h = [], []
    t_stop = MAX_ITER
    for t in range(MAX_ITER):
        resid  = op.fwd(u) - v
        rn     = np.linalg.norm(resid)
        loss_h.append(0.5 * rn**2)
        res_h.append(rn)
        if rn <= TAU * delta:
            t_stop = t; break
        ATr    = op.adj(resid)
        ATr_sq = float(np.dot(ATr, ATr))
        alpha  = np.clip(ETA0 * rn**2 / (ATr_sq + 1e-14), 1e-8, ETA1)
        Lu     = L @ u
        Lu_n   = np.linalg.norm(Lu)
        beta   = min(MU0*rn**2/Lu_n, MU1/Lu_n, MU2) if Lu_n > 1e-12 else 0.0
        u      = np.clip(u - STEP * (alpha * ATr + beta * Lu), 0, 1)
    return u, loss_h, res_h, t_stop


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def mets(gt, est, h, w):
    g = np.clip(gt.flatten(), 0, 1).reshape(h, w)
    e = np.clip(est.flatten(), 0, 1).reshape(h, w)
    return dict(
        PSNR = psnr(g, e, data_range=1.0),
        SSIM = ssim(g, e, data_range=1.0),
        MSE  = float(np.mean((g-e)**2)),
        MAE  = float(np.mean(np.abs(g-e))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEGRADE
# ─────────────────────────────────────────────────────────────────────────────

def degrade(img_flat, op, noise_level, seed=0):
    rng    = np.random.RandomState(seed)
    n      = img_flat.shape[0]
    nv     = rng.randn(n)
    nv     = nv / np.linalg.norm(nv) * noise_level
    Ax     = op.fwd(img_flat)
    v      = Ax + np.linalg.norm(Ax) * nv
    delta  = noise_level           # relative noise level used as δ
    return v, delta


# ─────────────────────────────────────────────────────────────────────────────
# RUN SCENARIO: all inits × DIP / GLDIP for one operator
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(img, op, noise_level, seed=0):
    h, w    = img.shape
    n       = h * w
    nvar    = noise_level**2
    v, delta = degrade(img.flatten(), op, noise_level, seed)

    results = {}
    for name, fn in INIT_FNS.items():
        u0   = fn(v, op, h, w, nvar)
        L    = build_laplacian(u0.reshape(h, w), R=GL_R, sigma=GL_SIGMA)

        ud,  lhd, rhd, tsd = run_dip  (op, v, u0,    delta)
        ug,  lhg, rhg, tsg = run_gldip(op, v, u0, L, delta)

        m_init  = mets(img, u0, h, w)
        m_dip   = mets(img, ud, h, w)
        m_gldip = mets(img, ug, h, w)

        results[name] = dict(
            u0=u0.reshape(h,w), ud=ud.reshape(h,w), ug=ug.reshape(h,w),
            m_init=m_init, m_dip=m_dip, m_gldip=m_gldip,
            loss_dip=lhd, res_dip=rhd, ts_dip=tsd,
            loss_gl =lhg, res_gl =rhg, ts_gl =tsg,
        )
        gain = m_gldip['PSNR'] - m_dip['PSNR']
        print(f"    [{name:6s}]  Init={m_init['PSNR']:.2f}  "
              f"DIP={m_dip['PSNR']:.2f}  GLDIP={m_gldip['PSNR']:.2f}  "
              f"Δ(GLDIP−DIP)={gain:+.2f} dB")

    return v.reshape(h, w), results


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Visual grid
#   Rows: Raw / Wiener / TV
#   Cols: GT | Degraded | Init | DIP | GLDIP
# ─────────────────────────────────────────────────────────────────────────────

def plot_visual(img, v, results, title, save_path):
    fig, axes = plt.subplots(3, 5, figsize=(18, 12), facecolor=BG)
    fig.suptitle(title, fontsize=12, fontweight='bold', color=FG, y=1.01)

    col_titles  = ['Ground Truth', 'Degraded (v=Au+η)', 'Init Ψ(v)', 'DIP', 'GLDIP+Ψ']
    col_colors  = ['#27ae60', '#7f8c8d', '#bdc3c7', DIP_C, GLDIP_C]

    for ci, (ct, cc) in enumerate(zip(col_titles, col_colors)):
        axes[0, ci].set_title(ct, fontsize=10, color=cc, fontweight='bold', pad=4)

    for ri, iname in enumerate(INITS):
        r   = results[iname]
        panels = [img, v, r['u0'], r['ud'], r['ug']]
        annots = [
            None, None,
            f"PSNR {r['m_init']['PSNR']:.2f} / SSIM {r['m_init']['SSIM']:.3f}",
            f"PSNR {r['m_dip']['PSNR']:.2f}  / SSIM {r['m_dip']['SSIM']:.3f}",
            f"PSNR {r['m_gldip']['PSNR']:.2f} / SSIM {r['m_gldip']['SSIM']:.3f}",
        ]
        ann_colors = [None, None, '#7f8c8d', DIP_C, GLDIP_C]

        for ci, (im, ann, ac) in enumerate(zip(panels, annots, ann_colors)):
            ax = axes[ri, ci]
            ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
            ax.axis('off')
            bc = IC[iname] if ci == 0 else col_colors[ci]
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(2); s.set_edgecolor(bc)
            if ci == 0:
                ax.set_ylabel(f'Ψ = {iname}', fontsize=10, color=IC[iname],
                              fontweight='bold', labelpad=6)
            if ann:
                fw = 'bold' if ci >= 3 else 'normal'
                ax.text(0.5, -0.07, ann, transform=ax.transAxes,
                        ha='center', va='top', fontsize=8, color=ac, fontweight=fw)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Side-by-side metric bars: Aligned vs Non-Aligned
#           Each group: Init / DIP / GLDIP per initialization
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_comparison(res_a, res_na, save_path):
    """
    2 rows (Aligned / Non-Aligned) × 4 metric columns.
    Within each panel: 3 groups (Raw/Wiener/TV), 3 bars each (Init/DIP/GLDIP).
    The GLDIP−DIP delta is annotated above each GLDIP bar.
    """
    metric_info = [
        ('PSNR', 'PSNR (dB) ↑', True),
        ('SSIM', 'SSIM ↑',      True),
        ('MSE',  'MSE ↓',       False),
        ('MAE',  'MAE ↓',       False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor=BG)
    fig.suptitle('Aligned ($A_a$) vs Non-Aligned ($A_{na}$) — DIP vs GLDIP+Ψ\n'
                 'Annotated delta = GLDIP − DIP',
                 fontsize=13, fontweight='bold', color=FG)

    for row, (op_label, res) in enumerate([
            ('Aligned  $A_a$  (diagonal, standard-basis eigenvectors)', res_a),
            ('Non-aligned  $A_{na}=HA_aH^T$  (random orthogonal eigenbasis)', res_na)]):
        for col, (mn, ml, hb) in enumerate(metric_info):
            ax = axes[row, col]
            ax.set_facecolor(BG)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            x = np.arange(3); bw = 0.26
            vi   = [res[n]['m_init'][mn]  for n in INITS]
            vd   = [res[n]['m_dip'][mn]   for n in INITS]
            vg   = [res[n]['m_gldip'][mn] for n in INITS]

            ax.bar(x - bw, vi, bw, color='#bdc3c7', edgecolor='white',
                   lw=0.5, label='Init', alpha=0.75)
            ax.bar(x,      vd, bw, color=DIP_C,    edgecolor='white',
                   lw=0.5, label='DIP',  alpha=0.9)
            ax.bar(x + bw, vg, bw, color=GLDIP_C,  edgecolor='white',
                   lw=0.5, label='GLDIP+Ψ', alpha=0.9)

            for xi, (d_val, g_val) in enumerate(zip(vd, vg)):
                diff  = g_val - d_val
                # "better" depends on metric direction
                good  = (diff > 0) == hb
                color = '#1e8449' if good else '#c0392b'
                ymax  = max(d_val, g_val)
                ax.text(xi + bw/2, ymax, f'{diff:+.3f}',
                        ha='center', va='bottom', fontsize=7.5,
                        color=color, fontweight='bold')

            ax.set_title(f'{op_label[:30]}…\n{ml}' if len(op_label) > 30
                         else f'{op_label}\n{ml}',
                         fontsize=8.5, color=FG, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(INITS, fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Convergence curves: Aligned vs Non-Aligned, DIP vs GLDIP
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(res_a, res_na, save_path):
    """
    2 rows (Aligned / Non-Aligned) × 2 cols (Loss / Residual).
    DIP = dashed, GLDIP = solid.  One colour per initialization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle('Convergence: Aligned vs Non-Aligned  |  DIP (- -) vs GLDIP+Ψ (—)',
                 fontsize=12, fontweight='bold', color=FG)

    labels_map = {'Aligned': res_a, 'Non-aligned': res_na}
    for ri, (op_label, res) in enumerate(labels_map.items()):
        ax_l, ax_r = axes[ri]
        for iname in INITS:
            r = res[iname]; c = IC[iname]
            ax_l.semilogy(r['loss_dip'], color=c, lw=1.8, ls='--', alpha=0.65,
                          label=f'{iname} DIP  (t={r["ts_dip"]})')
            ax_l.semilogy(r['loss_gl'],  color=c, lw=2.2, ls='-',
                          label=f'{iname} GLDIP (t={r["ts_gl"]})')
            ax_r.plot(r['res_dip'],      color=c, lw=1.8, ls='--', alpha=0.65)
            ax_r.plot(r['res_gl'],       color=c, lw=2.2, ls='-')

            for ax_, hd, hg, td, tg in [
                    (ax_l, r['loss_dip'], r['loss_gl'],  r['ts_dip'], r['ts_gl']),
                    (ax_r, r['res_dip'],  r['res_gl'],   r['ts_dip'], r['ts_gl'])]:
                if td < len(hd): ax_.axvline(td, color=c, ls=':', lw=0.8, alpha=0.45)
                if tg < len(hg): ax_.axvline(tg, color=c, ls=':', lw=0.9, alpha=0.8)

        for ax in [ax_l, ax_r]:
            ax.set_facecolor(BG)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=8)

        ax_l.set_title(f'{op_label}: Loss  ½||Au−v||²', fontsize=10,
                       color=FG, fontweight='bold')
        ax_l.set_xlabel('Iteration'); ax_l.set_ylabel('Loss (log)')
        ax_l.legend(fontsize=7, ncol=2)

        ax_r.set_title(f'{op_label}: Residual  ||Au−v||', fontsize=10,
                       color=FG, fontweight='bold')
        ax_r.set_xlabel('Iteration'); ax_r.set_ylabel('Residual norm')
        handles = [plt.Line2D([0],[0], ls='--', color='gray', lw=1.8, label='DIP'),
                   plt.Line2D([0],[0], ls='-',  color='gray', lw=2.2, label='GLDIP+Ψ')]
        ax_r.legend(handles=handles, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=145, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Noise-level sweep: PSNR vs δ for both operators
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_sweep(img, op_a, op_na, save_path):
    """
    Sweep noise_level ∈ {0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20}
    Plot PSNR (DIP dashed, GLDIP solid) for each initialization.
    Left panel = Aligned, Right panel = Non-Aligned.
    Shaded fill highlights GLDIP gain over DIP.
    """
    noise_levels = [0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20]
    h, w = img.shape
    n    = h * w

    sweep = {
        'Aligned':     {nm: {'dip': [], 'gldip': []} for nm in INITS},
        'Non-aligned': {nm: {'dip': [], 'gldip': []} for nm in INITS},
    }
    ops = {'Aligned': op_a, 'Non-aligned': op_na}

    for op_name, op in ops.items():
        print(f"  Noise sweep — {op_name}")
        for nl in noise_levels:
            nvar     = nl**2
            v, delta = degrade(img.flatten(), op, nl)
            for iname in INITS:
                u0 = INIT_FNS[iname](v, op, h, w, nvar)
                L  = build_laplacian(u0.reshape(h, w))
                ud, *_ = run_dip  (op, v, u0,    delta)
                ug, *_ = run_gldip(op, v, u0, L, delta)
                sweep[op_name][iname]['dip'].append(
                    psnr(img, np.clip(ud.reshape(h,w), 0, 1), data_range=1.0))
                sweep[op_name][iname]['gldip'].append(
                    psnr(img, np.clip(ug.reshape(h,w), 0, 1), data_range=1.0))
            print(f"    noise={nl:.2f} ✓")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle('PSNR vs Noise Level δ  |  Aligned ($A_a$) and Non-Aligned ($A_{na}$)\n'
                 'Solid = GLDIP+Ψ   Dashed = DIP   Shaded = GLDIP gain',
                 fontsize=12, fontweight='bold', color=FG)

    for ax, op_name in zip(axes, ['Aligned', 'Non-aligned']):
        ax.set_facecolor(BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for iname in INITS:
            c   = IC[iname]
            vd  = sweep[op_name][iname]['dip']
            vg  = sweep[op_name][iname]['gldip']
            ax.plot(noise_levels, vd, color=c, ls='--', lw=1.8, alpha=0.7,
                    label=f'{iname} DIP')
            ax.plot(noise_levels, vg, color=c, ls='-',  lw=2.2,
                    label=f'{iname} GLDIP')
            ax.fill_between(noise_levels, vd, vg, color=c, alpha=0.09)

        ax.set_xlabel('Noise level δ', fontsize=11)
        ax.set_ylabel('PSNR (dB)',     fontsize=11)
        title_str = ('Aligned $A_a$ (diagonal)'
                     if op_name == 'Aligned'
                     else 'Non-aligned $A_{na} = HA_aH^T$')
        ax.set_title(title_str, fontsize=11, color=FG, fontweight='bold')
        ax.tick_params(labelsize=9)

    handles = []
    for iname in INITS:
        handles += [
            plt.Line2D([0],[0], color=IC[iname], lw=2.2, ls='-',
                       label=f'{iname} GLDIP'),
            plt.Line2D([0],[0], color=IC[iname], lw=1.8, ls='--', alpha=0.7,
                       label=f'{iname} DIP'),
        ]
    axes[1].legend(handles=handles, fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=145, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Full metrics table
# ─────────────────────────────────────────────────────────────────────────────

def plot_table(res_a, res_na, save_path):
    fig, ax = plt.subplots(figsize=(26, 8.5), facecolor=BG)
    ax.axis('off')
    fig.suptitle('Aligned vs Non-Aligned — DIP vs GLDIP+Ψ — Complete Metrics',
                 fontsize=13, fontweight='bold', color=FG, y=1.02)

    hdrs = ['Operator', 'Init (Ψ)',
            'PSNR Init', 'PSNR DIP', 'PSNR GLDIP', 'Δ PSNR',
            'SSIM Init', 'SSIM DIP', 'SSIM GLDIP', 'Δ SSIM',
            'MSE DIP',   'MSE GLDIP',
            'MAE DIP',   'MAE GLDIP',
            'DIP t*',    'GLDIP t*']

    rows = []
    for op_label, res in [('Aligned $A_a$', res_a),
                           ('Non-aligned $A_{na}$', res_na)]:
        for iname in INITS:
            r = res[iname]
            b, d, g = r['m_init'], r['m_dip'], r['m_gldip']
            rows.append([
                op_label, iname,
                f"{b['PSNR']:.2f}", f"{d['PSNR']:.2f}", f"{g['PSNR']:.2f}",
                f"{g['PSNR']-d['PSNR']:+.2f}",
                f"{b['SSIM']:.4f}", f"{d['SSIM']:.4f}", f"{g['SSIM']:.4f}",
                f"{g['SSIM']-d['SSIM']:+.4f}",
                f"{d['MSE']:.5f}", f"{g['MSE']:.5f}",
                f"{d['MAE']:.5f}", f"{g['MAE']:.5f}",
                str(r['ts_dip']), str(r['ts_gl']),
            ])

    tbl = ax.table(cellText=rows, colLabels=hdrs, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2.8)

    for j in range(len(hdrs)):
        tbl[(0,j)].set_facecolor('#2c3e50')
        tbl[(0,j)].set_text_props(color='white', fontweight='bold')

    ifc = {'Raw':'#fde8e8','Wiener':'#e8fde8','TV':'#e8eeff'}
    for i, row in enumerate(rows):
        iname  = row[1]
        dpsnr  = float(row[5])
        dssim  = float(row[9])
        for j in range(len(hdrs)):
            cell = tbl[(i+1, j)]
            if j == 1:
                cell.set_facecolor(ifc[iname])
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

    # --- Load image ---
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
        img = resize(raw, (SZ, SZ), anti_aliasing=True).astype(np.float32)
        print("Using fallback: astronaut")

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    h, w = img.shape
    n    = h * w

    print(f"Image: {SZ}×{SZ}, n={n}")

    # ── Build operators ──────────────────────────────────────────────────────
    op_a  = AlignedOp(n,    eps=0.5)
    op_na = NonAlignedOp(n, eps=0.5, seed=42)

    print(f"\nOperator A_a  : diagonal, d ∈ [{op_a.d.min():.2f}, {op_a.d.max():.2f}]")
    print(f"Operator A_na : H A_a H^T, ||H||_F = {np.linalg.norm(op_na.H):.2f} "
          f"(expected {np.sqrt(n):.2f}), "
          f"||H H^T - I||_F = {np.linalg.norm(op_na.H @ op_na.HT - np.eye(n)):.2e}")

    NOISE = 0.10

    # ── Scenario A: Aligned operator ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Aligned  A_a  |  noise={NOISE}")
    print(f"{'='*55}")
    v_a, res_a = run_scenario(img, op_a, noise_level=NOISE)

    # ── Scenario B: Non-Aligned operator ─────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Non-aligned  A_na  |  noise={NOISE}")
    print(f"{'='*55}")
    v_na, res_na = run_scenario(img, op_na, noise_level=NOISE)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_visual(img, v_a,  res_a,
                f'Aligned $A_a$ — DIP vs GLDIP+Ψ  (noise={NOISE})',
                'outputs/aligned_visual.png')

    plot_visual(img, v_na, res_na,
                f'Non-aligned $A_{{na}}=HA_aH^T$ — DIP vs GLDIP+Ψ  (noise={NOISE})',
                'outputs/nonaligned_visual.png')

    plot_metrics_comparison(res_a, res_na,
                            'outputs/aligned_vs_nonaligned_metrics.png')

    plot_curves(res_a, res_na,
                'outputs/aligned_vs_nonaligned_curves.png')

    plot_noise_sweep(img, op_a, op_na,
                     'outputs/aligned_vs_nonaligned_sweep.png')

    plot_table(res_a, res_na,
               'outputs/aligned_vs_nonaligned_table.png')

    print("\n" + "="*55)
    print("  Done — all outputs in outputs/")
    print("="*55)