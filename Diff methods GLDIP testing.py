
import numpy as np
import scipy.sparse as sp
from scipy.signal import wiener
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import os
from PIL import Image
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH LAPLACIAN (Eqs 2.1–2.4)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_laplacian(image, R=1, sigma=0.2):
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
                    jdx = r1 * w + c1
                    wij = np.exp(-((x[idx] - x[jdx]) ** 2) / sigma)
                    rows.append(idx); cols.append(jdx); wvals.append(wij)
    W = sp.csr_matrix((wvals, (rows, cols)), shape=(N, N))
    D = sp.diags(np.array(W.sum(axis=1)).flatten())
    return (D - W).tocsr()


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

class IdentityOp:
    def __init__(self, n):
        self.n = n
    def fwd(self, x): return x.copy()
    def adj(self, x): return x.copy()


class BlurOp:
    def __init__(self, h, w, sigma=1.5, ks=5):
        self.h, self.w, self.n, self.sigma = h, w, h*w, sigma
        half = ks // 2
        ax = np.arange(-half, half+1)
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx**2+yy**2)/(2*sigma**2)); k /= k.sum()
        self.kernel = k
        rows, cols, vals = [], [], []
        for idx in range(self.n):
            r0, c0 = divmod(idx, w)
            for dr in range(-half, half+1):
                for dc in range(-half, half+1):
                    r1, c1 = r0+dr, c0+dc
                    if 0<=r1<h and 0<=c1<w:
                        jdx = r1*w+c1
                        rows.append(idx); cols.append(jdx)
                        vals.append(k[dr+half, dc+half])
        H = sp.csr_matrix((vals,(rows,cols)), shape=(self.n,self.n))
        self.H, self.HT = H.tocsr(), H.T.tocsr()
    def fwd(self, x): return self.H @ x
    def adj(self, x): return self.HT @ x


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATIONS Ψ
# ─────────────────────────────────────────────────────────────────────────────

def init_raw(v, op, h, w, nvar):
    return np.clip(v.copy(), 0, 1)


def init_wiener(v, op, h, w, nvar):
    img2d = v.reshape(h, w)
    if isinstance(op, IdentityOp):
        res = wiener(img2d, mysize=7, noise=nvar)
    else:
        kh, kw = op.kernel.shape
        pad = np.zeros((h, w))
        pad[:kh, :kw] = op.kernel
        pad = np.roll(np.roll(pad, -kh//2, 0), -kw//2, 1)
        Hf = np.fft.fft2(pad); Vf = np.fft.fft2(img2d)
        nsr = max(nvar * 80, 0.005)
        res = np.real(np.fft.ifft2(np.conj(Hf)/(np.abs(Hf)**2+nsr)*Vf))
    return np.clip(res.flatten(), 0, 1)


def init_tv(v, op, h, w, nvar):
    img2d = v.reshape(h, w)
    if isinstance(op, BlurOp):
        kh, kw = op.kernel.shape
        pad = np.zeros((h, w))
        pad[:kh, :kw] = op.kernel
        pad = np.roll(np.roll(pad, -kh//2, 0), -kw//2, 1)
        Hf = np.fft.fft2(pad); Vf = np.fft.fft2(img2d)
        nsr = max(nvar * 150, 0.01)
        img2d = np.clip(np.real(np.fft.ifft2(np.conj(Hf)/(np.abs(Hf)**2+nsr)*Vf)), 0, 1)
        weight = 0.05
    else:
        weight = 0.12
    return np.clip(denoise_tv_chambolle(img2d, weight=weight).flatten(), 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# GLDIP CORE (Algorithm 3.1, image-space formulation)
# ─────────────────────────────────────────────────────────────────────────────

def gldip(op, v_delta, u0, L, delta,
          tau=1.1, max_iter=1500,
          eta0=1.2, eta1=0.5,
          mu0=0.08, mu1=0.15, mu2=2.0,
          step=0.85, verbose=True):
    """
    GLDIP in image space.

    u_{t+1} = u_t - step * [α_t H^T(Hu_t - v^δ) + β_t L_{u0} u_t]

    α_t, β_t adapted per Eqs (3.4)-(3.5).
    Early stopping via discrepancy principle (Eq. 1.11): stop when ||Hu_t - v^δ|| ≤ τδ.
    """
    u = u0.copy().astype(np.float64)
    loss_h, res_h = [], []
    t_stop = max_iter

    for t in range(max_iter):
        Hu = op.fwd(u)
        resid = Hu - v_delta
        rn = np.linalg.norm(resid)
        loss_h.append(0.5 * rn**2)
        res_h.append(rn)

        if rn <= tau * delta:
            t_stop = t
            if verbose:
                print(f"    ✓ Early stop t={t:3d} | ||r||={rn:.4f} ≤ τδ={tau*delta:.4f}")
            break

        # Adaptive α (Eq. 3.4)
        HTr = op.adj(resid)
        HTr_sq = float(np.dot(HTr, HTr))
        rn_sq = rn**2
        alpha = max(0.8,np.clip(eta0 * rn_sq / (HTr_sq + 1e-14), 1e-8, eta1))

        # Graph Laplacian term β (Eq. 3.5)
        Lu = L @ u
        Lu_n = np.linalg.norm(Lu)
        if Lu_n > 1e-12:
            beta = min(mu0 * rn_sq / Lu_n, mu1 / Lu_n, mu2)
        else:
            beta = 0.0

        # Gradient step
        grad = alpha * HTr + beta * Lu
        u = np.clip(u - step * grad, 0, 1)

    else:
        if verbose:
            print(f"    Max iter | ||r||={res_h[-1]:.4f} | τδ={tau*delta:.4f}")

    return u, loss_h, res_h, t_stop


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def mets(gt, est):
    g = np.clip(gt.reshape(gt.shape), 0, 1)
    e = np.clip(est.reshape(gt.shape), 0, 1)
    return dict(
        PSNR=psnr(g, e, data_range=1.0),
        SSIM=ssim(g, e, data_range=1.0),
        MSE=float(np.mean((g-e)**2)),
        MAE=float(np.mean(np.abs(g-e)))
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_task(img, task, noise_level):
    h, w = img.shape
    n = h * w
    rng = np.random.RandomState(0)
    if task == 'denoise':
        op = IdentityOp(n)
        noise_vector=rng.randn(n)
        noise = noise_vector/np.linalg.norm(noise_vector) * noise_level
        v_d = np.clip(img.flatten() + np.linalg.norm(img.flatten()) * noise, 0, 1)
        # delta = float(np.linalg.norm(noise))
        nvar = noise_level**2
    elif task == 'deblur':
        op = BlurOp(h, w, sigma=1.5, ks=5)
        blurred = op.fwd(img.flatten())
        noise_vector=rng.randn(n)
        noise = noise_vector/np.linalg.norm(noise_vector) * noise_level
        v_d = np.clip(blurred + np.linalg.norm(blurred) * noise, 0, 1)
        # delta = float(np.linalg.norm(noise))
        nvar = noise_level**2
    init_fns = {'Raw': init_raw, 'Wiener': init_wiener, 'TV': init_tv}
    results = {}

    for name, fn in init_fns.items():
        print(f"\n  ── Ψ = {name} ──")
        u0 = fn(v_d, op, h, w, nvar)
        mb = mets(img, u0.reshape(h, w))
        print(f"    Before: PSNR={mb['PSNR']:.2f}  SSIM={mb['SSIM']:.4f}  "
              f"MSE={mb['MSE']:.5f}  MAE={mb['MAE']:.5f}")

        L = build_graph_laplacian(u0.reshape(h, w), R=1, sigma=0.2)

        uf, lh, rh, ts = gldip(op, v_d, u0, L, np.linalg.norm(img.flatten())*noise_level,
                                tau=1.0001, max_iter=5000,
                                eta0=0.2, eta1=0.9,
                                mu0=0.3, mu1=0.25, mu2=2.5,
                                step=1, verbose=True)

        ma = mets(img, uf.reshape(h, w))
        print(f"    After:  PSNR={ma['PSNR']:.2f}  SSIM={ma['SSIM']:.4f}  "
              f"MSE={ma['MSE']:.5f}  MAE={ma['MAE']:.5f}")
        print(f"    Δ PSNR={ma['PSNR']-mb['PSNR']:+.2f}  "
              f"Δ SSIM={ma['SSIM']-mb['SSIM']:+.4f}  t_stop={ts}")

        results[name] = dict(u0=u0.reshape(h,w), ug=uf.reshape(h,w),
                             before=mb, after=ma, loss=lh, res=rh, ts=ts)

    return v_d.reshape(h, w), results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

IC = {'Raw': '#e74c3c', 'Wiener': '#27ae60', 'TV': '#2980b9'}
BG, FG = '#f0f2f5', '#2c3e50'
INITS = ['Raw', 'Wiener', 'TV']


def fig_main(img, v_dn, r_dn, v_db, r_db):
    fig = plt.figure(figsize=(26, 17), facecolor=BG)
    fig.suptitle('GLDIP+Ψ — Graph Laplacian Assisted Deep Image Prior\n'
                 'Denoising & Deblurring | Initializations: Raw / Wiener / TV',
                 fontsize=14, fontweight='bold', color=FG, y=0.995)

    outer = gridspec.GridSpec(2, 1, fig, hspace=0.46,
                               top=0.945, bottom=0.04, left=0.02, right=0.98)

    tasks = [('', v_dn, r_dn),
             ('', v_db, r_db)]

    for ri, (tlabel, vd, res) in enumerate(tasks):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, outer[ri], hspace=0.12)
        img_gs = gridspec.GridSpecFromSubplotSpec(1, 8, inner[0], wspace=0.04)

        panels = [img, vd]
        ctitles = ['Ground\nTruth', 'Degraded\nInput']
        bcolors = ['#27ae60', '#7f8c8d']
        for iname in INITS:
            panels += [res[iname]['u0'], res[iname]['ug']]
            ctitles += [f'{iname}\n(Before)', f'{iname}\n(After GLDIP)']
            bcolors += ['#bdc3c7', IC[iname]]

        for ci, (im, ct, bc) in enumerate(zip(panels, ctitles, bcolors)):
            ax = fig.add_subplot(img_gs[0, ci])
            ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
            ax.axis('off')
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(2.5); s.set_edgecolor(bc)
            ax.set_title(ct, fontsize=7.5, color=FG,
                         fontweight='bold' if ci in [0,3,5,7] else 'normal', pad=2.5)

            if ci >= 2:
                iidx = (ci-2)//2
                after = (ci-2)%2 == 1
                iname = INITS[iidx]
                m = res[iname]['after'] if after else res[iname]['before']
                tc = IC[iname] if after else '#95a5a6'
                ax.text(0.5, -0.14,
                        f"PSNR {m['PSNR']:.1f}dB\nSSIM {m['SSIM']:.3f}",
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=6.8, color=tc,
                        fontweight='bold' if after else 'normal')

        # bar charts
        ax_lbl = fig.add_subplot(inner[1])
        ax_lbl.axis('off')
        ax_lbl.text(0.0, 1.02, f'  {tlabel}',
                    transform=ax_lbl.transAxes, fontsize=9.5,
                    fontweight='bold', color=FG, va='bottom')

        bar_gs = gridspec.GridSpecFromSubplotSpec(1, 4, inner[1], wspace=0.38)
        x = np.arange(3); w = 0.38
        for mi, (mn, ml) in enumerate(zip(['PSNR','SSIM','MSE','MAE'],
                                           ['PSNR (dB) ↑','SSIM ↑','MSE ↓','MAE ↓'])):
            ax_b = fig.add_subplot(bar_gs[0, mi])
            ax_b.set_facecolor(BG)
            vb = [res[n]['before'][mn] for n in INITS]
            va = [res[n]['after'][mn] for n in INITS]
            barsb = ax_b.bar(x-w/2, vb, w, color='#bdc3c7', edgecolor='white', lw=0.5, label='Before')
            barsa = ax_b.bar(x+w/2, va, w, color=[IC[n] for n in INITS],
                             edgecolor='white', lw=0.5, alpha=0.92, label='After GLDIP')
            for b in barsb:
                ax_b.text(b.get_x()+b.get_width()/2, b.get_height(),
                          f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=5.8, color='#7f8c8d')
            for b in barsa:
                ax_b.text(b.get_x()+b.get_width()/2, b.get_height(),
                          f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=5.8, color=FG, fontweight='bold')
            ax_b.set_title(ml, fontsize=8.5, color=FG, fontweight='bold')
            ax_b.set_xticks(x); ax_b.set_xticklabels(INITS, fontsize=8)
            ax_b.tick_params(axis='y', labelsize=7)
            ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
            if mi == 0: ax_b.legend(fontsize=7)

    plt.savefig('outputs/gldip_results.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    print("  → gldip_results.png saved")

def fig_main_portrait_images(img, v_dn, r_dn, v_db, r_db):
    fig = plt.figure(figsize=(10, 16), facecolor=BG)

    fig.suptitle(
        'GLDIP+Ψ — Denoising and Deblurring\n'
        'Initializations: Raw / Wiener / TV',
        fontsize=14, fontweight='bold', color=FG, y=0.98
    )

    # 4 rows × 4 columns = portrait layout
    gs = gridspec.GridSpec(4, 4, figure=fig,
                        hspace=0.25, wspace=0.05,
                        top=0.94, bottom=0.05,
                        left=0.06, right=0.98)

    def plot_block(start_row, vd, res, label):
        panels = [img, vd]
        ctitles = ['Ground Truth', f'{label} Input']

        for iname in INITS:
            panels += [res[iname]['u0'], res[iname]['ug']]
            ctitles += [f'{iname} (Before)', f'{iname} (After)']

        # split into 2 rows × 4 cols
        for i in range(8):
            r = start_row + (i // 4)
            c = i % 4

            ax = fig.add_subplot(gs[r, c])
            ax.imshow(panels[i], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            ax.set_title(ctitles[i], fontsize=8, color=FG, pad=3)

            # metrics
            if i >= 2:
                iidx = (i - 2) // 2
                after = (i - 2) % 2 == 1
                iname = INITS[iidx]

                m = res[iname]['after'] if after else res[iname]['before']

                ax.text(
                    0.5, -0.12,
                    f"PSNR {m['PSNR']:.1f} dB\nSSIM {m['SSIM']:.3f}",
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=7,
                    color=IC[iname] if after else '#7f8c8d',
                    fontweight='bold' if after else 'normal'
                )

        # section label
        fig.text(0.01, 0.75 if start_row == 0 else 0.32,
                label, fontsize=11,
                fontweight='bold', color=FG,
                rotation=90, va='center')

    # Top half: Denoising
    plot_block(0, v_dn, r_dn, 'Denoising')

    # Bottom half: Deblurring
    plot_block(2, v_db, r_db, 'Deblurring')

    plt.savefig('outputs/gldip_portrait.png',
                dpi=300, bbox_inches='tight', facecolor=BG)

    print("→ gldip_portrait.png saved")

def fig_convergence(r_dn, r_db):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.suptitle('GLDIP Convergence: Loss & Residual Histories',
                 fontsize=13, fontweight='bold', color=FG)

    for ri, (task, res) in enumerate([('Denoising', r_dn), ('Deblurring', r_db)]):
        al, ar = axes[ri]
        for iname, r in res.items():
            c = IC[iname]; ts = r['ts']
            al.semilogy(r['loss'], color=c, label=iname, lw=2)
            ar.plot(r['res'], color=c, label=iname, lw=2)
            for ax in [al, ar]:
                if ts < len(r['loss']):
                    ax.axvline(ts, color=c, ls='--', lw=1, alpha=0.6)

        for ax in [al, ar]:
            ax.set_facecolor(BG)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=8)

        al.set_title(f'{task}: Loss', fontsize=10, color=FG, fontweight='bold')
        al.set_xlabel('Iteration', fontsize=9); al.set_ylabel('½||Hf-vᵟ||² (log)', fontsize=9)
        al.legend(fontsize=9)

        ar.set_title(f'{task}: Residual Norm', fontsize=10, color=FG, fontweight='bold')
        ar.set_xlabel('Iteration', fontsize=9); ar.set_ylabel('||Hf-vᵟ||', fontsize=9)
        ar.legend(fontsize=9)

        # annotate stopping threshold
        ar.axhline(y=min(r['res'][-1] for r in res.values()),
                   color='gray', ls=':', lw=1, alpha=0.5, label='τδ')

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig('outputs/gldip_convergence.png',
                dpi=140, bbox_inches='tight', facecolor=BG)
    print("  → gldip_convergence.png saved")


def fig_table(r_dn, r_db):
    fig, ax = plt.subplots(figsize=(18, 6.5), facecolor=BG)
    ax.set_facecolor(BG); ax.axis('off')
    fig.suptitle('GLDIP+Ψ — Complete Metrics Comparison Table',
                 fontsize=13, fontweight='bold', color=FG, y=1.02)

    hdrs = ['Task','Init (Ψ)','PSNR Before','PSNR After','Δ PSNR',
            'SSIM Before','SSIM After','Δ SSIM',
            'MSE Before','MSE After','MAE Before','MAE After']
    rows = []
    for tn, res in [('Denoising', r_dn), ('Deblurring', r_db)]:
        for iname in INITS:
            r = res[iname]; b, a = r['before'], r['after']
            rows.append([tn, iname,
                f"{b['PSNR']:.2f}", f"{a['PSNR']:.2f}", f"{a['PSNR']-b['PSNR']:+.2f}",
                f"{b['SSIM']:.4f}", f"{a['SSIM']:.4f}", f"{a['SSIM']-b['SSIM']:+.4f}",
                f"{b['MSE']:.5f}", f"{a['MSE']:.5f}",
                f"{b['MAE']:.5f}", f"{a['MAE']:.5f}"])

    tbl = ax.table(cellText=rows, colLabels=hdrs, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 2.6)

    for j in range(len(hdrs)):
        c = tbl[(0,j)]; c.set_facecolor('#2c3e50')
        c.set_text_props(color='white', fontweight='bold')

    ifc = {'Raw':'#fde8e8','Wiener':'#e8fde8','TV':'#e8eeff'}
    for i, row in enumerate(rows):
        iname = row[1]
        for j in range(len(hdrs)):
            cell = tbl[(i+1,j)]
            if j == 1:
                cell.set_facecolor(ifc[iname])
                cell.set_text_props(color=IC[iname], fontweight='bold')
            elif j == 4:
                v = float(row[4])
                cell.set_facecolor('#d5f5e3' if v>0 else '#fadbd8')
                cell.set_text_props(fontweight='bold')
            elif j == 7:
                v = float(row[7])
                cell.set_facecolor('#d5f5e3' if v>0 else '#fadbd8')
                cell.set_text_props(fontweight='bold')
            elif j == 3:
                cell.set_text_props(fontweight='bold')
            elif i%2 == 0 and j not in [0,1,3,4,7]:
                cell.set_facecolor('#f9f9f9')

    plt.tight_layout()
    plt.savefig('outputs/gldip_metrics_table.png',
                dpi=140, bbox_inches='tight', facecolor=BG)
    print("  → gldip_metrics_table.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    SZ = 64
    gt_img=None
    image_path = 'ok.jpg' if os.path.exists('ok.jpg') else 'ok.png'
    if os.path.exists(image_path):
        img_pil = Image.open(image_path).convert('L') 
        img_pil = img_pil.resize((128, 128), Image.Resampling.BILINEAR)
        gt_img = np.array(img_pil, dtype=np.float32) / 255.0

    print("█"*60)
    print("  GLDIP+Ψ Experiment")
    print("█"*60)

    raw = gt_img
    h0, w0 = raw.shape; s = min(h0,w0)
    raw = raw[(h0-s)//2:(h0+s)//2, (w0-s)//2:(w0+s)//2]
    img = resize(raw, (SZ,SZ), anti_aliasing=True)
    img = (img-img.min())/(img.max()-img.min()+1e-8)

    print(f"Image: astronaut {SZ}×{SZ}")

    print("\n" + "="*60 + "\n  DENOISING\n" + "="*60)
    v_dn, r_dn = run_task(img, 'denoise', 0.5)

    print("\n" + "="*60 + "\n  DEBLURRING\n" + "="*60)
    v_db, r_db = run_task(img, 'deblur', 0.5)

    print("\nPlotting...")
    fig_main(img, v_dn, r_dn, v_db, r_db)
    fig_convergence(r_dn, r_db)
    fig_table(r_dn, r_db)
    print("\nAll done.")