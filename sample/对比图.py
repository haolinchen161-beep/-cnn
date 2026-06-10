"""
对比图：预测vs真实，幅值+实部+虚部，与测试.py统一风格.
用法: F:\pytorch_cuda12\python.exe sample\对比图.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SAMPLE_IDX = 2
EPS = 1e-12

base_dir = os.path.dirname(__file__)
out_dir = os.path.join(base_dir, 'output')
npz_path = os.path.join(out_dir, 'final_results.npz')

data = np.load(npz_path, allow_pickle=True)

f_np = np.asarray(data['frequencies'][SAMPLE_IDX], dtype=float)
points = np.asarray(data['points'][SAMPLE_IDX], dtype=float)

t_amp = np.asarray(data['target_frf'][SAMPLE_IDX], dtype=float)
p_amp = np.asarray(data['predicted_frf'][SAMPLE_IDX], dtype=float)

t_re = np.asarray(data['target_re'][SAMPLE_IDX], dtype=float)
p_re = np.asarray(data['predicted_re'][SAMPLE_IDX], dtype=float)
t_im = np.asarray(data['target_im'][SAMPLE_IDX], dtype=float)
p_im = np.asarray(data['predicted_im'][SAMPLE_IDX], dtype=float)

true_freq = np.asarray(data['true_freq_hz'][SAMPLE_IDX], dtype=float).reshape(-1)
pred_freq = np.asarray(data['pred_freq_hz'][SAMPLE_IDX], dtype=float).reshape(-1)
freq_rel = np.asarray(data['freq_rel'][SAMPLE_IDX], dtype=float).reshape(-1)

true_zeta = np.asarray(data['true_zeta'][SAMPLE_IDX], dtype=float).reshape(-1)
pred_zeta = np.asarray(data['pred_zeta'][SAMPLE_IDX], dtype=float).reshape(-1)
zeta_rel = np.asarray(data['zeta_rel'][SAMPLE_IDX], dtype=float).reshape(-1)

true_phi = np.asarray(data['true_phi'][SAMPLE_IDX], dtype=float)
pred_phi = np.asarray(data['pred_phi'][SAMPLE_IDX], dtype=float)
phi_mac = np.asarray(data['phi_mac'][SAMPLE_IDX], dtype=float).reshape(-1)
phi_nrmse = np.asarray(data['phi_nrmse'][SAMPLE_IDX], dtype=float).reshape(-1)
phi_std_ratio = np.asarray(data['phi_std_ratio'][SAMPLE_IDX], dtype=float).reshape(-1)

peak_shift_hz = np.asarray(data['peak_shift_hz'][SAMPLE_IDX], dtype=float).reshape(-1) if 'peak_shift_hz' in data else None
peak_amp_rel = np.asarray(data['peak_amp_rel'][SAMPLE_IDX], dtype=float).reshape(-1) if 'peak_amp_rel' in data else None


print(f"Sample {SAMPLE_IDX}: FRF shape={t_amp.shape}, Freq=[{f_np[0]:.1f}, {f_np[-1]:.1f}] Hz")
print("True modes Hz:", [f"{x:.2f}" for x in true_freq])
print("Pred modes Hz:", [f"{x:.2f}" for x in pred_freq])
print("Freq rel %:", [f"{x*100:.3f}" for x in freq_rel])
print("True zeta:", [f"{x:.6f}" for x in true_zeta])
print("Pred zeta:", [f"{x:.6f}" for x in pred_zeta])
print("Zeta rel %:", [f"{x*100:.2f}" for x in zeta_rel])
if phi_mac is not None:
    print("Phi MAC:", [f"{x:.4f}" for x in phi_mac])
    print("Phi NRMSE:", [f"{x:.4f}" for x in phi_nrmse])
    print("Phi std ratio:", [f"{x:.4f}" for x in phi_std_ratio])


def build_stretched_axis(freqs, mode_freqs):
    pk = np.sort(np.asarray(mode_freqs, dtype=float))
    bw = [0.012 * p for p in pk]
    weights = [
        (15.0 if i == 0 else 4.0) * np.exp(-0.5 * ((freqs - pk[i]) / (bw[i] * 0.5)) ** 2)
        for i in range(len(pk))
    ]
    total_w = 1.0 + sum(weights)
    xs = np.zeros_like(freqs)
    for i in range(1, len(freqs)):
        xs[i] = xs[i - 1] + (total_w[i] + total_w[i - 1]) / 2 * (freqs[i] - freqs[i - 1])
    tick_freqs = [freqs[0]]
    for p, b in zip(pk, bw):
        tick_freqs.extend([p - b, p, p + b])
    tick_freqs.append(freqs[-1])
    tick_freqs = np.unique(np.sort(tick_freqs))
    tick_freqs = tick_freqs[(tick_freqs >= freqs[0]) & (tick_freqs <= freqs[-1])]
    tick_locs = np.interp(tick_freqs, freqs, xs)
    tick_labels = [f"{f:.1f}" for f in tick_freqs]
    return xs, tick_locs, tick_labels


xs, tls, tlbs = build_stretched_axis(f_np, true_freq)


def mark_modes(ax, label_once=True):
    true_labeled = False; pred_labeled = False
    for mp in true_freq:
        lab = 'True mode' if label_once and not true_labeled else None
        ax.axvline(np.interp(mp, f_np, xs), color='green', linestyle='-', linewidth=0.9, alpha=0.75, label=lab)
        true_labeled = True
    for mp in pred_freq:
        lab = 'Pred mode' if label_once and not pred_labeled else None
        ax.axvline(np.interp(mp, f_np, xs), color='red', linestyle='--', linewidth=0.9, alpha=0.75, label=lab)
        pred_labeled = True


def shade_true_mode_windows(ax):
    for mp in true_freq:
        b = 0.012 * mp
        ax.axvspan(np.interp(mp - b, f_np, xs), np.interp(mp + b, f_np, xs), color='gray', alpha=0.08)


def safe_ylim(ax, target, pred, allow_negative=False):
    mx = max(float(np.max(target)), float(np.max(pred)), 1e-12)
    mn = min(float(np.min(target)), float(np.min(pred)))
    if not allow_negative: mn = min(0.0, mn)
    if abs(mx - mn) < 1e-12: mx = mn + 1.0
    ax.set_ylim(mn * 1.15, mx * 1.15)


def plot_row_db_lin(ax_db, ax_lin, target, pred, node_idx, tag):
    amp_t = target[node_idx]; amp_p = pred[node_idx]
    # dB
    amp_t_db = 20 * np.log10(amp_t + 1e-12); amp_p_db = 20 * np.log10(amp_p + 1e-12)
    ax_db.plot(xs, amp_t_db, '-', linewidth=1.2, label='Target', alpha=0.9)
    ax_db.plot(xs, amp_p_db, '--', linewidth=1.2, label='Predicted', alpha=0.9)
    shade_true_mode_windows(ax_db); mark_modes(ax_db, label_once=True)
    db_max = max(amp_t_db.max(), amp_p_db.max())
    ax_db.set_ylim(db_max - 60, db_max + 5)
    ax_db.set_ylabel(f"{tag}\n({points[node_idx,0]:.3f},{points[node_idx,1]:.3f})\nMag(dB)", fontsize=8)
    ax_db.grid(alpha=0.2)
    # linear
    ax_lin.plot(xs, amp_t, '-', linewidth=1.2, label='Target', alpha=0.9)
    ax_lin.plot(xs, amp_p, '--', linewidth=1.2, label='Predicted', alpha=0.9)
    shade_true_mode_windows(ax_lin); mark_modes(ax_lin, label_once=False)
    ax_lin.set_ylabel(f"{tag}\nMag(lin)", fontsize=8)
    p95 = np.percentile(amp_t, 95); p99 = np.percentile(amp_t, 99.9)
    ax_lin.set_ylim(0, min(p95 * 3, p99 * 0.8))
    ax_lin.grid(alpha=0.2)


def select_nodes():
    x_sorted_idx = np.argsort(points[:, 0])
    n_pts = points.shape[0]
    base = [x_sorted_idx[0], x_sorted_idx[n_pts//4], x_sorted_idx[n_pts//2],
            x_sorted_idx[3*n_pts//4], x_sorted_idx[-1]]
    target_energy = np.mean(t_amp, axis=1)
    err_energy = np.mean(np.abs(p_amp - t_amp), axis=1)
    all_idx, tags = [], []
    for idx, tag in zip(base, ['x_min','x_q1','x_med','x_q3','x_max']):
        if idx not in all_idx: all_idx.append(idx); tags.append(tag)
    peak_idx = int(np.argmax(target_energy))
    err_idx = int(np.argmax(err_energy))
    if peak_idx not in all_idx: all_idx.append(peak_idx); tags.append('max_amp')
    if err_idx not in all_idx: all_idx.append(err_idx); tags.append('max_err')
    return all_idx, tags

selected, selected_tags = select_nodes()


# ============================================================
# 图 0：模态参数 summary
# ============================================================
fig0, axes0 = plt.subplots(3, 2, figsize=(14, 12))
modes = np.arange(1, len(true_freq) + 1); width = 0.35

ax = axes0[0, 0]
ax.bar(modes - width/2, true_freq, width, label='True'); ax.bar(modes + width/2, pred_freq, width, label='Pred')
ax.set_title('Natural Frequency'); ax.legend(); ax.grid(alpha=0.2)

ax = axes0[0, 1]; ax.bar(modes, freq_rel * 100.0)
ax.set_title('Frequency Relative Error (%)'); ax.grid(alpha=0.2)

ax = axes0[1, 0]
ax.bar(modes - width/2, true_zeta, width, label='True'); ax.bar(modes + width/2, pred_zeta, width, label='Pred')
ax.set_title('Damping Ratio'); ax.legend(); ax.grid(alpha=0.2)

ax = axes0[1, 1]; ax.bar(modes, zeta_rel * 100.0)
ax.set_title('Zeta Relative Error (%)'); ax.grid(alpha=0.2)

ax = axes0[2, 0]
if phi_mac is not None: ax.bar(modes, phi_mac); ax.set_ylim(0, 1.05)
ax.set_title('Mode Shape MAC'); ax.grid(alpha=0.2)

ax = axes0[2, 1]
if phi_nrmse is not None: ax.plot(modes, phi_nrmse, 'o-', label='NRMSE')
if phi_std_ratio is not None: ax.plot(modes, phi_std_ratio, 's--', label='std ratio')
ax.set_title('Phi Error / Scale'); ax.legend(); ax.grid(alpha=0.2)

fig0.suptitle(f"Modal Summary — Sample {SAMPLE_IDX}", fontsize=14)
fig0.tight_layout()
plt.savefig(os.path.join(out_dir, 'modal_summary_sample.png'), dpi=160, bbox_inches='tight'); plt.close()
print(f"Modal summary: {os.path.join(out_dir, 'modal_summary_sample.png')}")


# ============================================================
# 图 1：FRF 幅值 (dB + linear)，5选点
# ============================================================
n_sel = len(selected)
fig1 = plt.figure(figsize=(20, 2.8 * n_sel))
for i, (si, tag) in enumerate(zip(selected, selected_tags)):
    ax_db = fig1.add_subplot(n_sel, 2, 2*i + 1)
    ax_lin = fig1.add_subplot(n_sel, 2, 2*i + 2)
    plot_row_db_lin(ax_db, ax_lin, t_amp, p_amp, si, tag)
    if i == 0: ax_db.legend(fontsize=7)
for ax in fig1.get_axes()[-2:]:
    ax.set_xticks(tls); ax.set_xticklabels(tlbs, fontsize=8); ax.set_xlabel('Frequency (Hz)')
fig1.suptitle(f"FRF Amplitude — Sample {SAMPLE_IDX} | "
              f"True f={','.join(f'{x:.1f}' for x in true_freq)} Hz | "
              f"Pred f={','.join(f'{x:.1f}' for x in pred_freq)} Hz", fontsize=13)
fig1.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图.png'), dpi=160, bbox_inches='tight'); plt.close()
print(f"Amp: {os.path.join(out_dir, '对比图.png')}")


# ============================================================
# 图 2：Re / Im
# ============================================================
fig2, axes2 = plt.subplots(len(selected), 2, figsize=(18, 2.8 * len(selected)), sharex=True)
if len(selected) == 1: axes2 = np.expand_dims(axes2, axis=0)
axes2[0, 0].set_title('Real Part', fontsize=12, fontweight='bold')
axes2[0, 1].set_title('Imaginary Part', fontsize=12, fontweight='bold')
for row, (si, tag) in enumerate(zip(selected, selected_tags)):
    ax_r = axes2[row, 0]; ax_i = axes2[row, 1]
    ax_r.plot(xs, t_re[si], '-', lw=1.2, label='Target', alpha=0.9)
    ax_r.plot(xs, p_re[si], '--', lw=1.2, label='Predicted', alpha=0.9)
    shade_true_mode_windows(ax_r); mark_modes(ax_r, label_once=False)
    safe_ylim(ax_r, t_re[si], p_re[si], allow_negative=True)
    ax_r.set_ylabel(f"{tag}\n(x={points[si,0]:.3f},y={points[si,1]:.3f})", fontsize=8); ax_r.grid(alpha=0.2)
    ax_i.plot(xs, t_im[si], '-', lw=1.2, label='Target', alpha=0.9)
    ax_i.plot(xs, p_im[si], '--', lw=1.2, label='Predicted', alpha=0.9)
    shade_true_mode_windows(ax_i); mark_modes(ax_i, label_once=False)
    safe_ylim(ax_i, t_im[si], p_im[si], allow_negative=True)
    ax_i.set_ylabel(f"{tag}", fontsize=8); ax_i.grid(alpha=0.2)
    if row == 0: ax_r.legend(fontsize=7); ax_i.legend(fontsize=7)
for ax in axes2[-1, :]:
    ax.set_xticks(tls); ax.set_xticklabels(tlbs, fontsize=8); ax.set_xlabel('Frequency (Hz)')
fig2.suptitle(f"FRF Real / Imag — Sample {SAMPLE_IDX}", fontsize=12)
fig2.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_reim.png'), dpi=160, bbox_inches='tight'); plt.close()
print(f"Re/Im: {os.path.join(out_dir, '对比图_reim.png')}")


# ============================================================
# 图 3：单点 FRF (误差最大)
# ============================================================
err_energy = np.mean(np.abs(p_amp - t_amp), axis=1)
idx_single = int(np.argmax(err_energy))
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(xs, t_amp[idx_single], 'b-', linewidth=1.2, label='Target', alpha=0.9)
ax3.plot(xs, p_amp[idx_single], 'r--', linewidth=1.2, label='Predicted', alpha=0.9)
shade_true_mode_windows(ax3); mark_modes(ax3, label_once=True)
safe_ylim(ax3, t_amp[idx_single], p_amp[idx_single])
ax3.set_xticks(tls); ax3.set_xticklabels(tlbs, fontsize=8)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel(f"Amplitude\n(x={points[idx_single,0]:.4f}, y={points[idx_single,1]:.4f})")
ax3.set_title(f"Single Point FRF — Max Error Node {idx_single} | "
              f"freq_rel%={','.join(f'{x*100:.2f}' for x in freq_rel)}")
ax3.legend(); ax3.grid(alpha=0.2)
fig3.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_single.png'), dpi=160, bbox_inches='tight'); plt.close()
print(f"Single: {os.path.join(out_dir, '对比图_single.png')}")


# ============================================================
# 图 4：复数 FRF
# ============================================================
fig4, (ar, ai, aa) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
ar.plot(xs, t_re[idx_single], 'b-', lw=1.2, label='Target')
ar.plot(xs, p_re[idx_single], 'r--', lw=1.2, label='Predicted')
mark_modes(ar, label_once=True)
ar.axhline(0, color='gray', linestyle='--', alpha=0.5)
ar.set_ylabel('Real'); ar.set_title(f"Real @ max error node {idx_single}")
ar.legend(); ar.grid(alpha=0.2)
ai.plot(xs, t_im[idx_single], 'b-', lw=1.2, label='Target')
ai.plot(xs, p_im[idx_single], 'r--', lw=1.2, label='Predicted')
mark_modes(ai, label_once=False)
ai.axhline(0, color='gray', linestyle='--', alpha=0.5)
ai.set_ylabel('Imaginary'); ai.set_title(f"Imaginary @ max error node {idx_single}")
ai.legend(); ai.grid(alpha=0.2)
aa.plot(xs, t_amp[idx_single], 'b-', lw=1.2, label='Target')
aa.plot(xs, p_amp[idx_single], 'r--', lw=1.2, label='Predicted')
shade_true_mode_windows(aa); mark_modes(aa, label_once=False)
safe_ylim(aa, t_amp[idx_single], p_amp[idx_single])
aa.set_ylabel('Amplitude'); aa.set_title('Amplitude')
aa.legend(); aa.grid(alpha=0.2)
aa.set_xticks(tls); aa.set_xticklabels(tlbs, fontsize=8)
aa.set_xlabel('Frequency (Hz)')
fig4.suptitle(f"Complex FRF — Sample {SAMPLE_IDX}", fontsize=12)
fig4.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_complex.png'), dpi=160, bbox_inches='tight'); plt.close()
print(f"Complex: {os.path.join(out_dir, '对比图_complex.png')}")


# ============================================================
# 图 5：振型空间分布
# ============================================================
if true_phi is not None and pred_phi is not None:
    k_modes = true_phi.shape[1]
    fig5, axes5 = plt.subplots(k_modes, 3, figsize=(18, 4.5 * k_modes))
    if k_modes == 1: axes5 = np.expand_dims(axes5, axis=0)
    for k in range(k_modes):
        tphi = true_phi[:, k]; pphi = pred_phi[:, k]; ephi = pphi - tphi
        vmax = max(abs(tphi).max(), abs(pphi).max(), 1e-12); evmax = max(abs(ephi).max(), 1e-12)
        sc0 = axes5[k, 0].scatter(points[:, 0], points[:, 1], c=tphi, s=8, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes5[k, 0].set_title(f"True phi{k+1}"); plt.colorbar(sc0, ax=axes5[k, 0])
        sc1 = axes5[k, 1].scatter(points[:, 0], points[:, 1], c=pphi, s=8, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes5[k, 1].set_title(f"Pred phi{k+1}"); plt.colorbar(sc1, ax=axes5[k, 1])
        sc2 = axes5[k, 2].scatter(points[:, 0], points[:, 1], c=ephi, s=8, cmap='coolwarm', vmin=-evmax, vmax=evmax)
        title = f"Error phi{k+1}"
        if phi_mac is not None: title += f" | MAC={phi_mac[k]:.4f}"
        if phi_nrmse is not None: title += f" | NRMSE={phi_nrmse[k]:.3f}"
        axes5[k, 2].set_title(title); plt.colorbar(sc2, ax=axes5[k, 2])
        for j in range(3):
            axes5[k, j].set_xlabel('X (m)'); axes5[k, j].set_ylabel('Y (m)')
            axes5[k, j].set_aspect('equal', adjustable='box'); axes5[k, j].grid(alpha=0.15)
    fig5.suptitle(f"Mode Shape Spatial Comparison — Sample {SAMPLE_IDX}", fontsize=14)
    fig5.tight_layout()
    plt.savefig(os.path.join(out_dir, '振型对比图.png'), dpi=160, bbox_inches='tight'); plt.close()
    print(f"Phi: {os.path.join(out_dir, '振型对比图.png')}")
else:
    print("No pred_phi / true_phi found, skip.")
