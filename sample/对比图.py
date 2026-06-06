"""
对比图.py — 模态参数 + FRF 对比图。

依赖:
    sample/evaluate.py 生成的 sample/output/final_results.npz

用法:
    F:/pytorch_cuda12/python.exe sample/对比图.py
"""
import os
import sys
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

true_omega = np.asarray(data['true_omega'][SAMPLE_IDX], dtype=float).reshape(-1)
pred_omega = np.asarray(data['pred_omega'][SAMPLE_IDX], dtype=float).reshape(-1)

true_freq = np.asarray(data['true_freq_hz'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'true_freq_hz' in data else true_omega / (2 * np.pi)

pred_freq = np.asarray(data['pred_freq_hz'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'pred_freq_hz' in data else pred_omega / (2 * np.pi)

freq_rel = np.asarray(data['freq_rel'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'freq_rel' in data else np.abs(pred_freq - true_freq) / (true_freq + EPS)

true_zeta = np.asarray(data['true_zeta'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'true_zeta' in data else None

pred_zeta = np.asarray(data['pred_zeta'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'pred_zeta' in data else None

zeta_rel = np.asarray(data['zeta_rel'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'zeta_rel' in data else None

true_phi = np.asarray(data['true_phi'][SAMPLE_IDX], dtype=float) \
    if 'true_phi' in data else None

pred_phi = np.asarray(data['pred_phi'][SAMPLE_IDX], dtype=float) \
    if 'pred_phi' in data else None

phi_mac = np.asarray(data['phi_mac'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'phi_mac' in data else None

phi_nrmse = np.asarray(data['phi_nrmse'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'phi_nrmse' in data else None

phi_std_ratio = np.asarray(data['phi_std_ratio'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'phi_std_ratio' in data else None

peak_shift_hz = np.asarray(data['peak_shift_hz'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'peak_shift_hz' in data else None

peak_amp_rel = np.asarray(data['peak_amp_rel'][SAMPLE_IDX], dtype=float).reshape(-1) \
    if 'peak_amp_rel' in data else None


print(f"Sample {SAMPLE_IDX}: FRF shape={t_amp.shape}, Freq=[{f_np[0]:.1f}, {f_np[-1]:.1f}] Hz")
print("True modes Hz:", [f"{x:.2f}" for x in true_freq])
print("Pred modes Hz:", [f"{x:.2f}" for x in pred_freq])
print("Freq rel %:", [f"{x*100:.3f}" for x in freq_rel])

if true_zeta is not None:
    print("True zeta:", [f"{x:.6f}" for x in true_zeta])
    print("Pred zeta:", [f"{x:.6f}" for x in pred_zeta])
    print("Zeta rel %:", [f"{x*100:.2f}" for x in zeta_rel])

if phi_mac is not None:
    print("Phi MAC:", [f"{x:.4f}" for x in phi_mac])
    print("Phi NRMSE:", [f"{x:.4f}" for x in phi_nrmse])
    print("Phi std ratio:", [f"{x:.4f}" for x in phi_std_ratio])


def build_stretched_axis(freqs, mode_freqs):
    """
    与原脚本类似：在模态峰附近拉伸频率轴。
    使用真实频率作为主拉伸中心。
    """
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
    """
    绿色实线：真实模态频率
    红色虚线：预测模态频率
    """
    true_labeled = False
    pred_labeled = False

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
        ax.axvspan(
            np.interp(mp - b, f_np, xs),
            np.interp(mp + b, f_np, xs),
            color='gray',
            alpha=0.08,
        )


def safe_ylim(ax, target, pred, allow_negative=False):
    mx = max(float(np.max(target)), float(np.max(pred)), 1e-12)
    mn = min(float(np.min(target)), float(np.min(pred)))

    if not allow_negative:
        mn = min(0.0, mn)

    if abs(mx - mn) < 1e-12:
        mx = mn + 1.0

    ax.set_ylim(mn * 1.15, mx * 1.15)


def plot_row(ax, target_arr, pred_arr, node_idx, tag, ylabel, do_ylim=False, allow_negative=False):
    ax.plot(xs, target_arr[node_idx], '-', linewidth=1.2, label='Target', alpha=0.9)
    ax.plot(xs, pred_arr[node_idx], '--', linewidth=1.2, label='Predicted', alpha=0.9)

    shade_true_mode_windows(ax)
    mark_modes(ax, label_once=False)

    if do_ylim:
        safe_ylim(ax, target_arr[node_idx], pred_arr[node_idx], allow_negative=allow_negative)

    ax.set_ylabel(
        f"{tag}\n(x={points[node_idx,0]:.4f}, y={points[node_idx,1]:.4f})",
        fontsize=8,
    )
    ax.grid(alpha=0.2)


def select_nodes():
    """
    选择代表节点：
    1. x_min
    2. x_q1
    3. x_med
    4. x_q3
    5. x_max
    6. target FRF 平均幅值最大节点
    7. FRF 误差最大节点
    """
    x_sorted_idx = np.argsort(points[:, 0])
    n_pts = points.shape[0]

    base = [
        x_sorted_idx[0],
        x_sorted_idx[n_pts // 4],
        x_sorted_idx[n_pts // 2],
        x_sorted_idx[3 * n_pts // 4],
        x_sorted_idx[-1],
    ]

    target_energy = np.mean(t_amp, axis=1)
    err_energy = np.mean(np.abs(p_amp - t_amp), axis=1)

    idx_peak = int(np.argmax(target_energy))
    idx_err = int(np.argmax(err_energy))

    all_idx = []
    tags = []

    for idx, tag in zip(base, ['x_min', 'x_q1', 'x_med', 'x_q3', 'x_max']):
        if idx not in all_idx:
            all_idx.append(idx)
            tags.append(tag)

    if idx_peak not in all_idx:
        all_idx.append(idx_peak)
        tags.append('max_target_amp')

    if idx_err not in all_idx:
        all_idx.append(idx_err)
        tags.append('max_error')

    return all_idx, tags


node_indices, node_tags = select_nodes()


# ============================================================
# 图 0：模态参数 summary
# ============================================================
fig0, axes0 = plt.subplots(3, 2, figsize=(14, 12))
modes = np.arange(1, len(true_freq) + 1)
width = 0.35

# f true/pred
ax = axes0[0, 0]
ax.bar(modes - width/2, true_freq, width, label='True')
ax.bar(modes + width/2, pred_freq, width, label='Pred')
ax.set_title('Natural Frequency')
ax.set_xlabel('Mode')
ax.set_ylabel('Frequency (Hz)')
ax.grid(alpha=0.2)
ax.legend()

# frequency relative error
ax = axes0[0, 1]
ax.bar(modes, freq_rel * 100.0)
ax.set_title('Frequency Relative Error')
ax.set_xlabel('Mode')
ax.set_ylabel('Error (%)')
ax.grid(alpha=0.2)

# zeta true/pred
ax = axes0[1, 0]
if true_zeta is not None:
    ax.bar(modes - width/2, true_zeta, width, label='True')
    ax.bar(modes + width/2, pred_zeta, width, label='Pred')
    ax.legend()
ax.set_title('Damping Ratio ζ')
ax.set_xlabel('Mode')
ax.set_ylabel('ζ')
ax.grid(alpha=0.2)

# zeta relative error
ax = axes0[1, 1]
if zeta_rel is not None:
    ax.bar(modes, zeta_rel * 100.0)
ax.set_title('Zeta Relative Error')
ax.set_xlabel('Mode')
ax.set_ylabel('Error (%)')
ax.grid(alpha=0.2)

# MAC
ax = axes0[2, 0]
if phi_mac is not None:
    ax.bar(modes, phi_mac)
    ax.set_ylim(0, 1.05)
ax.set_title('Mode Shape MAC')
ax.set_xlabel('Mode')
ax.set_ylabel('MAC')
ax.grid(alpha=0.2)

# NRMSE / std ratio
ax = axes0[2, 1]
if phi_nrmse is not None:
    ax.plot(modes, phi_nrmse, 'o-', label='NRMSE')
if phi_std_ratio is not None:
    ax.plot(modes, phi_std_ratio, 's--', label='std(pred)/std(true)')
ax.set_title('Mode Shape Error / Scale')
ax.set_xlabel('Mode')
ax.set_ylabel('Value')
ax.grid(alpha=0.2)
ax.legend()

extra_text = []
if peak_shift_hz is not None:
    extra_text.append("Peak shift Hz: " + ", ".join(f"{x:.2f}" for x in peak_shift_hz))
if peak_amp_rel is not None:
    extra_text.append("Peak amp rel %: " + ", ".join(f"{x*100:.1f}" for x in peak_amp_rel))

fig0.suptitle(
    f"Modal Summary — Sample {SAMPLE_IDX}\n" + " | ".join(extra_text),
    fontsize=14,
)
fig0.tight_layout()
plt.savefig(os.path.join(out_dir, 'modal_summary_sample.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"Modal summary: {os.path.join(out_dir, 'modal_summary_sample.png')}")


# ============================================================
# 图 1：FRF 幅值，多节点
# ============================================================
fig1, axes1 = plt.subplots(len(node_indices), 1, figsize=(14, 2.5 * len(node_indices)), sharex=True)
if len(node_indices) == 1:
    axes1 = [axes1]

for ax, node_idx, tag in zip(axes1, node_indices, node_tags):
    plot_row(ax, t_amp, p_amp, node_idx, tag, 'Amplitude', do_ylim=True, allow_negative=False)
    ax.legend(fontsize=7)

axes1[-1].set_xticks(tls)
axes1[-1].set_xticklabels(tlbs, fontsize=8)
axes1[-1].set_xlabel('Frequency (Hz)')
fig1.suptitle(
    f"FRF Amplitude — Sample {SAMPLE_IDX} | "
    f"True f={','.join(f'{x:.1f}' for x in true_freq)} Hz | "
    f"Pred f={','.join(f'{x:.1f}' for x in pred_freq)} Hz",
    fontsize=12,
)
fig1.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"Amp: {os.path.join(out_dir, '对比图.png')}")


# ============================================================
# 图 2：Re / Im，多节点
# ============================================================
fig2, axes2 = plt.subplots(len(node_indices), 2, figsize=(18, 2.8 * len(node_indices)), sharex=True)
if len(node_indices) == 1:
    axes2 = np.expand_dims(axes2, axis=0)

for row, (node_idx, tag) in enumerate(zip(node_indices, node_tags)):
    plot_row(axes2[row, 0], t_re, p_re, node_idx, tag, 'Real', do_ylim=True, allow_negative=True)
    plot_row(axes2[row, 1], t_im, p_im, node_idx, tag, 'Imag', do_ylim=True, allow_negative=True)
    axes2[row, 0].legend(fontsize=7)
    axes2[row, 1].legend(fontsize=7)

for ax in axes2[-1, :]:
    ax.set_xticks(tls)
    ax.set_xticklabels(tlbs, fontsize=8)
    ax.set_xlabel('Frequency (Hz)')

fig2.suptitle(f"FRF Real / Imag — Sample {SAMPLE_IDX}", fontsize=12)
fig2.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_reim.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"Re/Im: {os.path.join(out_dir, '对比图_reim.png')}")


# ============================================================
# 图 3：单点 FRF，选误差最大节点
# ============================================================
err_energy = np.mean(np.abs(p_amp - t_amp), axis=1)
idx_single = int(np.argmax(err_energy))

fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(xs, t_amp[idx_single], 'b-', linewidth=1.2, label='Target', alpha=0.9)
ax3.plot(xs, p_amp[idx_single], 'r--', linewidth=1.2, label='Predicted', alpha=0.9)
shade_true_mode_windows(ax3)
mark_modes(ax3, label_once=True)
safe_ylim(ax3, t_amp[idx_single], p_amp[idx_single], allow_negative=False)

ax3.set_xticks(tls)
ax3.set_xticklabels(tlbs, fontsize=8)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel(f"Amplitude\n(x={points[idx_single,0]:.4f}, y={points[idx_single,1]:.4f})")
ax3.set_title(
    f"Single Point FRF — Max Error Node {idx_single} | "
    f"freq_rel%={','.join(f'{x*100:.2f}' for x in freq_rel)}"
)
ax3.legend()
ax3.grid(alpha=0.2)
fig3.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_single.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"Single: {os.path.join(out_dir, '对比图_single.png')}")


# ============================================================
# 图 4：复数 FRF，误差最大节点
# ============================================================
fig4, (ar, ai, aa) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

ar.plot(xs, t_re[idx_single], 'b-', linewidth=1.2, label='Target')
ar.plot(xs, p_re[idx_single], 'r--', linewidth=1.2, label='Predicted')
mark_modes(ar, label_once=True)
ar.axhline(0, color='gray', linestyle='--', alpha=0.5)
ar.set_ylabel('Real')
ar.set_title(f"Real @ max error node {idx_single}")
ar.legend()
ar.grid(alpha=0.2)

ai.plot(xs, t_im[idx_single], 'b-', linewidth=1.2, label='Target')
ai.plot(xs, p_im[idx_single], 'r--', linewidth=1.2, label='Predicted')
mark_modes(ai, label_once=False)
ai.axhline(0, color='gray', linestyle='--', alpha=0.5)
ai.set_ylabel('Imaginary')
ai.set_title(f"Imaginary @ max error node {idx_single}")
ai.legend()
ai.grid(alpha=0.2)

aa.plot(xs, t_amp[idx_single], 'b-', linewidth=1.2, label='Target')
aa.plot(xs, p_amp[idx_single], 'r--', linewidth=1.2, label='Predicted')
shade_true_mode_windows(aa)
mark_modes(aa, label_once=False)
safe_ylim(aa, t_amp[idx_single], p_amp[idx_single], allow_negative=False)
aa.set_ylabel('Amplitude')
aa.set_title('Amplitude = sqrt(Re^2 + Im^2)')
aa.legend()
aa.grid(alpha=0.2)

aa.set_xticks(tls)
aa.set_xticklabels(tlbs, fontsize=8)
aa.set_xlabel('Frequency (Hz)')

fig4.suptitle(f"Complex FRF — Sample {SAMPLE_IDX}", fontsize=12)
fig4.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_complex.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"Complex: {os.path.join(out_dir, '对比图_complex.png')}")


# ============================================================
# 图 5：振型空间分布 true / pred / error
# ============================================================
if true_phi is not None and pred_phi is not None:
    k_modes = true_phi.shape[1]
    fig5, axes5 = plt.subplots(k_modes, 3, figsize=(18, 4.5 * k_modes))

    if k_modes == 1:
        axes5 = np.expand_dims(axes5, axis=0)

    for k in range(k_modes):
        tphi = true_phi[:, k]
        pphi = pred_phi[:, k]
        ephi = pphi - tphi

        vmax = max(abs(tphi).max(), abs(pphi).max(), 1e-12)
        evmax = max(abs(ephi).max(), 1e-12)

        sc0 = axes5[k, 0].scatter(points[:, 0], points[:, 1], c=tphi, s=8, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes5[k, 0].set_title(f"True φ{k+1}")
        plt.colorbar(sc0, ax=axes5[k, 0])

        sc1 = axes5[k, 1].scatter(points[:, 0], points[:, 1], c=pphi, s=8, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes5[k, 1].set_title(f"Pred φ{k+1}")
        plt.colorbar(sc1, ax=axes5[k, 1])

        sc2 = axes5[k, 2].scatter(points[:, 0], points[:, 1], c=ephi, s=8, cmap='coolwarm', vmin=-evmax, vmax=evmax)
        title = f"Error φ{k+1}"
        if phi_mac is not None:
            title += f" | MAC={phi_mac[k]:.4f}"
        if phi_nrmse is not None:
            title += f" | NRMSE={phi_nrmse[k]:.3f}"
        axes5[k, 2].set_title(title)
        plt.colorbar(sc2, ax=axes5[k, 2])

        for j in range(3):
            axes5[k, j].set_xlabel('X (m)')
            axes5[k, j].set_ylabel('Y (m)')
            axes5[k, j].set_aspect('equal', adjustable='box')
            axes5[k, j].grid(alpha=0.15)

    fig5.suptitle(f"Mode Shape Spatial Comparison — Sample {SAMPLE_IDX}", fontsize=14)
    fig5.tight_layout()
    plt.savefig(os.path.join(out_dir, '振型对比图.png'), dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Phi: {os.path.join(out_dir, '振型对比图.png')}")
else:
    print("No pred_phi / true_phi found in final_results.npz, skip phi comparison.")
