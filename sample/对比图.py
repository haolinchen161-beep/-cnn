"""
对比图：预测vs真实，幅值+实部+虚部，与测试.py统一风格.
用法: F:\pytorch_cuda12\python.exe sample\对比图.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SAMPLE_IDX = 1

data = np.load(os.path.join(os.path.dirname(__file__), 'output', 'final_results.npz'), allow_pickle=True)
f_np = data['frequencies'][SAMPLE_IDX]
points = data['points'][SAMPLE_IDX]
t_amp = data['target_frf'][SAMPLE_IDX]
p_amp = data['predicted_frf'][SAMPLE_IDX]
t_re = data['target_re'][SAMPLE_IDX]; p_re = data['predicted_re'][SAMPLE_IDX]
t_im = data['target_im'][SAMPLE_IDX]; p_im = data['predicted_im'][SAMPLE_IDX]

print(f'Sample {SAMPLE_IDX}: {t_amp.shape}, Freq: [{f_np[0]:.1f}, {f_np[-1]:.1f}] Hz')

# 直接从模态数据取三阶固有频率
true_omega = data['true_omega'][SAMPLE_IDX]
pred_omega = data['pred_omega'][SAMPLE_IDX]
pk_true = true_omega / (2*np.pi)
pk_pred = pred_omega / (2*np.pi)
pk_flat = pk_true.flatten() if hasattr(pk_true,'flatten') else pk_true
pk = sorted(pk_flat.tolist())
pk_strs = [f'{p:.1f}' for p in pk]
pred_flat = pk_pred.flatten() if hasattr(pk_pred,'flatten') else pk_pred
pred_strs = [f'{p:.1f}' for p in sorted(pred_flat)]
print(f'True modes: {pk_strs} Hz')
print(f'Pred modes: {pred_strs} Hz')

# stretch x-axis — 与测试.py完全一致
bw = [0.012 * p for p in pk]
ws = [(15.0 if i==0 else 4.0) * np.exp(-0.5 * ((f_np - pk[i]) / (bw[i] * 0.5))**2) for i in range(len(pk))]
tw = 1.0 + sum(ws)
xs = np.zeros_like(f_np)
for i in range(1, len(f_np)):
    xs[i] = xs[i-1] + (tw[i] + tw[i-1]) / 2 * (f_np[i] - f_np[i-1])
tfs = [f_np[0]]
for p, b in zip(pk, bw): tfs.extend([p-b, p, p+b])
tfs.append(f_np[-1])
tfs = np.unique(np.sort(tfs))
tfs = tfs[(tfs >= f_np[0]) & (tfs <= f_np[-1])]
tls = np.interp(tfs, f_np, xs)
tlbs = [f'{f:.1f}' for f in tfs]

# 选5个代表性点 (按x坐标分布) — 与测试.py完全一致
x_sorted_idx = np.argsort(points[:, 0])
n_pts = points.shape[0]
all_i = [x_sorted_idx[0], x_sorted_idx[n_pts//4], x_sorted_idx[n_pts//2],
         x_sorted_idx[3*n_pts//4], x_sorted_idx[-1]]
tags = ['x_min', 'x_q1', 'x_med', 'x_q3', 'x_max']
colors = plt.cm.viridis(np.linspace(0, 1, 5))

out_dir = os.path.join(os.path.dirname(__file__), 'output')

def plot_row(ax, t_arr, p_arr, i, tag, color, ylabel, do_ylim=False):
    ax.plot(xs, t_arr[i], '-', color=color, linewidth=1.2, label='Target', alpha=0.9)
    ax.plot(xs, p_arr[i], '--', color=color, linewidth=1.2, label='Predicted', alpha=0.9)
    for mp, b in zip(pk, bw):
        ax.axvspan(np.interp(mp-b, f_np, xs), np.interp(mp+b, f_np, xs), color='gray', alpha=0.08)
    if do_ylim:
        tmx = max(1e-6, t_arr[i].max())
        ax.set_ylim(max(-50, t_arr[i].min()*1.15), tmx*1.15)
    ax.set_ylabel(f'{tag}\n(x={points[i,0]:.4f},y={points[i,1]:.4f})', fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

# 图1: 幅值 — 与测试.py相同的5行布局
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.subplots_adjust(hspace=0.15)
for ax, i, tag, c in zip(axes, all_i, tags, colors):
    plot_row(ax, t_amp, p_amp, i, tag, c, 'Amplitude', True)
axes[-1].set_xticks(tls)
axes[-1].set_xticklabels(tlbs, fontsize=8)
axes[-1].set_xlabel('Frequency (Hz)', fontsize=10)
fig.suptitle(f'FRF Amplitude — Sample {SAMPLE_IDX}', fontsize=14, y=0.92)
plt.savefig(os.path.join(out_dir, '对比图.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Amp: {out_dir}/对比图.png')

# 图2: 实部+虚部
fig2, axes2 = plt.subplots(5, 2, figsize=(18, 14), sharex=True)
fig2.subplots_adjust(hspace=0.15)
for row, (i, tag, c) in enumerate(zip(all_i, tags, colors)):
    plot_row(axes2[row, 0], t_re, p_re, i, tag, c, 'Real')
    plot_row(axes2[row, 1], t_im, p_im, i, tag, c, 'Imag')
for a in axes2[-1, :]:
    a.set_xticks(tls); a.set_xticklabels(tlbs, fontsize=8); a.set_xlabel('Hz', fontsize=10)
fig2.suptitle(f'FRF Re/Im — Sample {SAMPLE_IDX}', fontsize=14, y=0.92)
plt.savefig(os.path.join(out_dir, '对比图_reim.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Re/Im: {out_dir}/对比图_reim.png')

# 图3: 单点幅值 (选 x_max 处节点, 与测试.py图1同布局)
idx1 = x_sorted_idx[-1]
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(xs, t_amp[idx1], '-', color='blue', linewidth=1.2, label='Target', alpha=0.9)
ax3.plot(xs, p_amp[idx1], '--', color='red', linewidth=1.2, label='Predicted', alpha=0.9)
for mp, b in zip(pk, bw):
    ax3.axvspan(np.interp(mp-b, f_np, xs), np.interp(mp+b, f_np, xs), color='gray', alpha=0.08)
for mp in pk:
    ax3.axvline(np.interp(mp, f_np, xs), color='green', linestyle=':', linewidth=0.8, alpha=0.6)
ax3.set_xticks(tls); ax3.set_xticklabels(tlbs, fontsize=8)
ax3.set_xlabel('Frequency (Hz)', fontsize=10)
ax3.set_ylabel(f'Amplitude\n(x={points[idx1,0]:.4f},y={points[idx1,1]:.4f})', fontsize=9)
ax3.legend(fontsize=9); ax3.grid(alpha=0.2)
ax3.set_title(f'Single Point FRF — Sample {SAMPLE_IDX} | True: {pk_strs} Hz | Pred: {pred_strs} Hz', fontsize=12)
fig3.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_single.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Single: {out_dir}/对比图_single.png')

# 图4: 复数FRF (Re+Im+Amplitude) 单点 — 与测试.py图3一致
fig4, (ar, ai, aa) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
ar.plot(xs, t_re[idx1], 'b-', linewidth=1.2, label='Target', alpha=0.9)
ar.plot(xs, p_re[idx1], 'r--', linewidth=1.2, label='Predicted', alpha=0.9)
ar.fill_between(xs, 0, t_re[idx1], alpha=0.08, color='blue')
ar.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ar.set_ylabel('Real Part'); ar.set_title(f'Real @ x_max (x={points[idx1,0]:.4f},y={points[idx1,1]:.4f})')
ar.legend(fontsize=8); ar.grid(alpha=0.2)

ai.plot(xs, t_im[idx1], 'b-', linewidth=1.2, label='Target', alpha=0.9)
ai.plot(xs, p_im[idx1], 'r--', linewidth=1.2, label='Predicted', alpha=0.9)
ai.fill_between(xs, 0, t_im[idx1], alpha=0.08, color='red')
ai.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ai.set_ylabel('Imaginary Part'); ai.set_title('Imaginary @ x_max')
ai.legend(fontsize=8); ai.grid(alpha=0.2)

aa.plot(xs, t_amp[idx1], 'b-', linewidth=1.2, label='Target', alpha=0.9)
aa.plot(xs, p_amp[idx1], 'r--', linewidth=1.2, label='Predicted', alpha=0.9)
for mp in pk:
    aa.axvline(np.interp(mp, f_np, xs), color='green', linestyle=':', linewidth=0.8, alpha=0.6)
aa.set_ylabel('Amplitude'); aa.set_title('Amplitude = sqrt(Re^2 + Im^2)')
aa.legend(fontsize=8); aa.grid(alpha=0.2)

aa.set_xticks(tls); aa.set_xticklabels(tlbs, fontsize=8); aa.set_xlabel('Hz', fontsize=10)
fig4.suptitle(f'Complex FRF — Sample {SAMPLE_IDX} | True: {pk_strs} Hz | Pred: {pred_strs} Hz', fontsize=12)
fig4.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图_complex.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Complex: {out_dir}/对比图_complex.png')
