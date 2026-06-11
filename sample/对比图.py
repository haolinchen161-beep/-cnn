"""
对比图dB：左边真实值，右边预测值，对数dB绘图.
用法: F:\pytorch_cuda12\python.exe sample\对比图db.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SAMPLE_IDX = 2

base_dir = os.path.dirname(__file__)
out_dir = os.path.join(base_dir, 'output')
npz_path = os.path.join(out_dir, 'final_results.npz')

data = np.load(npz_path, allow_pickle=True)

f_np = np.asarray(data['frequencies'][SAMPLE_IDX], dtype=float)
points = np.asarray(data['points'][SAMPLE_IDX], dtype=float)

t_amp = np.asarray(data['target_frf'][SAMPLE_IDX], dtype=float)
p_amp = np.asarray(data['predicted_frf'][SAMPLE_IDX], dtype=float)

true_freq = np.asarray(data['true_freq_hz'][SAMPLE_IDX], dtype=float).reshape(-1)
pred_freq = np.asarray(data['pred_freq_hz'][SAMPLE_IDX], dtype=float).reshape(-1)
freq_rel = np.asarray(data['freq_rel'][SAMPLE_IDX], dtype=float).reshape(-1)
true_zeta = np.asarray(data['true_zeta'][SAMPLE_IDX], dtype=float).reshape(-1)
pred_zeta = np.asarray(data['pred_zeta'][SAMPLE_IDX], dtype=float).reshape(-1)
phi_mac = np.asarray(data['phi_mac'][SAMPLE_IDX], dtype=float).reshape(-1)

print(f"Sample {SAMPLE_IDX}: FRF shape={t_amp.shape}, Freq=[{f_np[0]:.1f}, {f_np[-1]:.1f}] Hz")
print("True modes Hz:", [f"{x:.2f}" for x in true_freq])
print("Pred modes Hz:", [f"{x:.2f}" for x in pred_freq])


def to_db(amp):
    return 20.0 * np.log10(np.maximum(amp, 1e-30))


# 选择展示节点：最大幅值 + 最大误差 + 均匀分布
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
n_sel = len(selected)

# 转dB
t_db = to_db(t_amp)
p_db = to_db(p_amp)


# ============================================================
# 图：左边真实FRF(dB)，右边预测FRF(dB)
# ============================================================
fig, axes = plt.subplots(n_sel, 2, figsize=(18, 3.2 * n_sel), sharex=True)
if n_sel == 1:
    axes = np.expand_dims(axes, axis=0)

axes[0, 0].set_title('Target FRF (dB)', fontsize=13, fontweight='bold')
axes[0, 1].set_title('Predicted FRF (dB)', fontsize=13, fontweight='bold')

for row, (si, tag) in enumerate(zip(selected, selected_tags)):
    ax_true = axes[row, 0]
    ax_pred = axes[row, 1]

    coord_str = f"({points[si,0]:.3f}, {points[si,1]:.3f})"

    # 左：真实
    ax_true.plot(f_np, t_db[si], 'b-', linewidth=1.0, alpha=0.9)
    for mp in true_freq:
        ax_true.axvline(mp, color='green', linestyle='-', linewidth=0.8, alpha=0.6)
    for mp in pred_freq:
        ax_true.axvline(mp, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    db_max_t = t_db[si].max()
    ax_true.set_ylim(db_max_t - 60, db_max_t + 5)
    ax_true.set_ylabel(f"{tag}\n{coord_str}\nMag (dB)", fontsize=8)
    ax_true.grid(alpha=0.2)
    # 标注真实频率
    for k, mp in enumerate(true_freq):
        y_top = db_max_t + 2
        ax_true.annotate(f"f{k+1}={mp:.1f}", xy=(mp, db_max_t - 2),
                         fontsize=6, color='green', ha='center', va='bottom')

    # 右：预测
    ax_pred.plot(f_np, p_db[si], 'r-', linewidth=1.0, alpha=0.9)
    for mp in true_freq:
        ax_pred.axvline(mp, color='green', linestyle='-', linewidth=0.8, alpha=0.6)
    for mp in pred_freq:
        ax_pred.axvline(mp, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    db_max_p = p_db[si].max()
    ax_pred.set_ylim(db_max_p - 60, db_max_p + 5)
    ax_pred.set_ylabel(f"{tag}\n{coord_str}\nMag (dB)", fontsize=8)
    ax_pred.grid(alpha=0.2)
    # 标注预测频率
    for k, mp in enumerate(pred_freq):
        ax_pred.annotate(f"f{k+1}={mp:.1f}", xy=(mp, db_max_p - 2),
                         fontsize=6, color='red', ha='center', va='bottom')

# x轴
for ax in axes[-1, :]:
    ax.set_xlabel('Frequency (Hz)', fontsize=10)

# 图例说明
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', linestyle='-', linewidth=0.8, label='True mode'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=0.8, label='Pred mode'),
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=9)

fig.suptitle(f"FRF Comparison (dB) — Sample {SAMPLE_IDX} | "
             f"True f=[{','.join(f'{x:.1f}' for x in true_freq)}] Hz | "
             f"Pred f=[{','.join(f'{x:.1f}' for x in pred_freq)}] Hz\n"
             f"Freq rel: [{', '.join(f'{x*100:.2f}%' for x in freq_rel)}]",
             fontsize=11)
fig.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图db.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(out_dir, '对比图db.png')}")


# ============================================================
# 图2：单点对比（真实vs预测叠加在一张dB图上）
# ============================================================
fig2, axes2 = plt.subplots(n_sel, 1, figsize=(16, 3.0 * n_sel), sharex=True)
if n_sel == 1:
    axes2 = [axes2]

for row, (si, tag) in enumerate(zip(selected, selected_tags)):
    ax = axes2[row]
    coord_str = f"({points[si,0]:.3f}, {points[si,1]:.3f})"
    ax.plot(f_np, t_db[si], 'b-', linewidth=1.0, label='Target', alpha=0.85)
    ax.plot(f_np, p_db[si], 'r--', linewidth=1.0, label='Predicted', alpha=0.85)
    for mp in true_freq:
        ax.axvline(mp, color='green', linestyle='-', linewidth=0.8, alpha=0.5)
    for mp in pred_freq:
        ax.axvline(mp, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    db_max = max(t_db[si].max(), p_db[si].max())
    ax.set_ylim(db_max - 60, db_max + 5)
    ax.set_ylabel(f"{tag}\n{coord_str}\n(dB)", fontsize=8)
    ax.grid(alpha=0.2)
    if row == 0:
        ax.legend(fontsize=8, loc='upper right')

axes2[-1].set_xlabel('Frequency (Hz)', fontsize=10)
fig2.suptitle(f"FRF Overlay (dB) — Sample {SAMPLE_IDX}", fontsize=12)
fig2.tight_layout()
plt.savefig(os.path.join(out_dir, '对比图db_overlay.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(out_dir, '对比图db_overlay.png')}")
