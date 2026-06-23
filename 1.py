# -*- coding: utf-8 -*-
"""
FRF 重建脚本：使用三个最优模型（频率、阻尼、振型）重建频率响应函数并与真实值对比。

公式：H(x, ω) = Σ_{r=1}^{3}  A_r(x) / (ω_r² - ω² + j·2·ζ_r·ω_r·ω)
其中：A_r(x) = φ_r,z(x) × φ_r,z(x_excitation)

使用模型：
  - 频率 ω_r：shape模型内嵌的frequency_full模型（测试集误差0.43%）
  - 阻尼 ζ_r：独立的DampingTokenMLP（测试集误差2.44%）
  - 振型 φ_z：SymmetricSymlogModalOperator
"""
import sys
import os

# Force stdout to UTF-8 to prevent GBK encoding issues on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # best_modified/frf
BEST_MODIFIED_DIR = os.path.dirname(SCRIPT_DIR) # best_modified
PROJECT_ROOT = os.path.dirname(os.path.dirname(BEST_MODIFIED_DIR)) # stage1-modal-residue-dataset

# 导入包路径设置
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(BEST_MODIFIED_DIR, "training_shape_optimized"))
sys.path.insert(0, os.path.join(BEST_MODIFIED_DIR, "training_frequency_optimized"))
sys.path.insert(0, os.path.join(BEST_MODIFIED_DIR, "training_damping_optimized"))

from dataset import SymmetricSymlogModalDataset
from model import SymmetricSymlogModalOperator
from model_frequency import FrequencyTokenMLP as FrequencyModel
from model_damping import DampingTokenMLP as DampingModel

# ============================================================================
# 配置
# ============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "data_modal_residue_stage1500")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")

SHAPE_CKPT = os.path.join(BEST_MODIFIED_DIR, "training_shape_optimized", "checkpoints", "best_model.pth")
FREQ_CKPT = os.path.join(BEST_MODIFIED_DIR, "training_frequency_optimized", "runs", "best_frequency_model.pt")
DAMPING_CKPT = os.path.join(BEST_MODIFIED_DIR, "training_damping_optimized", "runs", "best_damping_model.pt")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "frf_reconstruction")

TARGET_MODES = 3
N_FREQ_POINTS = 500  # FRF频率分辨率

# 选择的样本和节点（将自动选择有代表性的）
SAMPLE_INDICES = [0, 5, 10, 20, 50]  # 测试集样本索引
NODES_PER_SAMPLE = 4  # 每个样本选几个节点

L_BASE = 0.160
W_BASE = 0.060
H_BASE = 0.010
COORD_SCALE = np.array([L_BASE, W_BASE, H_BASE], dtype=np.float32)


def load_shape_model(device):
    """加载振型模型"""
    print(f"加载振型模型: {SHAPE_CKPT}")
    model = SymmetricSymlogModalOperator(target_modes=3).to(device)
    model.load_state_dict(torch.load(SHAPE_CKPT, map_location=device))
    model.eval()
    return model


def load_frequency_model(device):
    """加载频率模型"""
    print(f"加载频率模型: {FREQ_CKPT}")
    ckpt = torch.load(FREQ_CKPT, map_location="cpu")
    
    model = FrequencyModel(
        pocket_dim=8, clamp_dim=11, global_dim=9,
        token_dim=96, hidden_dim=192, fusion_dim=256,
        out_modes=TARGET_MODES, token_layers=3, fusion_layers=4, dropout=0.05
    ).to(device)
    
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    stats = {}
    if "stats" in ckpt:
        stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
    return model, stats


def load_damping_model(device):
    """加载阻尼模型"""
    print(f"加载阻尼模型: {DAMPING_CKPT}")
    ckpt = torch.load(DAMPING_CKPT, map_location="cpu")
    
    model = DampingModel(
        pocket_dim=8, clamp_dim=11, global_dim=13,
        token_dim=96, hidden_dim=192, fusion_dim=256,
        out_modes=TARGET_MODES, token_layers=3, fusion_layers=4, dropout=0.05
    ).to(device)
    
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    stats = {}
    if "stats" in ckpt:
        stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
    return model, stats


def read_scalar(group, key, default=0.0):
    """从H5 group中读取标量"""
    if key not in group:
        return float(default)
    arr = np.asarray(group[key])
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def read_material_features(group):
    """读取材料特征 E_ratio 和 rho_ratio"""
    if "point_features" not in group:
        return 1.0, 1.0
    pf = np.asarray(group["point_features"], dtype=np.float32)
    if pf.ndim != 2 or pf.shape[0] == 0 or pf.shape[1] < 3:
        return 1.0, 1.0
    return float(pf[0, 0]), float(pf[0, 2])


def predict_frequency(freq_model, freq_stats, group, device, eps=1e-6):
    """用频率模型预测固有频率"""
    if "pocket_features" in group:
        pocket = np.asarray(group["pocket_features"], dtype=np.float32)
        if pocket.shape != (7, 8):
            fixed = np.zeros((7, 8), dtype=np.float32)
            fixed[:min(7, pocket.shape[0]), :min(8, pocket.shape[1])] = pocket[:min(7, pocket.shape[0]), :min(8, pocket.shape[1])]
            pocket = fixed
    else:
        pocket = np.zeros((7, 8), dtype=np.float32)
    
    if "clamp_features" in group:
        clamp = np.asarray(group["clamp_features"], dtype=np.float32)
        if clamp.shape != (7, 11):
            fixed = np.zeros((7, 11), dtype=np.float32)
            fixed[:min(7, clamp.shape[0]), :min(11, clamp.shape[1])] = clamp[:min(7, clamp.shape[0]), :min(11, clamp.shape[1])]
            clamp = fixed
    else:
        clamp = np.zeros((7, 11), dtype=np.float32)
    
    e_ratio, rho_ratio = read_material_features(group)
    layout_type = read_scalar(group, "layout_type", 0.0)
    coverage_code = read_scalar(group, "coverage_level_code", 0.0)
    clamp_code = read_scalar(group, "clamp_level_code", read_scalar(group, "clamp_model_code", 0.0))
    removed_volume_ratio = read_scalar(group, "removed_volume_ratio", 0.0)
    grid_jitter = read_scalar(group, "grid_jitter", 0.0)
    finished_count = read_scalar(group, "finished_count", 0.0)
    current_progress = read_scalar(group, "current_progress", 1.0)
    
    clamp_norm = clamp.copy()
    clamp_norm[:, 5:8] /= 12.0
    clamp_norm[:, 8:11] /= 8.0
    
    global_features = np.asarray([
        e_ratio, rho_ratio,
        layout_type / 7.0, coverage_code / 2.0, clamp_code / 2.0,
        removed_volume_ratio, grid_jitter, finished_count / 7.0, current_progress
    ], dtype=np.float32)
    
    p = torch.from_numpy(pocket).unsqueeze(0).float().to(device)
    c = torch.from_numpy(clamp_norm).unsqueeze(0).float().to(device)
    g = torch.from_numpy(global_features).unsqueeze(0).float().to(device)
    
    if freq_stats:
        p = (p - freq_stats["pocket_features_mean"].view(1, 1, -1)) / freq_stats["pocket_features_std"].view(1, 1, -1).clamp_min(eps)
        c = (c - freq_stats["clamp_features_mean"].view(1, 1, -1)) / freq_stats["clamp_features_std"].view(1, 1, -1).clamp_min(eps)
        g = (g - freq_stats["global_features_mean"].view(1, -1)) / freq_stats["global_features_std"].view(1, -1).clamp_min(eps)
        
    with torch.no_grad():
        pred_norm = freq_model(p, c, g)
        
    if freq_stats and "omega_log_mean" in freq_stats:
        logw = pred_norm * freq_stats["omega_log_std"].view(1, -1) + freq_stats["omega_log_mean"].view(1, -1)
        omega = torch.exp(logw).clamp_min(eps)
    else:
        omega = pred_norm
        
    return omega.squeeze(0).cpu().numpy()


def predict_damping(damping_model, damping_stats, group, omega_pred, phi_z_norm_pred, device, eps=1e-6):
    """用阻尼模型预测阻尼比"""
    if "pocket_features" in group:
        pocket = np.asarray(group["pocket_features"], dtype=np.float32)
        if pocket.shape != (7, 8):
            fixed = np.zeros((7, 8), dtype=np.float32)
            fixed[:min(7, pocket.shape[0]), :min(8, pocket.shape[1])] = pocket[:min(7, pocket.shape[0]), :min(8, pocket.shape[1])]
            pocket = fixed
    else:
        pocket = np.zeros((7, 8), dtype=np.float32)
    
    if "clamp_features" in group:
        clamp = np.asarray(group["clamp_features"], dtype=np.float32)
        if clamp.shape != (7, 11):
            fixed = np.zeros((7, 11), dtype=np.float32)
            fixed[:min(7, clamp.shape[0]), :min(11, clamp.shape[1])] = clamp[:min(7, clamp.shape[0]), :min(11, clamp.shape[1])]
            clamp = fixed
    else:
        clamp = np.zeros((7, 11), dtype=np.float32)
    
    e_ratio, rho_ratio = read_material_features(group)
    layout_type = read_scalar(group, "layout_type", 0.0)
    coverage_code = read_scalar(group, "coverage_level_code", 0.0)
    clamp_code = read_scalar(group, "clamp_level_code", read_scalar(group, "clamp_model_code", 0.0))
    removed_volume_ratio = read_scalar(group, "removed_volume_ratio", 0.0)
    grid_jitter = read_scalar(group, "grid_jitter", 0.0)
    finished_count = read_scalar(group, "finished_count", 0.0)
    current_progress = read_scalar(group, "current_progress", 1.0)
    
    clamp_norm = clamp.copy()
    clamp_norm[:, 5:8] /= 12.0
    clamp_norm[:, 8:11] /= 8.0
    
    pocket_centers = np.zeros((7, 2), dtype=np.float32)
    pocket_centers[:, 0] = (pocket[:, 0] + pocket[:, 1]) / 2.0
    pocket_centers[:, 1] = (pocket[:, 2] + pocket[:, 3]) / 2.0
    
    clamp_centers = np.zeros((7, 2), dtype=np.float32)
    clamp_centers[:, 0] = (clamp_norm[:, 0] + clamp_norm[:, 1]) / 2.0
    clamp_centers[:, 1] = (clamp_norm[:, 2] + clamp_norm[:, 3]) / 2.0
    
    global_features = np.asarray([
        layout_type / 7.0, coverage_code / 2.0, clamp_code / 2.0,
        removed_volume_ratio, grid_jitter, finished_count / 7.0, current_progress
    ], dtype=np.float32)
    global_features = np.concatenate([global_features, omega_pred, phi_z_norm_pred]).astype(np.float32)
    
    log_material_scale = -0.5 * np.log(e_ratio * rho_ratio)
    
    p = torch.from_numpy(pocket).unsqueeze(0).float().to(device)
    p_c = torch.from_numpy(pocket_centers).unsqueeze(0).float().to(device)
    c = torch.from_numpy(clamp_norm).unsqueeze(0).float().to(device)
    c_c = torch.from_numpy(clamp_centers).unsqueeze(0).float().to(device)
    g = torch.from_numpy(global_features).unsqueeze(0).float().to(device)
    scale = torch.tensor([log_material_scale], dtype=torch.float32).to(device)
    
    if damping_stats:
        p = (p - damping_stats["pocket_features_mean"].view(1, 1, -1)) / damping_stats["pocket_features_std"].view(1, 1, -1).clamp_min(eps)
        c = (c - damping_stats["clamp_features_mean"].view(1, 1, -1)) / damping_stats["clamp_features_std"].view(1, 1, -1).clamp_min(eps)
        # Note: damping_stats["global_features_mean"] now has 13 elements, matching our new g
        g = (g - damping_stats["global_features_mean"].view(1, -1)) / damping_stats["global_features_std"].view(1, -1).clamp_min(eps)
    
    with torch.no_grad():
        pred_norm = damping_model(p, p_c, c, c_c, g)  # [1, 3]
    
    # 反归一化得到物理阻尼比
    if damping_stats and "zeta_log_mean" in damping_stats:
        logz_base = pred_norm * damping_stats["zeta_log_std"].view(1, -1) + damping_stats["zeta_log_mean"].view(1, -1)
        logz = logz_base + scale.view(-1, 1)
        zeta = torch.exp(logz).clamp_min(eps) + 0.002
    else:
        zeta = pred_norm  # fallback
    
    return zeta.squeeze(0).cpu().numpy()  # [3]


def predict_shape_and_freq(shape_model, dataset_item, device):
    """用振型模型预测频率和振型"""
    pocket = dataset_item["pocket_features"].unsqueeze(0).to(device)
    clamp = dataset_item["clamp_features"].unsqueeze(0).to(device)
    gf = dataset_item["global_features"].unsqueeze(0).to(device)
    q_coord = dataset_item["q_coord"].unsqueeze(0).to(device)
    q_node = dataset_item["q_node_features"].unsqueeze(0).to(device)
    
    p_coord = dataset_item["p_coord"].to(device).unsqueeze(0).unsqueeze(1)
    p_node = dataset_item["p_node_features"].to(device).unsqueeze(0).unsqueeze(1)
    
    qn = q_coord.shape[1]
    all_coords = torch.cat([q_coord, p_coord], dim=1)
    all_nodes = torch.cat([q_node, p_node], dim=1)
    
    with torch.no_grad():
        outputs = shape_model(pocket, clamp, gf, all_coords, all_nodes)
        omega_pred = outputs["omega"].squeeze(0).cpu().numpy()  # [3] rad/s
        phi_all = outputs["symlog_phi_z"].squeeze(0).cpu().numpy()  # [N+1, 3]
    
    phi_q = phi_all[:qn, :]     # 响应节点的振型 [N, 3]
    phi_p = phi_all[qn:, :].squeeze(0)  # 激励点的振型 [3]
    
    return omega_pred, phi_q, phi_p


def compute_frf(omega_r, zeta_r, residue, freq_hz):
    """
    计算 FRF（频率响应函数）
    
    H(x, f) = Σ_r A_r / (ω_r² - ω² + j·2·ζ_r·ω_r·ω)
    
    Parameters:
        omega_r: [K] 固有角频率 (rad/s)
        zeta_r: [K] 阻尼比
        residue: [N, K] 模态留数 A_r(x) = φ_z(x) * φ_z(x_exc)
        freq_hz: [F] 频率点 (Hz)
    
    Returns:
        frf_complex: [N, F] 复数FRF
    """
    omega_q = 2.0 * np.pi * freq_hz  # [F]
    n_nodes = residue.shape[0]
    n_freqs = len(freq_hz)
    
    frf_real = np.zeros((n_nodes, n_freqs), dtype=np.float64)
    frf_imag = np.zeros((n_nodes, n_freqs), dtype=np.float64)
    
    for r in range(len(omega_r)):
        wr = omega_r[r]
        zr = zeta_r[r]
        ar = residue[:, r]  # [N]
        
        dw = wr**2 - omega_q**2       # [F]
        gm = 2.0 * zr * wr * omega_q  # [F]
        denom = dw**2 + gm**2 + 1e-30  # [F]
        
        frf_real += np.outer(ar, dw / denom)    # [N, F]
        frf_imag += np.outer(ar, -gm / denom)   # [N, F]
    
    return frf_real + 1j * frf_imag


def select_representative_nodes(coords_normalized, n_nodes, n_select=4):
    """
    选择有代表性的节点：四角附近 + 中心
    coords_normalized: [N, 3] 归一化坐标（0~1范围）
    """
    if n_nodes <= n_select:
        return list(range(n_nodes))
    
    # 选择策略：
    # 1. 中心点
    # 2. 左下角附近
    # 3. 右上角附近
    # 4. 中间偏上
    targets = np.array([
        [0.5, 0.5, 0.5],   # 中心
        [0.2, 0.2, 0.5],   # 左下
        [0.8, 0.8, 0.5],   # 右上
        [0.5, 0.8, 0.5],   # 中上
    ], dtype=np.float32)
    
    selected = []
    for t in targets[:n_select]:
        dists = np.linalg.norm(coords_normalized - t.reshape(1, 3), axis=1)
        idx = np.argmin(dists)
        # 避免重复
        while idx in selected and len(selected) < n_nodes:
            dists[idx] = 1e10
            idx = np.argmin(dists)
        selected.append(idx)
    
    return selected


def plot_frf_comparison(freq_hz, frf_true, frf_pred, sample_idx, node_idx, 
                        node_coord_mm, save_dir, omega_true, omega_pred, 
                        zeta_true, zeta_pred):
    """绘制单个节点的FRF真实值vs预测值对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    
    # 幅值（dB）
    amp_true = np.abs(frf_true)
    amp_pred = np.abs(frf_pred)
    
    # 避免 log(0)
    amp_true_db = 20 * np.log10(np.maximum(amp_true, 1e-20))
    amp_pred_db = 20 * np.log10(np.maximum(amp_pred, 1e-20))
    
    axes[0].plot(freq_hz, amp_true_db, 'b-', linewidth=1.5, label='真实值 (Ground Truth)', alpha=0.8)
    axes[0].plot(freq_hz, amp_pred_db, 'r--', linewidth=1.5, label='预测值 (Prediction)', alpha=0.8)
    axes[0].set_xlabel('频率 (Hz)', fontsize=12)
    axes[0].set_ylabel('幅值 (dB, ref: 1 m/N)', fontsize=12)
    axes[0].set_title(f'样本 {sample_idx} - 节点 {node_idx} 坐标({node_coord_mm[0]:.1f}, {node_coord_mm[1]:.1f}, {node_coord_mm[2]:.1f}) mm\n'
                      f'FRF 幅值对比', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 标注共振峰
    for k in range(len(omega_true)):
        f_true = omega_true[k] / (2 * np.pi)
        f_pred = omega_pred[k] / (2 * np.pi)
        axes[0].axvline(f_true, color='blue', alpha=0.3, linestyle=':')
        axes[0].axvline(f_pred, color='red', alpha=0.3, linestyle=':')
    
    # 相位
    phase_true_rad = np.angle(frf_true)
    phase_pred_rad = np.angle(frf_pred)
    
    # 进行相位解包裹 (unwrap)，使图线平滑，消除 ±180° 跳变的视觉干扰
    phase_true_unwrapped = np.degrees(np.unwrap(phase_true_rad))
    phase_pred_unwrapped = np.degrees(np.unwrap(phase_pred_rad))
    
    axes[1].plot(freq_hz, phase_true_unwrapped, 'b-', linewidth=1.5, label='真实值 (Ground Truth)', alpha=0.8)
    axes[1].plot(freq_hz, phase_pred_unwrapped, 'r--', linewidth=1.5, label='预测值 (Prediction)', alpha=0.8)
    axes[1].set_xlabel('频率 (Hz)', fontsize=12)
    axes[1].set_ylabel('相位 (°)', fontsize=12)
    axes[1].set_title('FRF 相位对比 (已去包裹)', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    # 取消硬编码的 -200 到 200 限制，让 matplotlib 根据解包裹后的范围自适应缩放
    
    # 添加模态参数信息文本框
    info_text = "模态参数对比:\n"
    for k in range(len(omega_true)):
        f_t = omega_true[k] / (2*np.pi)
        f_p = omega_pred[k] / (2*np.pi)
        z_t = zeta_true[k]
        z_p = zeta_pred[k]
        info_text += f"Mode {k+1}: f真={f_t:.1f}Hz / f预={f_p:.1f}Hz (Δ={abs(f_p-f_t)/f_t*100:.2f}%) | "
        info_text += f"ζ真={z_t:.4f} / ζ预={z_p:.4f} (Δ={abs(z_p-z_t)/z_t*100:.1f}%)\n"
    
    fig.text(0.05, 0.01, info_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    fname = f"frf_sample{sample_idx}_node{node_idx}.png"
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_frf_grid(freq_hz, frf_trues, frf_preds, sample_idx, node_indices, 
                  node_coords_mm, save_dir, omega_true, omega_pred):
    """为一个样本绘制多节点FRF对比网格图"""
    n_nodes = len(node_indices)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(14, 4*n_nodes), dpi=150)
    if n_nodes == 1:
        axes = [axes]
    
    for i, (ax, nidx) in enumerate(zip(axes, node_indices)):
        amp_true = 20 * np.log10(np.maximum(np.abs(frf_trues[i]), 1e-20))
        amp_pred = 20 * np.log10(np.maximum(np.abs(frf_preds[i]), 1e-20))
        
        ax.plot(freq_hz, amp_true, 'b-', linewidth=1.2, label='真实值', alpha=0.8)
        ax.plot(freq_hz, amp_pred, 'r--', linewidth=1.2, label='预测值', alpha=0.8)
        coord = node_coords_mm[i]
        ax.set_title(f'节点 {nidx} ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) mm', fontsize=11)
        ax.set_ylabel('幅值 (dB)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 标注共振频率
        for k in range(len(omega_true)):
            f_t = omega_true[k] / (2*np.pi)
            f_p = omega_pred[k] / (2*np.pi)
            ax.axvline(f_t, color='blue', alpha=0.2, linestyle=':')
            ax.axvline(f_p, color='red', alpha=0.2, linestyle=':')
    
    axes[-1].set_xlabel('频率 (Hz)', fontsize=12)
    fig.suptitle(f'样本 {sample_idx} — 多节点 FRF 幅值对比 (真实 vs 预测)', fontsize=14, y=1.01)
    plt.tight_layout()
    
    fname = f"frf_grid_sample{sample_idx}.png"
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def main():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"输出目录: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========== 加载模型 ===========
    shape_model = load_shape_model(device)
    freq_model, freq_stats = load_frequency_model(device)
    damping_model, damping_stats = load_damping_model(device)
    
    # =========== 加载数据集 ===========
    print(f"加载测试数据集: {TEST_H5}")
    dataset = SymmetricSymlogModalDataset(
        h5_path=TEST_H5, target_modes=TARGET_MODES, 
        query_per_sample=-1, random_query=False
    )
    n_total = len(dataset)
    print(f"测试集总样本数: {n_total}")
    
    # 调整样本索引
    sample_indices = [i for i in SAMPLE_INDICES if i < n_total]
    print(f"将处理的样本: {sample_indices}")
    
    all_results = []
    
    for si, sample_idx in enumerate(sample_indices):
        print(f"\n{'='*60}")
        print(f"处理样本 {sample_idx} ({si+1}/{len(sample_indices)})")
        print(f"{'='*60}")
        
        # 获取 dataset item
        item = dataset[sample_idx]
        sample_key = dataset.sample_keys[sample_idx]
        
        # ===== 1. 用振型模型预测振型 =====
        _, phi_q_pred, phi_p_pred = predict_shape_and_freq(shape_model, item, device)
        
        # Compute predicted shape norm (consistent with training)
        phi_all_pred = np.concatenate([phi_q_pred, phi_p_pred[np.newaxis, :]], axis=0)
        phi_z_norm_pred = np.linalg.norm(phi_all_pred, axis=0) # [3]
        
        # ===== 2. 用独立频率和阻尼模型预测 =====
        with h5py.File(TEST_H5, "r") as h5:
            g = h5[sample_key]
            omega_pred = predict_frequency(freq_model, freq_stats, g, device)
            zeta_pred = predict_damping(damping_model, damping_stats, g, omega_pred, phi_z_norm_pred, device)
            print(f"  预测频率 (Hz): {omega_pred / (2*np.pi)}")
            
            # ===== 3. 读取真实值 =====
            omega_true = np.asarray(g["modal_omega"], dtype=np.float32)[:TARGET_MODES]
            zeta_true = np.asarray(g["modal_zeta"], dtype=np.float32)[:TARGET_MODES]
            residue_true = np.asarray(g["modal_residue_z"], dtype=np.float32)[:, :TARGET_MODES]
            phi_z_true = np.asarray(g["modal_phi_z"], dtype=np.float32)[:, :TARGET_MODES]
            
            points = np.asarray(g["points"], dtype=np.float32)[:, :3]  # 物理坐标 (m)
            exc_idx = int(np.asarray(g["excitation_index"]).reshape(-1)[0])
            
            # 真实的全模态FRF计算（10阶）
            omega_all = np.asarray(g["modal_omega"], dtype=np.float32)
            zeta_all = np.asarray(g["modal_zeta"], dtype=np.float32)
            residue_all = np.asarray(g["modal_residue_z"], dtype=np.float32)
            n_all_modes = min(10, omega_all.shape[0])
            
            # 如果H5中保存了FRF频率网格
            if "frequencies" in g and np.asarray(g["frequencies"]).size > 0:
                freq_hz = np.asarray(g["frequencies"], dtype=np.float64)
            else:
                f_max = float(omega_true.max() / (2*np.pi)) * 1.5
                freq_hz = np.linspace(1.0, f_max, N_FREQ_POINTS)
            
            # 同时包含已保存的真实FRF（如果有的话）
            has_saved_frf = ("point_frf" in g and np.asarray(g["point_frf"]).shape[0] > 0 
                            and np.asarray(g["point_frf"]).shape[1] > 0)
            if has_saved_frf:
                saved_frf = np.asarray(g["point_frf"], dtype=np.float64)  # [N, F, 2]
                print(f"  ✓ H5中有保存的完整FRF数据 (shape: {saved_frf.shape})")
            else:
                print(f"  ✗ H5中无保存的完整FRF数据，将用10阶模态参数重建真实FRF")
        
        print(f"  真实频率 (Hz): {omega_true / (2*np.pi)}")
        print(f"  预测阻尼比: {zeta_pred}")
        print(f"  真实阻尼比: {zeta_true}")
        
        # ===== 4. 计算预测的留数 =====
        # 预测留数 A_pred(x) = φ_pred(x) * φ_pred(x_exc)
        residue_pred = phi_q_pred * phi_p_pred[np.newaxis, :]  # [N, 3]
        
        # ===== 5. 选择代表性节点 =====
        coords_norm = item["q_coord"].numpy()  # 归一化坐标 [N, 3]
        n_nodes = coords_norm.shape[0]
        node_sel = select_representative_nodes(coords_norm, n_nodes, NODES_PER_SAMPLE)
        print(f"  选择的节点索引: {node_sel}")
        
        # ===== 6. 构建频率轴 =====
        f_max = max(float(omega_true.max()), float(omega_pred.max())) / (2*np.pi) * 1.3
        f_min = max(1.0, min(float(omega_true.min()), float(omega_pred.min())) / (2*np.pi) * 0.5)
        freq_hz_plot = np.linspace(f_min, f_max, N_FREQ_POINTS)
        
        # ===== 7. 计算真实FRF（使用全部10阶模态参数） =====
        if has_saved_frf:
            # 有保存的FRF，但频率轴可能不同，这里重新用模态参数计算
            pass
        
        frf_true_full = compute_frf(
            omega_all[:n_all_modes], zeta_all[:n_all_modes], 
            residue_all[:, :n_all_modes], freq_hz_plot
        )
        
        # 真实的3阶FRF（公平对比）
        frf_true_3mode = compute_frf(omega_true, zeta_true, residue_true, freq_hz_plot)
        
        # ===== 8. 计算预测FRF =====
        frf_pred = compute_frf(omega_pred, zeta_pred, residue_pred, freq_hz_plot)
        
        # ===== 9. 绘图 =====
        node_coords_mm_list = []
        frf_trues_sel = []
        frf_preds_sel = []
        
        for ni, nidx in enumerate(node_sel):
            coord_mm = coords_norm[nidx] * COORD_SCALE * 1000  # 转mm
            node_coords_mm_list.append(coord_mm)
            
            # 单节点FRF
            frf_t = frf_true_3mode[nidx, :]
            frf_p = frf_pred[nidx, :]
            frf_trues_sel.append(frf_t)
            frf_preds_sel.append(frf_p)
            
            # 单节点详细对比图
            save_path = plot_frf_comparison(
                freq_hz_plot, frf_t, frf_p, sample_idx, nidx,
                coord_mm, OUTPUT_DIR, omega_true, omega_pred,
                zeta_true, zeta_pred
            )
            print(f"  ✓ 已保存: {os.path.basename(save_path)}")
        
        # 多节点网格图
        grid_path = plot_frf_grid(
            freq_hz_plot, frf_trues_sel, frf_preds_sel, sample_idx, node_sel,
            node_coords_mm_list, OUTPUT_DIR, omega_true, omega_pred
        )
        print(f"  ✓ 已保存网格图: {os.path.basename(grid_path)}")
        
        # 收集统计信息
        freq_err_pct = np.abs(omega_pred - omega_true) / omega_true * 100
        zeta_err_pct = np.abs(zeta_pred - zeta_true) / zeta_true * 100
        
        result = {
            "sample_idx": sample_idx,
            "omega_true_hz": omega_true / (2*np.pi),
            "omega_pred_hz": omega_pred / (2*np.pi),
            "freq_err_pct": freq_err_pct,
            "zeta_true": zeta_true,
            "zeta_pred": zeta_pred,
            "zeta_err_pct": zeta_err_pct,
            "n_nodes": n_nodes,
        }
        all_results.append(result)
    
    # ===== 10. 打印总结 =====
    print(f"\n{'='*70}")
    print(f"FRF 重建总结")
    print(f"{'='*70}")
    print(f"{'样本':>6} | {'f1真/预(Hz)':>20} | {'f2真/预(Hz)':>20} | {'f3真/预(Hz)':>20} | {'ζ均误差%':>8}")
    print("-" * 90)
    for r in all_results:
        f1 = f"{r['omega_true_hz'][0]:.1f}/{r['omega_pred_hz'][0]:.1f}"
        f2 = f"{r['omega_true_hz'][1]:.1f}/{r['omega_pred_hz'][1]:.1f}"
        f3 = f"{r['omega_true_hz'][2]:.1f}/{r['omega_pred_hz'][2]:.1f}"
        z_mean = np.mean(r['zeta_err_pct'])
        print(f"{r['sample_idx']:>6} | {f1:>20} | {f2:>20} | {f3:>20} | {z_mean:>7.2f}%")
    
    print(f"\n所有图片保存在: {OUTPUT_DIR}")
    print("完成！")


if __name__ == "__main__":
    main()
