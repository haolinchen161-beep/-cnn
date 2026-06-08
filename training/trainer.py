"""
trainer.py — MeshGraphNet/GNN 两阶段训练循环 + 评估。

当前数据流：
    25D node_features + edge_index + edge_attr + batch + normalized frequencies
        → MeshGraphFRFModel
        → omega, zeta, phi_z
        → PhysicsDecoder
        → point_frf(Re, Im)

训练策略：
    Phase 1: 仅监督模态参数 omega / zeta / phi_z
    Phase 2: 模态监督 + PhysicsDecoder FRF 联合监督
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .losses import modal_loss, frf_loss


GRAPH_TENSOR_KEYS = [
    "node_features", "edge_index", "edge_attr", "batch",
    "points", "query_coords", "point_features",
    "spring_k_xyz", "spring_c_xyz", "node_type",
    "pocket_bottom_mask", "cut_region_mask",
    "excitation_index", "excitation_index_global", "excitation_coord",
    "modal_omega_norm", "modal_omega_phys", "modal_zeta",
    "modal_phi", "modal_phi_exc", "modal_phi_xyz",
    "point_frf",
]


def _move_graph_batch(batch: Dict, device: str) -> Dict:
    """把 collate_geometry_batch 返回的图 batch 移动到训练设备。"""
    out = dict(batch)
    for key in GRAPH_TENSOR_KEYS:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)
    if torch.is_tensor(out.get("frequencies")):
        out["frequencies"] = out["frequencies"].to(device, non_blocking=True)
    return out


def _forward_modal(net, batch: Dict, frequencies=None, excitation_index_global=None, alpha: float = 1.0):
    return net(
        batch["node_features"],
        batch["edge_index"],
        batch["edge_attr"],
        batch["batch"],
        frequencies=frequencies,
        excitation_index_global=excitation_index_global,
        alpha=alpha,
    )


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler=None, logger=None, start_epoch=0):
    """MeshGraphNet 两阶段训练入口。"""
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16))

    total_epochs = int(config.get("epochs", 2000))
    frf_weight = float(config.get("frf_loss_weight", 0.05))
    unlock_after_epoch = int(config.get("phase1_max_epochs", config.get("phase1_epochs", 600)))
    unlock_omega_pct = float(config.get("phase1_unlock_omega_pct", 0.5))

    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "loss_log.csv")
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "epoch", "phase", "train_loss", "omega_rel_%", "zeta_rel_%", "phi_loss",
            "omega_share_%", "zeta_share_%", "phi_share_%", "frf_share_%",
            "val_mse", "amp_mae", "amp_mape_%", "omega_mae_rad_s", "lr",
        ])

    phase2_unlocked = False
    unlock_epoch = start_epoch

    try:
        for epoch in range(start_epoch, total_epochs):
            losses, omega_losses, zeta_losses = [], [], []
            weighted_w_losses, weighted_z_losses, weighted_p_losses = [], [], []
            frf_weighted_losses = []
            in_phase2 = phase2_unlocked
            phase_name = "phase2" if in_phase2 else "phase1"

            if not in_phase2 and epoch == start_epoch:
                _log("=== Phase1: MeshGraphNet modal-only training ===", logger)
            elif in_phase2 and not getattr(net, "_phase2_logged", False):
                _log(f"=== Phase2: modal + FRF joint training, unlock epoch={epoch} ===", logger)
                net._phase2_logged = True
                lowest = np.inf

            for raw_batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                batch = _move_graph_batch(raw_batch, args.device)
                current_phi_w = float(
                    config.get("phi_loss_weight", 100.0) if in_phase2
                    else config.get("phase1_phi_loss_weight", 1.0)
                )

                with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                    if in_phase2:
                        phase2_epoch = epoch - unlock_epoch
                        alpha = max(1.0, 10.0 - 9.0 * phase2_epoch / 200.0)
                        frequencies = _require_tensor_frequencies(batch)
                        frf_pred, omega_pred, zeta_pred, phi_pred = _forward_modal(
                            net, batch, frequencies=frequencies,
                            excitation_index_global=batch.get("excitation_index_global"),
                            alpha=alpha
                        )
                        loss_m, l_w, l_z, l_p = modal_loss(
                            omega_pred, batch["modal_omega_norm"],
                            zeta_pred, batch["modal_zeta"],
                            phi_pred, batch["modal_phi"],
                            batch_idx=batch["batch"],
                            omega_weight=config.get("omega_loss_weight", 200.0),
                            zeta_weight=config.get("zeta_loss_weight", 10.0),
                            phi_weight=current_phi_w,
                        )
                        raw_frf = frf_loss(frf_pred, batch["point_frf"])
                        current_frf_w = frf_weight * min(1.0, max(0.0, phase2_epoch / 20.0))
                        loss_frf = current_frf_w * raw_frf
                        loss = loss_m + loss_frf
                    else:
                        _, omega_pred, zeta_pred, phi_pred = _forward_modal(net, batch)
                        loss_m, l_w, l_z, l_p = modal_loss(
                            omega_pred, batch["modal_omega_norm"],
                            zeta_pred, batch["modal_zeta"],
                            phi_pred, batch["modal_phi"],
                            batch_idx=batch["batch"],
                            omega_weight=config.get("omega_loss_weight", 200.0),
                            zeta_weight=config.get("zeta_loss_weight", 10.0),
                            phi_weight=current_phi_w,
                        )
                        loss_frf = torch.zeros((), device=loss_m.device, dtype=loss_m.dtype)
                        loss = loss_m

                losses.append(float(loss.detach().cpu()))
                omega_pct, zeta_pct = _modal_relative_errors(omega_pred, zeta_pred, batch)
                omega_losses.append(omega_pct)
                zeta_losses.append(zeta_pct)
                weighted_w_losses.append(float(l_w.detach().cpu()))
                weighted_z_losses.append(float(l_z.detach().cpu()))
                weighted_p_losses.append(float(l_p.detach().cpu()))
                frf_weighted_losses.append(float(loss_frf.detach().cpu()))

                scaler.scale(loss).backward()
                _apply_gradient_clip(net, optimizer, scaler, config, enabled=bool(args.fp16))
                scaler.step(optimizer)
                scaler.update()

            if scheduler is not None:
                scheduler.step()

            mean_loss = float(np.mean(losses)) if losses else 0.0
            omega_pct = float(np.mean(omega_losses)) if omega_losses else 0.0
            zeta_pct = float(np.mean(zeta_losses)) if zeta_losses else 0.0
            wgt_w = float(np.mean(weighted_w_losses)) if weighted_w_losses else 0.0
            wgt_z = float(np.mean(weighted_z_losses)) if weighted_z_losses else 0.0
            wgt_p = float(np.mean(weighted_p_losses)) if weighted_p_losses else 0.0
            wgt_frf = float(np.mean(frf_weighted_losses)) if frf_weighted_losses else 0.0
            phi_loss_value = wgt_p / max(current_phi_w, 1e-8)
            omega_share = _safe_share(wgt_w, mean_loss)
            zeta_share = _safe_share(wgt_z, mean_loss)
            phi_share = _safe_share(wgt_p, mean_loss)
            frf_share = _safe_share(wgt_frf, mean_loss)

            _log(
                f"Epoch {epoch:4d} [{phase_name}] | omega={omega_pct:.3f}% "
                f"zeta={zeta_pct:.3f}% phiLoss={phi_loss_value:.3e} | "
                f"share omega={omega_share:.1f}% zeta={zeta_share:.1f}% "
                f"phi={phi_share:.1f}% frf={frf_share:.1f}% | total={mean_loss:.3e}",
                logger,
            )

            if (not phase2_unlocked) and (omega_pct < unlock_omega_pct or epoch >= unlock_after_epoch):
                reason = f"omega<{unlock_omega_pct}%" if omega_pct < unlock_omega_pct else f"epoch>={unlock_after_epoch}"
                phase2_unlocked = True
                unlock_epoch = epoch
                _log(f">>> {reason}, unlock Phase2 FRF joint training <<<", logger)

            lr = optimizer.param_groups[0]["lr"]
            val_freq = int(config.get("validation_frequency", 5))
            should_validate = (epoch % val_freq == 0) or (epoch == total_epochs - 1)
            if should_validate:
                save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last", config=config, model_cfg=model_cfg)
                val_results = evaluate(args, config, net, valloader, logger, epoch, verbose=phase2_unlocked)
                val_loss = val_results.get("loss (MSE)", np.inf)
                omega_mae = val_results.get("omega_MAE_rad_s", np.inf)

                log_writer.writerow([
                    epoch, phase_name, f"{mean_loss:.6e}", f"{omega_pct:.4f}", f"{zeta_pct:.4f}",
                    f"{phi_loss_value:.6e}", f"{omega_share:.2f}", f"{zeta_share:.2f}",
                    f"{phi_share:.2f}", f"{frf_share:.2f}",
                    "" if np.isinf(val_loss) else f"{val_loss:.6e}",
                    f"{val_results.get('Amplitude MAE', 0):.6e}",
                    f"{val_results.get('Amplitude MAPE (%)', 0):.4f}",
                    "" if np.isinf(omega_mae) else f"{omega_mae:.6e}",
                    f"{lr:.6e}",
                ])
                log_file.flush()

                best_metric = omega_mae if not phase2_unlocked else val_loss
                metric_name = "omega_MAE_rad_s" if not phase2_unlocked else "val_loss"
                if best_metric < lowest:
                    _log(f"best model ({metric_name}={best_metric:.6g})", logger)
                    save_model(args.dir, epoch, net, optimizer, best_metric, config=config, model_cfg=model_cfg)
                    lowest = best_metric
            else:
                log_writer.writerow([
                    epoch, phase_name, f"{mean_loss:.6e}", f"{omega_pct:.4f}", f"{zeta_pct:.4f}",
                    f"{phi_loss_value:.6e}", f"{omega_share:.2f}", f"{zeta_share:.2f}",
                    f"{phi_share:.2f}", f"{frf_share:.2f}", "", "", "", "", f"{lr:.6e}",
                ])
                log_file.flush()
    finally:
        log_file.close()

    return net


def _require_tensor_frequencies(batch: Dict) -> torch.Tensor:
    frequencies = batch.get("frequencies")
    if not torch.is_tensor(frequencies):
        raise ValueError("Phase2 FRF training requires equal-length frequency grids so frequencies can be stacked as a tensor.")
    return frequencies


def _modal_relative_errors(omega_pred, zeta_pred, batch: Dict):
    omega_sorted, sort_idx = torch.sort(omega_pred.detach(), dim=-1)
    zeta_sorted = torch.gather(zeta_pred.detach(), dim=-1, index=sort_idx)
    omega_rel = torch.abs(omega_sorted - batch["modal_omega_norm"]) / (batch["modal_omega_norm"].abs() + 1e-8)
    zeta_rel = torch.abs(zeta_sorted - batch["modal_zeta"]) / (batch["modal_zeta"].abs() + 1e-8)
    return float(omega_rel.mean().cpu() * 100.0), float(zeta_rel.mean().cpu() * 100.0)


def _safe_share(component: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(component / total * 100.0)


def _apply_gradient_clip(net, optimizer, scaler, config, enabled: bool):
    grad_clip = config.get("optimizer", {}).get("gradient_clip")
    if grad_clip is None:
        return
    if enabled:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True):
    prediction, output, omega_errs = _generate_preds(args, config, net, dataloader)
    return _evaluate(prediction, output, omega_errs, logger, epoch, verbose)


def _generate_preds(args, config, net, dataloader):
    was_training = net.training
    net.eval()
    predictions, outputs, omega_errs = [], [], []
    omega_max = float(config.get("omega_max", 25000.0))

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_graph_batch(raw_batch, args.device)
            frequencies = _require_tensor_frequencies(batch)
            target = batch["point_frf"]
            frf_pred, omega_pred, _, _ = _forward_modal(
                net, batch, frequencies=frequencies,
                excitation_index_global=batch.get("excitation_index_global")
            )
            predictions.append(frf_pred.detach().cpu())
            outputs.append(target.detach().cpu())

            omega_true = batch.get("modal_omega_phys")
            if omega_true is not None:
                omega_pred_val, _ = torch.sort(omega_pred.detach().cpu(), dim=-1)
                omega_errs.append((omega_pred_val * omega_max - omega_true.detach().cpu()).abs())

    if was_training:
        net.train()
    return torch.cat(predictions, dim=0), torch.cat(outputs, dim=0), omega_errs


def _evaluate(prediction, output, omega_errs, logger, epoch=None, verbose=True):
    results = {}
    if prediction.shape != output.shape:
        output = output.reshape(prediction.shape)
    results["loss (MSE)"] = F.mse_loss(prediction, output).item()
    if prediction.ndim >= 3 and prediction.shape[-1] == 2:
        p_amp = torch.sqrt(prediction[..., 0] ** 2 + prediction[..., 1] ** 2 + 1e-12)
        o_amp = torch.sqrt(output[..., 0] ** 2 + output[..., 1] ** 2 + 1e-12)
        results["Amplitude MAE"] = F.l1_loss(p_amp, o_amp).item()
        results["Amplitude MAPE (%)"] = (torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0
    if omega_errs:
        results["omega_MAE_rad_s"] = torch.cat([e.flatten() for e in omega_errs]).mean().item()
    if verbose:
        prefix = f"[val epoch {epoch}] " if epoch is not None else ""
        for key, val in results.items():
            _log(f"{prefix}{key} = {val:.6g}", logger)
    return results


def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best", config=None, model_cfg=None):
    os.makedirs(savepath, exist_ok=True)
    if torch.is_tensor(loss):
        loss_value = float(loss.detach().cpu())
    else:
        loss_value = float(loss)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_value,
    }
    if config is not None:
        payload["config"] = config
    if model_cfg is not None:
        payload["model_cfg"] = model_cfg
    torch.save(payload, os.path.join(savepath, name))


def _log(msg, logger):
    if logger and hasattr(logger, "info"):
        logger.info(msg)
    else:
        print(msg)
