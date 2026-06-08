"""
trainer.py — MeshGraphNet/GNN 两阶段训练循环 + 评估。

训练策略:
    Phase 1: 纯模态参数/振型监督
    Phase 2: 模态监督 + PhysicsDecoder FRF 联合监督

数据流:
    node_features + edge_index + edge_attr + batch + frequencies
        → MeshGraphFRFModel
        → omega, zeta, phi
        → PhysicsDecoder
        → point_frf
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .losses import modal_loss, frf_loss


def _move_graph_batch(batch: Dict, device: str) -> Dict:
    out = dict(batch)
    tensor_keys = [
        "node_features", "edge_index", "edge_attr", "batch", "points", "query_coords",
        "point_features", "modal_omega_norm", "modal_omega_phys", "modal_zeta",
        "modal_phi", "modal_phi_exc", "point_frf", "modal_phi_xyz",
    ]
    for key in tensor_keys:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(device)
    if torch.is_tensor(out.get("frequencies")):
        out["frequencies"] = out["frequencies"].to(device)
    return out


def _forward_modal(net, batch: Dict, frequencies=None, phi_exc=None, alpha: float = 1.0):
    return net(
        batch["node_features"],
        batch["edge_index"],
        batch["edge_attr"],
        batch["batch"],
        frequencies=frequencies,
        phi_exc=phi_exc,
        alpha=alpha,
    )


def _align_phi_exc(net, batch: Dict, frequencies=None):
    phi_exc = batch.get("modal_phi_exc")
    if phi_exc is None:
        return None
    with torch.no_grad():
        _, _, _, phi_scan = _forward_modal(net, batch, frequencies=frequencies, phi_exc=None)
    modal_phi = batch["modal_phi"]
    phi_exc_c = phi_exc.clone()
    batch_idx = batch["batch"]
    for i in range(int(batch_idx.max().item()) + 1):
        mask = batch_idx == i
        dot = torch.sum(phi_scan[mask] * modal_phi[mask], dim=0)
        phi_exc_c[i] = phi_exc[i] * torch.sign(dot + 1e-8)
    return phi_exc_c


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler, logger=None, start_epoch=0):
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

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
            "轮次", "训练损失", "ω%", "ζ%", "φMSE", "w占比%", "z占比%",
            "phi占比%", "FRF占比%", "验证MSE", "幅值MAE", "幅值MAPE%", "学习率"
        ])

    phase2_unlocked = False
    unlock_epoch = start_epoch

    try:
        for epoch in range(start_epoch, total_epochs):
            losses, omega_losses, zeta_losses = [], [], []
            weighted_w_losses, weighted_z_losses, weighted_p_losses = [], [], []
            in_phase2 = phase2_unlocked
            in_phase1 = not phase2_unlocked

            if in_phase1 and epoch == 0:
                _log("=== Phase1: MeshGraphNet 纯模态训练 ===", logger)
            elif in_phase2 and not getattr(net, "_phase2_logged", False):
                _log(f"=== Phase2: 模态 + FRF 联合训练, unlock epoch={epoch} ===", logger)
                net._phase2_logged = True
                lowest = np.inf

            for raw_batch in dataloader:
                optimizer.zero_grad()
                batch = _move_graph_batch(raw_batch, args.device)
                current_phi_w = config.get("phi_loss_weight", 100.0) if in_phase2 else config.get("phase1_phi_loss_weight", 1.0)

                with torch.cuda.amp.autocast(enabled=args.fp16):
                    if in_phase2:
                        phase2_epoch = epoch - unlock_epoch
                        alpha = max(1.0, 10.0 - 9.0 * phase2_epoch / 200.0)
                        frequencies = batch["frequencies"]
                        phi_exc = _align_phi_exc(net, batch, frequencies=frequencies)
                        frf_pred, omega_pred, zeta_pred, phi_pred = _forward_modal(
                            net, batch, frequencies=frequencies, phi_exc=phi_exc, alpha=alpha
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
                        current_frf_w = frf_weight * min(1.0, phase2_epoch / 20.0)
                        loss = loss_m + current_frf_w * raw_frf
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
                        loss = loss_m

                losses.append(loss.detach().cpu().item())
                omega_pred_sorted, sort_idx_log = torch.sort(omega_pred, dim=-1)
                zeta_pred_sorted = torch.gather(zeta_pred, dim=-1, index=sort_idx_log)
                omega_rel_err = torch.abs(omega_pred_sorted - batch["modal_omega_norm"]) / (batch["modal_omega_norm"] + 1e-8)
                zeta_rel_err = torch.abs(zeta_pred_sorted - batch["modal_zeta"]) / (batch["modal_zeta"] + 1e-8)
                omega_losses.append(omega_rel_err.mean().detach().cpu().item())
                zeta_losses.append(zeta_rel_err.mean().detach().cpu().item())
                weighted_w_losses.append(l_w.detach().cpu().item())
                weighted_z_losses.append(l_z.detach().cpu().item())
                weighted_p_losses.append(l_p.detach().cpu().item())

                scaler.scale(loss).backward()
                _apply_gradient_clip(net, config)
                scaler.step(optimizer)
                scaler.update()

            if scheduler is not None:
                scheduler.step()

            mean_loss = float(np.mean(losses)) if losses else 0.0
            omega_pct = float(np.mean(omega_losses) * 100.0) if omega_losses else 0.0
            zeta_pct = float(np.mean(zeta_losses) * 100.0) if zeta_losses else 0.0
            wgt_w = float(np.mean(weighted_w_losses)) if weighted_w_losses else 0.0
            wgt_z = float(np.mean(weighted_z_losses)) if weighted_z_losses else 0.0
            wgt_p = float(np.mean(weighted_p_losses)) if weighted_p_losses else 0.0
            phi_div = config.get("phi_loss_weight", 100.0) if in_phase2 else config.get("phase1_phi_loss_weight", 1.0)
            phi_mse = wgt_p / max(phi_div, 1e-8)
            omega_share = wgt_w / mean_loss * 100.0 if mean_loss > 0 else 0.0
            zeta_share = wgt_z / mean_loss * 100.0 if mean_loss > 0 else 0.0
            phi_share = wgt_p / mean_loss * 100.0 if mean_loss > 0 else 0.0
            frf_share = 0.0 if in_phase1 else max(0.0, 100.0 - omega_share - zeta_share - phi_share)

            _log(
                f"Epoch {epoch:4d} | w={omega_pct:.2f}% z={zeta_pct:.2f}% "
                f"phiMSE={phi_mse:.3e} | share w{omega_share:.0f}% z{zeta_share:.0f}% "
                f"phi{phi_share:.0f}% frf{frf_share:.0f}% | total={mean_loss:.2e}",
                logger,
            )

            if not phase2_unlocked and (omega_pct < unlock_omega_pct or epoch >= unlock_after_epoch):
                reason = f"ω<{unlock_omega_pct}%" if omega_pct < unlock_omega_pct else f"epoch>={unlock_after_epoch}"
                phase2_unlocked = True
                unlock_epoch = epoch
                _log(f">>> {reason}, 解锁 Phase2 FRF 联合训练 <<<", logger)

            lr = optimizer.param_groups[0]["lr"]
            val_freq = int(config.get("validation_frequency", 5))
            should_validate = (epoch % val_freq == 0) or (epoch == total_epochs - 1)
            if should_validate:
                save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last")
                val_results = evaluate(args, config, net, valloader, logger, epoch, verbose=not in_phase1)
                val_loss = val_results.get("loss (MSE)", np.inf)
                omega_mae = val_results.get("ω_MAE (rad/s)", np.inf)

                log_writer.writerow([
                    epoch, f"{mean_loss:.2e}", f"{omega_pct:.3f}", f"{zeta_pct:.3f}",
                    f"{phi_mse:.3e}", f"{omega_share:.1f}", f"{zeta_share:.1f}",
                    f"{phi_share:.1f}", f"{frf_share:.1f}",
                    "" if np.isinf(val_loss) else f"{val_loss:.6f}",
                    f"{val_results.get('Amplitude MAE', 0):.4f}",
                    f"{val_results.get('Amplitude MAPE (%)', 0):.2f}", f"{lr:.2e}",
                ])
                log_file.flush()

                best_metric = omega_mae if in_phase1 else val_loss
                metric_name = "ω_MAE" if in_phase1 else "val_loss"
                if best_metric < lowest:
                    _log(f"best model ({metric_name}={best_metric:.6g})", logger)
                    save_model(args.dir, epoch, net, optimizer, best_metric)
                    lowest = best_metric
            else:
                log_writer.writerow([
                    epoch, f"{mean_loss:.2e}", f"{omega_pct:.3f}", f"{zeta_pct:.3f}",
                    f"{phi_mse:.3e}", f"{omega_share:.1f}", f"{zeta_share:.1f}",
                    f"{phi_share:.1f}", f"{frf_share:.1f}", "", "", "", f"{lr:.2e}",
                ])
                log_file.flush()
    finally:
        log_file.close()

    return net


def _apply_gradient_clip(net, config):
    grad_clip = config.get("optimizer", {}).get("gradient_clip")
    if grad_clip is None:
        return
    torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True):
    prediction, output, omega_errs = _generate_preds(args, config, net, dataloader)
    return _evaluate(prediction, output, omega_errs, logger, epoch, verbose)


def _generate_preds(args, config, net, dataloader):
    net.eval()
    predictions, outputs, omega_errs = [], [], []
    omega_max = float(config.get("omega_max", 25000.0))

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_graph_batch(raw_batch, args.device)
            frequencies = batch["frequencies"]
            target = batch["point_frf"]
            phi_exc = _align_phi_exc(net, batch, frequencies=frequencies)
            frf_pred, omega_pred, _, _ = _forward_modal(net, batch, frequencies=frequencies, phi_exc=phi_exc)
            predictions.append(frf_pred.detach().cpu())
            outputs.append(target.detach().cpu())

            omega_true = batch.get("modal_omega_phys")
            if omega_true is not None:
                omega_pred_val, _ = torch.sort(omega_pred.detach().cpu(), dim=-1)
                omega_errs.append((omega_pred_val * omega_max - omega_true.detach().cpu()).abs())

    net.train()
    return torch.cat(predictions, dim=0), torch.cat(outputs, dim=0), omega_errs


def _evaluate(prediction, output, omega_errs, logger, epoch, verbose=True):
    results = {}
    if prediction.shape != output.shape:
        output = output.reshape(prediction.shape)
    results["loss (MSE)"] = F.mse_loss(prediction, output).item()
    if prediction.ndim >= 3 and prediction.shape[-1] == 2:
        p_amp = torch.sqrt(prediction[..., 0] ** 2 + prediction[..., 1] ** 2 + 1e-8)
        o_amp = torch.sqrt(output[..., 0] ** 2 + output[..., 1] ** 2 + 1e-8)
        results["Amplitude MAE"] = F.l1_loss(p_amp, o_amp).item()
        results["Amplitude MAPE (%)"] = (torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0
    if omega_errs:
        results["ω_MAE (rad/s)"] = torch.cat([e.flatten() for e in omega_errs]).mean().item()
    if verbose:
        for key, val in results.items():
            _log(f"{key} = {val:.6g}", logger)
    return results


def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best"):
    os.makedirs(savepath, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, os.path.join(savepath, name))


def _log(msg, logger):
    if logger and hasattr(logger, "info"):
        logger.info(msg)
    else:
        print(msg)
