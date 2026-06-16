from __future__ import annotations

import csv
import os
import time
from typing import Dict

import numpy as np
import torch

from .losses import modal_loss


GRAPH_TENSOR_KEYS = [
    "node_features", "edge_index", "edge_attr", "batch",
    "points", "query_coords", "spring_k_xyz", "node_type",
    "pocket_bottom_mask", "cut_region_mask", "local_thickness_ratio",
    "pocket_depth_ratio", "node_weight", "excitation_index",
    "excitation_index_global", "excitation_coord", "modal_omega_phys",
    "modal_omega_norm", "modal_freq_hz", "modal_phi_z", "modal_phi",
    "modal_phi_xyz",
]


def _log(msg: str, logger=None):
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)


def _move_graph_batch(batch: Dict, device: str) -> Dict:
    out = dict(batch)
    for key in GRAPH_TENSOR_KEYS:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)
    return out


def _phase_weights(config: Dict, epoch: int):
    omega_pretrain_epochs = int(config.get("omega_pretrain_epochs", 30))
    if epoch < omega_pretrain_epochs:
        return {
            "phase": "Phase0 omega-only",
            "omega_weight": float(config.get("omega_loss_weight", 1.0)) * 5.0,
            "phi_weight": 0.0,
            "compute_phi": False,
        }
    return {
        "phase": "Phase1 omega+phi_z",
        "omega_weight": float(config.get("omega_loss_weight", 1.0)),
        "phi_weight": float(config.get("phi_loss_weight", 3.0)),
        "compute_phi": True,
    }


def _format_vec(x, precision=2):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return "/".join(f"{v:.{precision}f}" for v in arr)


def _save_model(out_dir, epoch, net, optimizer, metric, name, config=None, model_cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pt")
    torch.save({
        "epoch": epoch,
        "metric": float(metric),
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "model_cfg": model_cfg,
    }, path)


def train(args, config, model_cfg, net, dataloader, optimizer, valloader,
          scheduler=None, logger=None, start_epoch=0):
    os.makedirs(args.dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16))
    total_epochs = int(config.get("epochs", 300))
    validation_frequency = int(config.get("validation_frequency", 5))
    progress_interval = int(config.get("progress_interval", 10))
    best_score = np.inf

    log_path = os.path.join(args.dir, "loss_log.csv")
    log_file = open(log_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(log_file)
    if start_epoch == 0 or os.path.getsize(log_path) == 0:
        writer.writerow([
            "epoch", "phase", "train_loss", "omega_loss", "phi_loss",
            "w1%", "w2%", "w3%", "w_mean%",
            "MAC1", "MAC2", "MAC3", "phi_amp1%", "phi_amp2%", "phi_amp3%",
            "val_w1%", "val_w2%", "val_w3%", "val_w_mean%",
            "val_MAC1", "val_MAC2", "val_MAC3", "lr", "time_s",
        ])

    try:
        for epoch in range(start_epoch, total_epochs):
            phase = _phase_weights(config, epoch)
            net.train()
            t0 = time.time()
            losses, omega_losses, phi_losses = [], [], []
            freq_list, mac_list, phi_amp_list = [], [], []
            total_batches = len(dataloader)

            for step, raw_batch in enumerate(dataloader, start=1):
                optimizer.zero_grad(set_to_none=True)
                batch = _move_graph_batch(raw_batch, args.device)

                with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                    out = net(
                        batch["node_features"], batch["edge_index"], batch["edge_attr"], batch["batch"],
                        compute_phi=bool(phase["compute_phi"]),
                    )
                    loss, metrics = modal_loss(
                        out, batch,
                        omega_weight=float(phase["omega_weight"]),
                        phi_weight=float(phase["phi_weight"]),
                        mac_weight=float(config.get("mac_weight", 5.0)),
                        scale_weight=float(config.get("scale_weight", 1.0)),
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(config.get("gradient_clip", 2.0)))
                scaler.step(optimizer)
                scaler.update()

                losses.append(float(metrics["loss"].cpu()))
                omega_losses.append(float(metrics["loss_omega"].cpu()))
                phi_losses.append(float(metrics["loss_phi"].cpu()))
                freq_list.append(metrics["freq_percent"].detach().cpu().numpy())
                mac_list.append(metrics["mac"].detach().cpu().numpy())
                phi_amp_list.append(metrics["phi_amp_percent"].detach().cpu().numpy())

                if progress_interval > 0 and (step == 1 or step % progress_interval == 0 or step == total_batches):
                    elapsed = time.time() - t0
                    avg = elapsed / max(step, 1)
                    w_now = metrics["freq_percent"].detach().cpu().numpy()
                    mac_now = metrics["mac"].detach().cpu().numpy()
                    phi_amp_now = metrics["phi_amp_percent"].detach().cpu().numpy()
                    if phase["compute_phi"]:
                        metric_text = (
                            f"w=[{_format_vec(w_now, 2)}]% mean={np.mean(w_now):.2f}% | "
                            f"MAC=[{_format_vec(mac_now, 3)}] | "
                            f"phiA=[{_format_vec(phi_amp_now, 1)}]% | "
                            f"Lw={float(metrics['loss_omega'].cpu()):.4g} Lphi={float(metrics['loss_phi'].cpu()):.4g}"
                        )
                    else:
                        metric_text = (
                            f"w=[{_format_vec(w_now, 2)}]% mean={np.mean(w_now):.2f}% | "
                            f"Lw={float(metrics['loss_omega'].cpu()):.4g}"
                        )
                    _log(
                        f"Epoch {epoch:04d} | {phase['phase']} | batch {step}/{total_batches} | "
                        f"loss={losses[-1]:.4g} | {metric_text} | "
                        f"avg={avg:.2f}s/batch | elapsed={elapsed:.1f}s",
                        logger,
                    )

            tr_freq = np.mean(np.stack(freq_list), axis=0)
            tr_mac = np.mean(np.stack(mac_list), axis=0)
            tr_phi_amp = np.mean(np.stack(phi_amp_list), axis=0)
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            if phase["compute_phi"]:
                _log(
                    f"Epoch {epoch:4d} | {phase['phase']} | "
                    f"w=[{_format_vec(tr_freq, 2)}]% mean={tr_freq.mean():.2f}% | "
                    f"MAC=[{_format_vec(tr_mac, 3)}] | "
                    f"phiA=[{_format_vec(tr_phi_amp, 1)}]% | "
                    f"loss={np.mean(losses):.4g} time={dt:.1f}s lr={lr:.2e}",
                    logger,
                )
            else:
                _log(
                    f"Epoch {epoch:4d} | {phase['phase']} | "
                    f"w=[{_format_vec(tr_freq, 2)}]% mean={tr_freq.mean():.2f}% | "
                    f"loss={np.mean(losses):.4g} time={dt:.1f}s lr={lr:.2e}",
                    logger,
                )

            val = {}
            if (epoch % validation_frequency == 0) or (epoch == total_epochs - 1):
                _save_model(args.dir, epoch, net, optimizer, np.mean(losses), "checkpoint_last", config, model_cfg)
                val = evaluate(args, config, net, valloader, logger=logger, epoch=epoch,
                               compute_phi=bool(phase["compute_phi"]), verbose=False)
                if phase["compute_phi"]:
                    _log(
                        f"Val modal | w=[{_format_vec(val['val_w'], 2)}]% mean={val['val_w_mean']:.2f}% | "
                        f"MAC=[{_format_vec(val['val_mac'], 3)}] | phiA=[{_format_vec(val['val_phi_amp'], 1)}]% | "
                        f"score={val['val_score']:.4g}",
                        logger,
                    )
                else:
                    _log(
                        f"Val omega | w=[{_format_vec(val['val_w'], 2)}]% mean={val['val_w_mean']:.2f}% | "
                        f"score={val['val_score']:.4g}",
                        logger,
                    )
                if val["val_score"] < best_score:
                    best_score = val["val_score"]
                    _save_model(args.dir, epoch, net, optimizer, best_score, "checkpoint_best", config, model_cfg)
                if scheduler is not None:
                    try:
                        scheduler.step(val["val_score"])
                    except TypeError:
                        scheduler.step()

            writer.writerow([
                epoch, phase["phase"], f"{np.mean(losses):.6e}",
                f"{np.mean(omega_losses):.6e}", f"{np.mean(phi_losses):.6e}",
                *[f"{x:.4f}" for x in tr_freq], f"{tr_freq.mean():.4f}",
                *[f"{x:.6f}" for x in tr_mac], *[f"{x:.4f}" for x in tr_phi_amp],
                *[f"{x:.4f}" for x in val.get("val_w", np.zeros(3))],
                f"{val.get('val_w_mean', 0.0):.4f}",
                *[f"{x:.6f}" for x in val.get("val_mac", np.zeros(3))],
                f"{lr:.6e}", f"{dt:.2f}",
            ])
            log_file.flush()
    finally:
        log_file.close()
    return net


@torch.no_grad()
def evaluate(args, config, net, dataloader, logger=None, epoch=None, compute_phi=True, verbose=True):
    was_training = net.training
    net.eval()
    freq_list, mac_list, phi_amp_list, loss_list = [], [], [], []
    for raw_batch in dataloader:
        batch = _move_graph_batch(raw_batch, args.device)
        with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
            out = net(batch["node_features"], batch["edge_index"], batch["edge_attr"], batch["batch"], compute_phi=compute_phi)
            loss, metrics = modal_loss(
                out, batch,
                omega_weight=1.0,
                phi_weight=float(config.get("phi_loss_weight", 3.0)) if compute_phi else 0.0,
                mac_weight=float(config.get("mac_weight", 5.0)),
                scale_weight=float(config.get("scale_weight", 1.0)),
            )
        loss_list.append(float(loss.cpu()))
        freq_list.append(metrics["freq_percent"].detach().cpu().numpy())
        mac_list.append(metrics["mac"].detach().cpu().numpy())
        phi_amp_list.append(metrics["phi_amp_percent"].detach().cpu().numpy())

    val_w = np.mean(np.stack(freq_list), axis=0)
    val_mac = np.mean(np.stack(mac_list), axis=0)
    val_phi_amp = np.mean(np.stack(phi_amp_list), axis=0)
    if compute_phi:
        val_score = float(val_w.mean() + (1.0 - val_mac.mean()) * 100.0 + 0.05 * val_phi_amp.mean())
    else:
        val_score = float(val_w.mean())

    if was_training:
        net.train()
    res = {
        "loss": float(np.mean(loss_list)),
        "val_w": val_w,
        "val_w_mean": float(val_w.mean()),
        "val_mac": val_mac,
        "val_phi_amp": val_phi_amp,
        "val_score": val_score,
    }
    if verbose:
        _log(
            f"Eval | w=[{_format_vec(val_w, 2)}]% mean={val_w.mean():.2f}% | "
            f"MAC=[{_format_vec(val_mac, 3)}] | phiA=[{_format_vec(val_phi_amp, 1)}]% | score={val_score:.4g}",
            logger,
        )
    return res


train_modal = train
evaluate_modal = evaluate
