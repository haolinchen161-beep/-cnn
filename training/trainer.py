"""
trainer.py — FEM-aware MeshGraphNet training loop.

Graph batch:
    node_features + edge_index + edge_attr + batch
        -> MeshGraphFRFModel
        -> omega(rad/s), zeta, phi[N,K,3]
        -> PhysicsDecoder -> FRF

Main phases:
    Phase0a: first omega_prior_only_epochs, train only omega prior MLP.
    Phase0b: until omega_pretrain_epochs, frequency only.
    Phase1: modal joint training.
    Phase2a: freeze phi/graph representation, tune omega/zeta with weak FRF using true phi_exc.
    Phase2b: weak end-to-end FRF fine-tuning using predicted phi_exc at excitation_index.

The validation FRF metric is self-phi-exc by default.  Teacher-phi-exc metrics are
also reported for diagnosis but are not used as the primary score.
"""
from __future__ import annotations

import csv
import os
from typing import Dict

import numpy as np
import torch

from .losses import modal_loss, modal_loss_z_only, frf_loss, branch_loss


GRAPH_TENSOR_KEYS = [
    "node_features", "edge_index", "edge_attr", "batch",
    "points", "query_coords", "point_features",
    "spring_k_xyz", "spring_c_xyz", "node_type",
    "pocket_bottom_mask", "cut_region_mask",
    "excitation_index", "excitation_index_global", "excitation_coord",
    "modal_omega_norm", "modal_omega_phys", "modal_zeta",
    "modal_phi", "modal_phi_exc", "modal_phi_xyz",
    "point_frf", "force_vector",
]


def _move_graph_batch(batch: Dict, device: str) -> Dict:
    out = dict(batch)
    for key in GRAPH_TENSOR_KEYS:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)
    if torch.is_tensor(out.get("frequencies")):
        out["frequencies"] = out["frequencies"].to(device, non_blocking=True)
    return out


def _forward_modal(net, batch: Dict, frequencies=None, phi_exc=None,
                   omega_true=None, detach_modal_for_frf=True, alpha: float = 1.0):
    return net(
        batch["node_features"],
        batch["edge_index"],
        batch["edge_attr"],
        batch["batch"],
        frequencies=frequencies,
        phi_exc=phi_exc,
        excitation_index_global=batch.get("excitation_index_global"),
        force_vector=batch.get("force_vector"),
        alpha=alpha,
        omega_true=omega_true,
        detach_modal_for_frf=detach_modal_for_frf,
    )


def _set_all_trainable(net):
    net.train()
    for p in net.parameters():
        p.requires_grad = True


def _set_omega_prior_only(net):
    for p in net.parameters():
        p.requires_grad = False
    net.eval()

    if hasattr(net, "omega_head"):
        net.omega_head.train()
        if hasattr(net.omega_head, "delta_mlp"):
            net.omega_head.delta_mlp.eval()
            for p in net.omega_head.delta_mlp.parameters():
                p.requires_grad = False
        if hasattr(net.omega_head, "prior_mlp"):
            net.omega_head.prior_mlp.train()
            for p in net.omega_head.prior_mlp.parameters():
                p.requires_grad = True


def _set_phase2a_omega_tune(net):
    for p in net.parameters():
        p.requires_grad = False

    for name in [
        "node_encoder", "edge_encoder", "blocks", "global_proj",
        "phi_decoder", "phi_scale_head", "branch_head",
    ]:
        if hasattr(net, name):
            getattr(net, name).eval()

    for name in ["omega_head", "zeta_head"]:
        if hasattr(net, name):
            module = getattr(net, name)
            module.train()
            for p in module.parameters():
                p.requires_grad = True


def _freeze_zeta_head_if_requested(net, disable_zeta_training: bool):
    if not disable_zeta_training:
        return
    if hasattr(net, "zeta_head"):
        net.zeta_head.eval()
        for p in net.zeta_head.parameters():
            p.requires_grad = False


def _phi_loss_mode(config: Dict) -> str:
    mode = str(config.get("phi_loss_mode", "xyz")).lower()
    if mode in {"z", "z_only", "phi_z", "zonly"}:
        return "z_only"
    return "xyz"


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler=None, logger=None, start_epoch=0):
    lowest = np.inf
    lowest_modal = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16))

    total_epochs = int(config.get("epochs", 300))
    frf_weight = float(config.get("frf_loss_weight", 0.005))
    phase2_min_epoch = int(config.get("phase2_min_epoch", 160))
    validation_frequency = int(config.get("validation_frequency", 5))
    phi_mode = _phi_loss_mode(config)
    loss_fn = modal_loss_z_only if phi_mode == "z_only" else modal_loss
    disable_zeta_training = bool(config.get("disable_zeta_training", False))

    if phi_mode == "z_only":
        _log("=== Z-only phi supervision enabled: loss/metrics use phi[...,2] only ===", logger)
    if disable_zeta_training or float(config.get("zeta_loss_weight", 10.0)) <= 0.0:
        _log("=== Zeta supervision disabled: zeta loss/score ignored ===", logger)

    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "loss_log.csv")
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, "a", newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_file)
    if not log_exists:
        val_header = [
            "val_w1%", "val_w2%", "val_w3%",
            "val_z1%", "val_z2%", "val_z3%",
            "val_MAC1", "val_MAC2", "val_MAC3",
            "val_phiN1%", "val_phiN2%", "val_phiN3%",
            "val_phiA1%", "val_phiA2%", "val_phiA3%",
            "val_dir2%", "val_dir3%", "val_modal_score",
        ]
        log_writer.writerow([
            "epoch", "train_loss",
            "w1%", "w2%", "w3%",
            "z1%", "z2%", "z3%",
            "phi_loss", "phiN1%", "phiN2%", "phiN3%",
            "phiA1%", "phiA2%", "phiA3%",
            "MAC1", "MAC2", "MAC3",
            "w_share%", "z_share%", "phi_share%", "FRF_share%",
            "val_self_MSE", "self_amp_MAE", "self_amp_MAPE%",
            "val_teacher_MSE", "teacher_amp_MAE", "teacher_amp_MAPE%",
            "kl", "dir2%", "dir3%", "lr",
        ] + val_header)

    phase2_unlocked = start_epoch >= phase2_min_epoch
    unlock_epoch = phase2_min_epoch if phase2_unlocked else start_epoch

    try:
        for epoch in range(start_epoch, total_epochs):
            losses = []
            weighted_w_losses, weighted_z_losses, weighted_p_losses, weighted_frf_losses = [], [], [], []
            kl_losses = []
            train_w, train_z, train_mac, train_phi_n, train_phi_a, train_dir = [], [], [], [], [], []

            in_phase1 = not phase2_unlocked
            in_phase2 = phase2_unlocked

            if in_phase1 and epoch == 0:
                _log("=== Phase1: graph modal training; FRF disabled before phase2 ===", logger)
            if in_phase1 and epoch >= phase2_min_epoch:
                phase2_unlocked = True
                in_phase1 = False
                in_phase2 = True
                unlock_epoch = epoch
                lowest = np.inf
                _log(f"=== Phase2 unlocked at epoch {epoch}: modal + weak FRF ===", logger)

            phase2_epoch = epoch - unlock_epoch if in_phase2 else -1
            phase2_omega_tune_epochs = int(config.get("phase2_omega_tune_epochs", 40))
            in_phase2a = bool(in_phase2 and phase2_epoch < phase2_omega_tune_epochs)

            omega_prior_only_epochs = int(config.get("omega_prior_only_epochs", 20))
            omega_pretrain_epochs = int(config.get("omega_pretrain_epochs", 40))
            in_omega_prior_only = in_phase1 and epoch < omega_prior_only_epochs
            in_phase0 = epoch < omega_pretrain_epochs

            if in_omega_prior_only:
                if not getattr(net, "_omega_prior_only_logged", False):
                    _log(f"=== Phase0a: train omega_head.prior_mlp only, epochs < {omega_prior_only_epochs} ===", logger)
                    net._omega_prior_only_logged = True
                _set_omega_prior_only(net)
                net._omega_prior_only_active = True
            elif in_phase1 and getattr(net, "_omega_prior_only_active", False):
                _log("=== Phase0b: unlock graph encoder + omega delta, still frequency-only ===", logger)
                _set_all_trainable(net)
                net._omega_prior_only_active = False

            if in_phase2a:
                if not getattr(net, "_phase2a_logged", False):
                    _log("=== Phase2a: freeze phi-related modules; tune omega/zeta with teacher phi_exc FRF ===", logger)
                    net._phase2a_logged = True
                _set_phase2a_omega_tune(net)
                phase2a_lr = float(config.get("phase2a_lr", 1e-4))
                for pg in optimizer.param_groups:
                    pg["lr"] = min(pg["lr"], phase2a_lr)

            if in_phase2 and (not in_phase2a) and not getattr(net, "_phase2b_logged", False):
                _log("=== Phase2b: unfreeze all; weak FRF uses predicted phi_exc at excitation_index ===", logger)
                _set_all_trainable(net)
                net._phase2b_logged = True

            _freeze_zeta_head_if_requested(net, disable_zeta_training)

            if in_phase0:
                current_phi_w = 0.0
                current_zeta_w = 0.0
                current_omega_w = float(config.get("omega_loss_weight", 1.0)) * 5.0
                kl_weight = 0.0
            else:
                current_phi_w = float(config.get("phi_loss_weight", 3.0))
                current_zeta_w = 0.0 if disable_zeta_training else float(config.get("zeta_loss_weight", 10.0))
                current_omega_w = float(config.get("omega_loss_weight", 1.0))
                kl_weight = 0.0 if phi_mode == "z_only" else float(config.get("branch_loss_weight", 20.0))

            if in_phase2a:
                current_phi_w = 0.0
                kl_weight = 0.0

            for raw_batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                batch = _move_graph_batch(raw_batch, args.device)

                with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                    if in_phase2:
                        frequencies = _require_tensor_frequencies(batch)
                        teacher_epochs = int(config.get("frf_teacher_epochs", 0))
                        omega_true = batch["modal_omega_phys"] if phase2_epoch < teacher_epochs else None
                        # Phase2a uses true excitation modal values to stabilize omega/zeta tuning.
                        # Phase2b switches to predicted excitation phi for true end-to-end FRF training.
                        frf_phi_exc = batch.get("modal_phi_exc") if in_phase2a else None
                        frf_pred, omega_pred, log_zeta_pred, zeta_pred, phi_pred = _forward_modal(
                            net, batch,
                            frequencies=frequencies,
                            phi_exc=frf_phi_exc,
                            omega_true=omega_true if not in_phase2a else None,
                            detach_modal_for_frf=not in_phase2a,
                            alpha=1.0,
                        )
                    else:
                        frf_pred, omega_pred, log_zeta_pred, zeta_pred, phi_pred = _forward_modal(net, batch)

                    loss_m, l_w, l_z, l_p, mac_val = loss_fn(
                        omega_pred, batch["modal_omega_phys"],
                        log_zeta_pred, batch["modal_zeta"],
                        phi_pred, batch["modal_phi"],
                        batch_idx=batch["batch"],
                        omega_weight=current_omega_w,
                        zeta_weight=current_zeta_w,
                        phi_weight=current_phi_w,
                    )

                    if kl_weight > 0 and hasattr(net, "branch_log_probs"):
                        loss_kl = branch_loss(net.branch_log_probs, batch["modal_phi"], batch["batch"]) * kl_weight
                    else:
                        loss_kl = loss_m.new_tensor(0.0)

                    if in_phase2:
                        raw_frf = frf_loss(frf_pred, batch["point_frf"])
                        warm = max(0.0, min(1.0, phase2_epoch / max(1, int(config.get("frf_warmup_epochs", 20)))))
                        loss_frf = frf_weight * warm * raw_frf
                    else:
                        loss_frf = loss_m.new_tensor(0.0)

                    loss = loss_m + loss_kl + loss_frf

                scaler.scale(loss).backward()
                _apply_gradient_clip(net, optimizer, scaler, config, enabled=bool(args.fp16))
                scaler.step(optimizer)
                scaler.update()

                losses.append(float(loss.detach().cpu()))
                weighted_w_losses.append(float(l_w.detach().cpu()))
                weighted_z_losses.append(float(l_z.detach().cpu()))
                weighted_p_losses.append(float(l_p.detach().cpu()))
                weighted_frf_losses.append(float(loss_frf.detach().cpu()))
                kl_losses.append(float(loss_kl.detach().cpu()))

                with torch.no_grad():
                    metrics = _compute_modal_metrics(
                        omega_pred.detach(), batch["modal_omega_phys"],
                        zeta_pred.detach(), batch["modal_zeta"],
                        phi_pred.detach(), batch["modal_phi"], batch["batch"],
                        phi_mode=phi_mode,
                    )
                    train_w.append(metrics["w"])
                    train_z.append(metrics["z"])
                    train_mac.append(metrics["mac"])
                    train_phi_n.append(metrics["phi_n"])
                    train_phi_a.append(metrics["phi_a"])
                    train_dir.append(metrics["dir"])

            mean_loss = float(np.mean(losses)) if losses else 0.0
            wgt_w = float(np.mean(weighted_w_losses)) if weighted_w_losses else 0.0
            wgt_z = float(np.mean(weighted_z_losses)) if weighted_z_losses else 0.0
            wgt_p = float(np.mean(weighted_p_losses)) if weighted_p_losses else 0.0
            wgt_frf = float(np.mean(weighted_frf_losses)) if weighted_frf_losses else 0.0
            mean_kl = float(np.mean(kl_losses)) if kl_losses else 0.0

            tr_w = np.mean(np.stack(train_w), axis=0)
            tr_z = np.mean(np.stack(train_z), axis=0)
            tr_mac = np.mean(np.stack(train_mac), axis=0)
            tr_phi_n = np.mean(np.stack(train_phi_n), axis=0)
            tr_phi_a = np.mean(np.stack(train_phi_a), axis=0)
            tr_dir = np.mean(np.stack(train_dir), axis=0)

            lr = optimizer.param_groups[0]["lr"]
            _log(
                f"Epoch {epoch:4d} | w=[{tr_w[0]:.1f}/{tr_w[1]:.1f}/{tr_w[2]:.1f}]% "
                f"z=[{tr_z[0]:.0f}/{tr_z[1]:.0f}/{tr_z[2]:.0f}]% "
                f"phiN=[{tr_phi_n[0]:.1f}/{tr_phi_n[1]:.1f}/{tr_phi_n[2]:.1f}]% "
                f"MAC=[{tr_mac[0]:.3f}/{tr_mac[1]:.3f}/{tr_mac[2]:.3f}] | "
                f"kl={mean_kl:.3f} dir2={tr_dir[1]:.0f}% dir3={tr_dir[2]:.0f}% | loss={mean_loss:.1f}",
                logger,
            )

            should_validate = (epoch % validation_frequency == 0) or (epoch == total_epochs - 1)
            val_results = {}
            val_modal_score = np.inf
            if should_validate:
                save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last", config=config, model_cfg=model_cfg)
                val_results = evaluate(args, config, net, valloader, logger, epoch, verbose=False)
                val_modal_score = val_results.get("val_modal_score", np.inf)
                val_loss = val_results.get("loss (MSE)", np.inf)

                _log(
                    f"Val modal | w=[{val_results['val_w'][0]:.3f}/{val_results['val_w'][1]:.3f}/{val_results['val_w'][2]:.3f}]% "
                    f"z=[{val_results['val_z'][0]:.1f}/{val_results['val_z'][1]:.1f}/{val_results['val_z'][2]:.1f}]% "
                    f"MAC=[{val_results['val_mac'][0]:.3f}/{val_results['val_mac'][1]:.3f}/{val_results['val_mac'][2]:.3f}] "
                    f"phiN=[{val_results['val_phi_n'][0]:.1f}/{val_results['val_phi_n'][1]:.1f}/{val_results['val_phi_n'][2]:.1f}]% "
                    f"dir2={val_results['val_dir'][1]:.0f}% dir3={val_results['val_dir'][2]:.0f}% | "
                    f"FRF_self={val_results.get('loss (MSE)', np.inf):.4g} teacher={val_results.get('Teacher loss (MSE)', np.inf):.4g}",
                    logger,
                )

                if val_modal_score < lowest_modal:
                    _log(f"best modal model (score={val_modal_score:.6g})", logger)
                    save_model(args.dir, epoch, net, optimizer, val_modal_score, "checkpoint_best_modal",
                               config=config, model_cfg=model_cfg)
                    lowest_modal = val_modal_score

                best_metric = val_modal_score if not in_phase2 else val_loss
                if best_metric < lowest:
                    _log(f"best model (metric={best_metric:.6g})", logger)
                    save_model(args.dir, epoch, net, optimizer, best_metric, "checkpoint_best",
                               config=config, model_cfg=model_cfg)
                    lowest = best_metric

                if scheduler is not None:
                    try:
                        scheduler.step(val_modal_score)
                    except TypeError:
                        scheduler.step()
            elif scheduler is not None:
                if scheduler.__class__.__name__ != "ReduceLROnPlateau":
                    scheduler.step()

            log_writer.writerow([
                epoch, f"{mean_loss:.6e}",
                f"{tr_w[0]:.4f}", f"{tr_w[1]:.4f}", f"{tr_w[2]:.4f}",
                f"{tr_z[0]:.4f}", f"{tr_z[1]:.4f}", f"{tr_z[2]:.4f}",
                f"{wgt_p / max(current_phi_w, 1e-8):.6e}",
                f"{tr_phi_n[0]:.4f}", f"{tr_phi_n[1]:.4f}", f"{tr_phi_n[2]:.4f}",
                f"{tr_phi_a[0]:.4f}", f"{tr_phi_a[1]:.4f}", f"{tr_phi_a[2]:.4f}",
                f"{tr_mac[0]:.6f}", f"{tr_mac[1]:.6f}", f"{tr_mac[2]:.6f}",
                f"{_safe_share(wgt_w, mean_loss):.2f}", f"{_safe_share(wgt_z, mean_loss):.2f}",
                f"{_safe_share(wgt_p, mean_loss):.2f}", f"{_safe_share(wgt_frf, mean_loss):.2f}",
                f"{val_results.get('loss (MSE)', '')}", f"{val_results.get('Amplitude MAE', '')}",
                f"{val_results.get('Amplitude MAPE (%)', '')}",
                f"{val_results.get('Teacher loss (MSE)', '')}", f"{val_results.get('Teacher Amplitude MAE', '')}",
                f"{val_results.get('Teacher Amplitude MAPE (%)', '')}",
                f"{mean_kl:.6e}", f"{tr_dir[1]:.2f}", f"{tr_dir[2]:.2f}", f"{lr:.6e}",
            ] + _format_val_results(val_results))
            log_file.flush()
    finally:
        log_file.close()

    return net


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True):
    net_was_training = net.training
    net.eval()
    modal_metrics = []
    frf_mse, amp_mae, amp_mape = [], [], []
    teacher_frf_mse, teacher_amp_mae, teacher_amp_mape = [], [], []
    phi_mode = _phi_loss_mode(config)
    evaluate_frf = bool(config.get("evaluate_frf", True))

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_graph_batch(raw_batch, args.device)

            if evaluate_frf and torch.is_tensor(batch.get("frequencies")) and "point_frf" in batch:
                # Primary validation: self excitation, i.e. use predicted phi at excitation_index.
                frf_pred, omega_pred, log_zeta_pred, zeta_pred, phi_pred = _forward_modal(
                    net, batch,
                    frequencies=batch["frequencies"],
                    phi_exc=None,
                    detach_modal_for_frf=True,
                )
                target = batch["point_frf"]
                mse, mae, mape = _compute_frf_metrics(frf_pred, target)
                frf_mse.append(mse)
                amp_mae.append(mae)
                amp_mape.append(mape)

                # Diagnostic only: teacher phi_exc shows how much error comes from excitation-point phi.
                if "modal_phi_exc" in batch:
                    frf_teacher, _, _, _, _ = _forward_modal(
                        net, batch,
                        frequencies=batch["frequencies"],
                        phi_exc=batch.get("modal_phi_exc"),
                        detach_modal_for_frf=True,
                    )
                    tmse, tmae, tmape = _compute_frf_metrics(frf_teacher, target)
                    teacher_frf_mse.append(tmse)
                    teacher_amp_mae.append(tmae)
                    teacher_amp_mape.append(tmape)
            else:
                _, omega_pred, log_zeta_pred, zeta_pred, phi_pred = _forward_modal(net, batch)

            modal_metrics.append(_compute_modal_metrics(
                omega_pred, batch["modal_omega_phys"],
                zeta_pred, batch["modal_zeta"],
                phi_pred, batch["modal_phi"], batch["batch"],
                phi_mode=phi_mode,
            ))

    if net_was_training:
        net.train()

    val_w = np.mean(np.stack([m["w"] for m in modal_metrics]), axis=0)
    val_z = np.mean(np.stack([m["z"] for m in modal_metrics]), axis=0)
    val_mac = np.mean(np.stack([m["mac"] for m in modal_metrics]), axis=0)
    val_phi_n = np.mean(np.stack([m["phi_n"] for m in modal_metrics]), axis=0)
    val_phi_a = np.mean(np.stack([m["phi_a"] for m in modal_metrics]), axis=0)
    val_dir = np.mean(np.stack([m["dir"] for m in modal_metrics]), axis=0)

    zeta_score_weight = float(config.get("modal_score_zeta_weight", 0.3))
    val_modal_score = float(
        np.mean(val_w) + zeta_score_weight * np.mean(val_z)
        + (1.0 - np.mean(val_mac)) * 100.0 + 0.05 * np.mean(val_phi_a)
    )

    return {
        "loss (MSE)": float(np.mean(frf_mse)) if frf_mse else np.inf,
        "Amplitude MAE": float(np.mean(amp_mae)) if amp_mae else 0.0,
        "Amplitude MAPE (%)": float(np.mean(amp_mape)) if amp_mape else 0.0,
        "Teacher loss (MSE)": float(np.mean(teacher_frf_mse)) if teacher_frf_mse else np.inf,
        "Teacher Amplitude MAE": float(np.mean(teacher_amp_mae)) if teacher_amp_mae else 0.0,
        "Teacher Amplitude MAPE (%)": float(np.mean(teacher_amp_mape)) if teacher_amp_mape else 0.0,
        "val_w": val_w,
        "val_z": val_z,
        "val_mac": val_mac,
        "val_phi_n": val_phi_n,
        "val_phi_a": val_phi_a,
        "val_dir": val_dir,
        "val_modal_score": val_modal_score,
    }


def _compute_frf_metrics(frf_pred: torch.Tensor, target: torch.Tensor):
    mse = float(torch.mean((frf_pred - target) ** 2).cpu())
    amp_p = torch.norm(frf_pred, dim=-1)
    amp_t = torch.norm(target, dim=-1)
    mae = float(torch.mean(torch.abs(amp_p - amp_t)).cpu())
    mape = float(torch.mean(torch.abs(amp_p - amp_t) / (amp_t.abs() + 1e-8)).cpu() * 100.0)
    return mse, mae, mape


def _compute_modal_metrics(omega_pred, omega_true, zeta_pred, zeta_true, phi_pred, phi_true, batch_idx, phi_mode="xyz"):
    f_pred = omega_pred / (2.0 * torch.pi)
    f_true = omega_true / (2.0 * torch.pi)
    w = torch.mean(torch.abs(f_pred - f_true) / (f_true.abs() + 1e-8) * 100.0, dim=0)

    z = torch.mean(torch.abs(zeta_pred - zeta_true) / (zeta_true.abs() + 1e-8) * 100.0, dim=0)

    if phi_true.dim() == 2:
        tmp = phi_true.new_zeros(phi_true.shape[0], phi_true.shape[1], 3)
        tmp[..., 2] = phi_true
        phi_true = tmp
    if phi_pred.dim() == 2:
        tmp = phi_pred.new_zeros(phi_pred.shape[0], phi_pred.shape[1], 3)
        tmp[..., 2] = phi_pred
        phi_pred = tmp

    phi_mode = str(phi_mode).lower()
    n_graphs = int(batch_idx.max().item()) + 1
    mac_list, phi_n_list, phi_a_list, dir_list = [], [], [], []
    for g in range(n_graphs):
        m = batch_idx == g
        p = phi_pred[m]
        t = phi_true[m]

        if phi_mode in {"z_only", "z", "phi_z", "zonly"}:
            pz = p[..., 2]
            tz = t[..., 2]
            dot = torch.sum(pz * tz, dim=0, keepdim=True)
            pz = pz * torch.sign(dot + 1e-8)

            num = torch.sum(pz * tz, dim=0) ** 2
            den = torch.sum(pz ** 2, dim=0) * torch.sum(tz ** 2, dim=0) + 1e-8
            mac = num / den

            rmse = torch.sqrt(torch.mean((pz - tz) ** 2, dim=0))
            t_std = torch.std(tz, dim=0) + 1e-8
            phi_n = rmse / t_std * 100.0

            norm_p = torch.sqrt(torch.sum(pz ** 2, dim=0) + 1e-8)
            norm_t = torch.sqrt(torch.sum(tz ** 2, dim=0) + 1e-8)
            phi_a = torch.abs(norm_p - norm_t) / (norm_t + 1e-8) * 100.0

            # Direction classification is intentionally not a target in Z-only mode.
            dir_acc = torch.ones_like(mac) * 100.0
        else:
            dot = torch.sum(p * t, dim=(0, 2), keepdim=True)
            p = p * torch.sign(dot + 1e-8)

            num = torch.sum(p * t, dim=(0, 2)) ** 2
            den = torch.sum(p ** 2, dim=(0, 2)) * torch.sum(t ** 2, dim=(0, 2)) + 1e-8
            mac = num / den

            rmse = torch.sqrt(torch.mean((p - t) ** 2, dim=(0, 2)))
            t_std = torch.std(t.transpose(0, 1).reshape(t.shape[1], -1), dim=1) + 1e-8
            phi_n = rmse / t_std * 100.0

            norm_p = torch.sqrt(torch.sum(p ** 2, dim=(0, 2)) + 1e-8)
            norm_t = torch.sqrt(torch.sum(t ** 2, dim=(0, 2)) + 1e-8)
            phi_a = torch.abs(norm_p - norm_t) / (norm_t + 1e-8) * 100.0

            e_p = torch.sum(p ** 2, dim=0)
            e_t = torch.sum(t ** 2, dim=0)
            dir_p = torch.argmax(e_p, dim=-1)
            dir_t = torch.argmax(e_t, dim=-1)
            dir_acc = (dir_p == dir_t).float() * 100.0

        mac_list.append(mac)
        phi_n_list.append(phi_n)
        phi_a_list.append(phi_a)
        dir_list.append(dir_acc)

    return {
        "w": w.detach().cpu().numpy(),
        "z": z.detach().cpu().numpy(),
        "mac": torch.stack(mac_list).mean(dim=0).detach().cpu().numpy(),
        "phi_n": torch.stack(phi_n_list).mean(dim=0).detach().cpu().numpy(),
        "phi_a": torch.stack(phi_a_list).mean(dim=0).detach().cpu().numpy(),
        "dir": torch.stack(dir_list).mean(dim=0).detach().cpu().numpy(),
    }


def _require_tensor_frequencies(batch: Dict) -> torch.Tensor:
    frequencies = batch.get("frequencies")
    if not torch.is_tensor(frequencies):
        raise ValueError("FRF training requires equal-length frequency grids so frequencies can be stacked.")
    return frequencies


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


def _format_val_results(val_results):
    if not val_results:
        return [""] * 18
    vals = []
    for key in ["val_w", "val_z", "val_mac", "val_phi_n", "val_phi_a"]:
        vals.extend([f"{x:.6f}" for x in val_results[key]])
    vals.extend([
        f"{val_results['val_dir'][1]:.6f}",
        f"{val_results['val_dir'][2]:.6f}",
        f"{val_results['val_modal_score']:.6f}",
    ])
    return vals


def save_model(path, epoch, net, optimizer, loss, filename="checkpoint_best", config=None, model_cfg=None):
    os.makedirs(path, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss),
        "config": config,
        "model_cfg": model_cfg,
    }, os.path.join(path, filename))


def _log(msg, logger=None):
    print(msg)
    if logger is not None:
        logger.info(msg)
