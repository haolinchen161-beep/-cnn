"""
trainer.py — GrooveTransFRF 两阶段训练循环 + 评估。

训练策略: Phase1 (纯模态) → 动态解锁 → Phase2 (模态+FRF)

数据流:
    geometry + frequencies → net → per_point_frf (B, N, n_freqs[, out_dim])
    损失: modal_loss (ω, ζ, φ) + frf_loss (物理约束)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from .losses import modal_loss, frf_loss, branch_loss


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler, logger=None, start_epoch=0):
    """
    GrooveTransFRF 两阶段训练循环。

    阶段1 (0 ~ 动态解锁):      全解冻纯模态, 仅 modal_loss
    阶段2 (动态解锁 ~ total):  模态+FRF 弱约束
    """
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    total_epochs = config.get('epochs', 2000)
    frf_weight = config.get('frf_loss_weight', 0.05)
    zeta_warmup_epochs = config.get('zeta_warmup_epochs', 40)

    # 损失日志
    import csv
    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "loss_log.csv")
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['轮次', '训练损失', 'w1%','w2%','w3%', 'z1%','z2%','z3%', 'φloss', 'φn1','φn2','φn3', 'φa1','φa2','φa3', 'MAC1','MAC2','MAC3', 'w占比%','z占比%','phi占比%','FRF占比%', 'kl', 'dir2%', 'dir3%', '验证MSE', '幅值MAE', '幅值MAPE%', '学习率'])

    phase2_unlocked = False
    unlock_epoch = start_epoch

    try:
      for epoch in range(start_epoch, total_epochs):
        losses, omega_losses, zeta_losses, mac_losses = [], [], [], []
        phi_n_losses, phi_a_losses, kl_losses, dir_accs, dir_accs_m2, dir_accs_m3 = [], [], [], [], [], []
        weighted_w_losses, weighted_z_losses, weighted_p_losses = [], [], []

        in_phase1 = not phase2_unlocked
        in_phase2 = phase2_unlocked

        # 阶段切换
        if in_phase1 and epoch == 0:
            _log("=== 阶段1: 纯模态训练 (enable_phase2=False, FRF 未解锁) ===", logger)
        elif in_phase2 and epoch > 0 and not getattr(net, '_phase2_logged', False):
            _log(f"=== 阶段2: FRF 联合训练 (第 {epoch} 轮解锁) ===", logger)
            net._phase2_logged = True
            lowest = np.inf

        # ================= 全新三阶段控制 =================
        omega_pretrain_epochs = config.get('omega_pretrain_epochs', 40)
        in_phase0 = epoch < omega_pretrain_epochs

        if in_phase0:
            if epoch == 0 and not getattr(net, '_phase0_logged', False):
                _log(f"=== 阶段 0: 频率专属预训练 (仅训频率, 冻结振型/阻尼/KL, 前 {omega_pretrain_epochs} 轮) ===", logger)
                net._phase0_logged = True

            # Phase 0: 彻底清场，只给频率梯度
            current_phi_w = 0.0
            current_zeta_w = 0.0
            current_omega_w = config.get('omega_loss_weight', 1.0) * 5.0  # 适当放大频率寻找速度
            kl_weight = 0.0
        else:
            if epoch == omega_pretrain_epochs and not getattr(net, '_phase1_logged', False):
                _log(f"=== 阶段 1: 全模态联合训练 (解锁振型与阻尼) ===", logger)
                net._phase1_logged = True

            # Phase 1 & 2: 恢复全模态正常权重
            current_phi_w = config.get('phi_loss_weight', 3.0)
            current_zeta_w = config.get('zeta_loss_weight', 10.0)
            current_omega_w = config.get('omega_loss_weight', 1.0)
            kl_weight = 20.0
        # ===============================================

        for batch in dataloader:
            optimizer.zero_grad()

            img = batch['image_tensor'].to(args.device)
            coords = batch['query_coords'].to(args.device)
            batch_idx_t = batch['batch'].to(args.device)

            # Node 信息 (传给 NodePhiRefiner)
            node_xyz = batch.get('node_xyz')
            node_features = batch.get('node_features')
            node_xyz = node_xyz.to(args.device) if node_xyz is not None else None
            node_features = node_features.to(args.device) if node_features is not None else None

            with torch.cuda.amp.autocast(enabled=args.fp16):
                if in_phase2:
                    phase2_epoch = epoch - unlock_epoch
                    # 永久 Teacher Forcing 已启用，峰位绝对对齐，无需虚增阻尼
                    damping_alpha = 1.0
                    omega_true = batch['modal_omega_phys'].to(args.device)

                    frequencies = batch['frequencies'].to(args.device)
                    phi_exc = batch.get('modal_phi_exc')
                    phi_exc = phi_exc.to(args.device) if phi_exc is not None else None

                    # φ_exc 符号对齐
                    if phi_exc is not None:
                        with torch.no_grad():
                            _, _, _, _, phi_scan = net(img, coords, None, None, batch_idx_t,
                                                       node_xyz=node_xyz, node_features=node_features)
                        modal_phi = batch['modal_phi'].to(args.device)
                        phi_exc_corrected = phi_exc.clone()
                        for i in range(int(batch_idx_t.max().item()) + 1):
                            mask = (batch_idx_t == i)
                            dot = torch.sum(phi_scan[mask] * modal_phi[mask], dim=(0, 2))
                            phi_exc_corrected[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                        phi_exc = phi_exc_corrected

                    frf_pred, omega_phys_pred, log_zeta_pred, zeta_pred, phi_pred = net(
                        img, coords, frequencies, phi_exc, batch_idx_t, alpha=damping_alpha,
                        node_xyz=node_xyz, node_features=node_features,
                        omega_true=omega_true)

                    loss_m, l_w, l_z, l_p, mac_val = modal_loss(
                        omega_phys_pred, batch['modal_omega_phys'].to(args.device),
                        log_zeta_pred, batch['modal_zeta'].to(args.device),
                        phi_pred, batch['modal_phi'].to(args.device),
                        batch_idx=batch_idx_t,
                        omega_weight=current_omega_w, zeta_weight=current_zeta_w,
                        phi_weight=current_phi_w)
                    mac_losses.append(mac_val.detach().cpu().numpy())
                    pn, pa = _compute_phi_metrics(phi_pred, batch['modal_phi'].to(args.device), batch_idx_t)
                    phi_n_losses.append(pn.detach().cpu().numpy())
                    phi_a_losses.append(pa.detach().cpu().numpy())

                    raw_frf = frf_loss(frf_pred, batch['point_frf'].to(args.device))
                    frf_warmup = config.get('frf_warmup_epochs', 20)
                    current_frf_w = frf_weight * min(1.0, phase2_epoch / max(frf_warmup, 1))
                    phi_tgt = batch['modal_phi'].to(args.device)
                    if kl_weight > 0:
                        kl = branch_loss(net.branch_log_probs, phi_tgt, batch_idx_t)
                        kl_losses.append(kl.detach().cpu().item())
                        loss = loss_m + current_frf_w * raw_frf + kl_weight * kl
                    else:
                        loss = loss_m + current_frf_w * raw_frf
                else:
                    _, omega_phys_pred, log_zeta_pred, zeta_pred, phi_pred = net(
                        img, coords, None, None, batch_idx_t,
                        node_xyz=node_xyz, node_features=node_features)

                    loss_m, l_w, l_z, l_p, mac_val = modal_loss(
                        omega_phys_pred, batch['modal_omega_phys'].to(args.device),
                        log_zeta_pred, batch['modal_zeta'].to(args.device),
                        phi_pred, batch['modal_phi'].to(args.device),
                        batch_idx=batch_idx_t,
                        omega_weight=current_omega_w, zeta_weight=current_zeta_w,
                        phi_weight=current_phi_w)
                    phi_tgt = batch['modal_phi'].to(args.device)
                    if kl_weight > 0:
                        kl = branch_loss(net.branch_log_probs, phi_tgt, batch_idx_t)
                        kl_losses.append(kl.detach().cpu().item())
                        loss = loss_m + kl_weight * kl
                    else:
                        loss = loss_m  # Phase 0 时只有纯净的频率损失
                    mac_losses.append(mac_val.detach().cpu().numpy())
                    pn, pa = _compute_phi_metrics(phi_pred, batch['modal_phi'].to(args.device), batch_idx_t)
                    phi_n_losses.append(pn.detach().cpu().numpy())
                    phi_a_losses.append(pa.detach().cpu().numpy())

            losses.append(loss.detach().cpu().item())

            # 日志: 物理单位直接算相对误差，OmegaHead 保证单调无需 sort
            omega_target = batch['modal_omega_phys'].to(args.device)
            omega_rel_err = torch.abs(omega_phys_pred - omega_target) / (omega_target + 1e-8)
            omega_losses.append(omega_rel_err.mean(dim=0).detach().cpu().numpy())  # [K] per-mode

            zeta_target = batch['modal_zeta'].to(args.device)
            zeta_rel_err = torch.abs(zeta_pred - zeta_target) / (zeta_target + 1e-8)
            zeta_losses.append(zeta_rel_err.mean(dim=0).detach().cpu().numpy())  # [K] per-mode

            weighted_w_losses.append(l_w.detach().cpu().item())
            weighted_z_losses.append(l_z.detach().cpu().item())
            weighted_p_losses.append(l_p.detach().cpu().item())

            scaler.scale(loss).backward()
            _apply_gradient_clip(net, config)
            scaler.step(optimizer)
            scaler.update()

        mean_loss = np.mean(losses)

        if scheduler is not None:
            scheduler.step()

        if omega_losses:
            omega_per_mode = np.stack(omega_losses).mean(axis=0) * 100
            w_str = f'w=[{omega_per_mode[0]:.1f}/{omega_per_mode[1]:.1f}/{omega_per_mode[2]:.1f}]%'
            omega_pct = omega_per_mode.mean()
        else:
            w_str = 'w=...'; omega_pct = 0; omega_per_mode = np.zeros(3)
        if zeta_losses:
            zeta_per_mode = np.stack(zeta_losses).mean(axis=0) * 100
            z_str = f'z=[{zeta_per_mode[0]:.0f}/{zeta_per_mode[1]:.0f}/{zeta_per_mode[2]:.0f}]%'
            zeta_pct = zeta_per_mode.mean()
        else:
            z_str = 'z=...'; zeta_pct = 0; zeta_per_mode = np.zeros(3)
        wgt_w = np.mean(weighted_w_losses) if weighted_w_losses else 0
        wgt_z = np.mean(weighted_z_losses) if weighted_z_losses else 0
        wgt_p = np.mean(weighted_p_losses) if weighted_p_losses else 0
        phi_loss = wgt_p if wgt_p > 0 else 0
        omega_share = wgt_w / mean_loss * 100 if mean_loss > 0 else 0
        zeta_share = wgt_z / mean_loss * 100 if mean_loss > 0 else 0
        phi_share = wgt_p / mean_loss * 100 if mean_loss > 0 else 0
        frf_s = 0 if in_phase1 else (100 - omega_share - zeta_share - phi_share)
        if phi_n_losses:
            phi_n_per_mode = np.stack(phi_n_losses).mean(axis=0)
            phi_a_per_mode = np.stack(phi_a_losses).mean(axis=0)
            phi_str = f'φn=[{phi_n_per_mode[0]:.1f}/{phi_n_per_mode[1]:.1f}/{phi_n_per_mode[2]:.1f}]% φa=[{phi_a_per_mode[0]:.1f}/{phi_a_per_mode[1]:.1f}/{phi_a_per_mode[2]:.1f}]%'
        else:
            phi_str = 'φn=... φa=...'
            phi_n_per_mode = np.zeros(3)
            phi_a_per_mode = np.zeros(3)
        if mac_losses:
            mac_per_mode = np.stack(mac_losses).mean(axis=0)
            mac_str = f'MAC=[{mac_per_mode[0]:.3f}/{mac_per_mode[1]:.3f}/{mac_per_mode[2]:.3f}]'
            mac_scalar = mac_per_mode.mean()
        else:
            mac_str = 'MAC=...'; mac_scalar = 0
        kl_avg = np.mean(kl_losses) if kl_losses else 0
        dir2 = np.mean(dir_accs_m2) * 100 if dir_accs_m2 else 0
        dir3 = np.mean(dir_accs_m3) * 100 if dir_accs_m3 else 0
        _log(f"Epoch {epoch:4d} | {w_str} {z_str} {phi_str} {mac_str} | w{omega_share:.0f}z{zeta_share:.0f}ph{phi_share:.0f} | kl={kl_avg:.3f} dir2={dir2:.0f}% dir3={dir3:.0f}% | loss={mean_loss:.1f}", logger)

        # 动态解锁: enable_phase2=True 且 epoch >= phase2_min_epoch
        enable_phase2 = config.get('enable_phase2', False)
        phase2_min_epoch = config.get('phase2_min_epoch', 200)
        if not phase2_unlocked and enable_phase2 and epoch >= phase2_min_epoch:
            phase2_unlocked = True
            unlock_epoch = epoch
            _log(f">>> Phase2 unlocked at epoch {epoch} (FRF weak constraint) <<<", logger)

        lr = optimizer.param_groups[0]['lr']
        val_freq = config.get('validation_frequency', 5)
        if epoch % val_freq == 0 or epoch % int(total_epochs / 10) == 0:
            save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last")
            if in_phase1:
                val_results = evaluate(args, config, net, valloader, logger, epoch, phase1=True)
                omega_mae = val_results.get("ω_MAE (rad/s)", -1)
                _log(f"Epoch {epoch:4d} | ω_MAE={omega_mae:.1f} rad/s (Phase1)", logger)
                log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_per_mode[0]:.3f}',f'{omega_per_mode[1]:.3f}',f'{omega_per_mode[2]:.3f}', f'{zeta_per_mode[0]:.1f}',f'{zeta_per_mode[1]:.1f}',f'{zeta_per_mode[2]:.1f}', f'{phi_loss:.2f}', f'{phi_n_per_mode[0]:.1f}',f'{phi_n_per_mode[1]:.1f}',f'{phi_n_per_mode[2]:.1f}', f'{phi_a_per_mode[0]:.1f}',f'{phi_a_per_mode[1]:.1f}',f'{phi_a_per_mode[2]:.1f}', f'{mac_per_mode[0]:.3f}',f'{mac_per_mode[1]:.3f}',f'{mac_per_mode[2]:.3f}', f'{omega_share:.1f}', f'{zeta_share:.1f}', f'{phi_share:.1f}', '0', '', '', '', f'{kl_avg:.4f}', f'{dir2:.1f}', f'{dir3:.1f}', f'{lr:.2e}'])
            else:
                val_results = evaluate(args, config, net, valloader, logger, epoch)
                val_loss = val_results["loss (MSE)"]
                frf_share = 100 - omega_share - zeta_share - phi_share
                log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_per_mode[0]:.3f}',f'{omega_per_mode[1]:.3f}',f'{omega_per_mode[2]:.3f}', f'{zeta_per_mode[0]:.1f}',f'{zeta_per_mode[1]:.1f}',f'{zeta_per_mode[2]:.1f}', f'{phi_loss:.2f}', f'{phi_n_per_mode[0]:.1f}',f'{phi_n_per_mode[1]:.1f}',f'{phi_n_per_mode[2]:.1f}', f'{phi_a_per_mode[0]:.1f}',f'{phi_a_per_mode[1]:.1f}',f'{phi_a_per_mode[2]:.1f}', f'{mac_per_mode[0]:.3f}',f'{mac_per_mode[1]:.3f}',f'{mac_per_mode[2]:.3f}', f'{omega_share:.1f}', f'{zeta_share:.1f}', f'{phi_share:.1f}', f'{frf_share:.1f}', f'{val_loss:.4f}',
                                     f'{val_results.get("Amplitude MAE", 0):.4f}',
                                     f'{val_results.get("Amplitude MAPE (%)", 0):.2f}',
                                     f'{kl_avg:.4f}', f'{dir2:.1f}', f'{dir3:.1f}', f'{lr:.2e}'])
            log_file.flush()

            # best checkpoint: 用验证集 ω_MAE (防过拟合)
            if in_phase1:
                best_metric = val_results.get("ω_MAE (rad/s)", omega_pct)
                metric_name = "val_ω_MAE"
                fmt = ".1f"
            else:
                best_metric = val_results["loss (MSE)"]
                metric_name = "val_loss"
                fmt = ".6f"
            if best_metric < lowest:
                _log(f"best model ({metric_name}={best_metric:{fmt}})", logger)
                save_model(args.dir, epoch, net, optimizer, best_metric)
                lowest = best_metric
        else:
            frf_s = 0 if in_phase1 else (100 - omega_share - zeta_share - phi_share)
            log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_per_mode[0]:.3f}',f'{omega_per_mode[1]:.3f}',f'{omega_per_mode[2]:.3f}', f'{zeta_per_mode[0]:.1f}',f'{zeta_per_mode[1]:.1f}',f'{zeta_per_mode[2]:.1f}', f'{phi_loss:.2f}', f'{phi_n_per_mode[0]:.1f}',f'{phi_n_per_mode[1]:.1f}',f'{phi_n_per_mode[2]:.1f}', f'{phi_a_per_mode[0]:.1f}',f'{phi_a_per_mode[1]:.1f}',f'{phi_a_per_mode[2]:.1f}', f'{mac_per_mode[0]:.3f}',f'{mac_per_mode[1]:.3f}',f'{mac_per_mode[2]:.3f}', f'{omega_share:.1f}', f'{zeta_share:.1f}', f'{phi_share:.1f}', f'{frf_s:.1f}', '', '', '', f'{kl_avg:.4f}', f'{dir2:.1f}', f'{dir3:.1f}', f'{lr:.2e}'])

        if epoch == (total_epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            if os.path.exists(path):
                net.load_state_dict(torch.load(path, map_location='cpu')["model_state_dict"])
            evaluate(args, config, net, valloader, logger, epoch, verbose=True, phase1=not phase2_unlocked)

    finally:
        log_file.close()

    return net


def _compute_phi_metrics(phi_pred, phi_target, batch_idx):
    """逐图计算 φn (NRMSE%) 和 φa (范数误差%)，phi 为 [N,K,3]。返回 per-mode [K]."""
    dot = torch.sum(phi_pred * phi_target, dim=(0, 2), keepdim=True)
    sign = torch.sign(dot + 1e-8)
    aligned_target = phi_target * sign

    n_graphs = int(batch_idx.max().item()) + 1
    nrmse_list, norm_err_list = [], []
    for i in range(n_graphs):
        mask = (batch_idx == i)
        p = phi_pred[mask]   # [N, K, 3]
        t = aligned_target[mask]
        # φn: 三维 RMSE / (max-min)，沿 (N, XYZ) 两维
        rmse = torch.sqrt(torch.mean((p - t) ** 2, dim=(0, 2)))
        ptp = t.amax(dim=(0, 2)) - t.amin(dim=(0, 2)) + 1e-8
        nrmse_list.append(rmse / ptp * 100.0)
        # φa: 三维总体范数误差
        norm_p = torch.sqrt(torch.sum(p ** 2, dim=(0, 2)))
        norm_t = torch.sqrt(torch.sum(t ** 2, dim=(0, 2))) + 1e-8
        norm_err_list.append(torch.abs(norm_p - norm_t) / norm_t * 100.0)
    phi_n = torch.stack(nrmse_list, dim=0).mean(dim=0)  # [K]
    phi_a = torch.stack(norm_err_list, dim=0).mean(dim=0)  # [K]
    return phi_n, phi_a


def _apply_gradient_clip(net, config):
    grad_clip = config.get('optimizer', {}).get('gradient_clip')
    if grad_clip is None:
        return
    _clip_module(net, 'encoder', 3.0)
    _clip_module(net, 'micro_decoder', 5.0)
    _clip_module(net, 'omega_head', 2.0)
    _clip_module(net, 'zeta_head', 2.0)
    _clip_module(net, 'phi_refiner', 2.0)
    _clip_module(net, 'phi_scale_head', 2.0)
    _clip_module(net, 'branch_head', 2.0)


def _clip_module(net, prefix, max_norm):
    params = [p for name, p in net.named_parameters()
              if name.startswith(prefix + '.') and p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True, phase1=False):
    """验证/测试评估。phase1=True 时只计算 ω_MAE，跳过 FRF 推理。"""
    prediction, output, omega_errs = _generate_preds(args, config, net, dataloader, phase1=phase1)
    results = _evaluate(prediction, output, omega_errs, logger, epoch, verbose, phase1=phase1)
    return results


def _generate_preds(args, config, net, dataloader, phase1=False):
    net.eval()
    with torch.no_grad():
        predictions, outputs = [], []
        omega_errs = []
        for batch in dataloader:
            img = batch['image_tensor'].to(args.device)
            coords = batch['query_coords'].to(args.device)
            bt = batch['batch'].to(args.device)
            target = batch['point_frf']
            frequencies = batch['frequencies']
            phi_exc = batch.get('modal_phi_exc')
            omega_true = batch.get('modal_omega_phys')

            _nx = batch.get('node_xyz'); _nf = batch.get('node_features')
            _nx = _nx.to(args.device) if _nx is not None else None
            _nf = _nf.to(args.device) if _nf is not None else None

            # Phase1: 只取 ω，跳过 FRF 推理
            if phase1:
                _, omega_pred, _, _, _ = net(img, coords, None, None, bt,
                                             node_xyz=_nx, node_features=_nf)
                if omega_true is not None:
                    omega_errs.append((omega_pred.detach().cpu() - omega_true).abs())
                continue

            if isinstance(frequencies, list):
                for i, freqs_i in enumerate(frequencies):
                    m = (bt == i)
                    img_i = img[i:i+1]; c_i = coords[m].unsqueeze(0)
                    bt_i = torch.zeros(m.sum(), dtype=torch.long, device=args.device)
                    pe_i = phi_exc[i:i+1].to(args.device) if phi_exc is not None else None
                    _nx_i = _nx[m] if _nx is not None else None
                    _nf_i = _nf[m] if _nf is not None else None
                    if pe_i is not None:
                        with torch.no_grad():
                            _, _, _, _, phi_scan = net(img_i, c_i, None, None, bt_i,
                                                       node_xyz=_nx_i, node_features=_nf_i)
                        dot = torch.sum(phi_scan.squeeze(0) * batch['modal_phi'].to(args.device)[m], dim=(0, 2))
                        pe_i = pe_i * torch.sign(dot + 1e-8).unsqueeze(0)
                    r = net(img_i, c_i, freqs_i.unsqueeze(0).to(args.device), pe_i, bt_i,
                            node_xyz=_nx_i, node_features=_nf_i)
                    if isinstance(r, tuple):
                        predictions.append(r[0].squeeze(0).cpu())
                        if omega_true is not None:
                            omega_pred_val = r[1].cpu()  # 已是 rad/s, OmegaHead 保证单调
                            omega_errs.append((omega_pred_val - omega_true[i]).abs())
                    else:
                        predictions.append(r.squeeze(0).cpu())
                    outputs.append(target[i].cpu())
            else:
                target = target.to(args.device)
                frequencies = frequencies.to(args.device)
                phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
                if phi_exc is not None:
                    with torch.no_grad():
                        _, _, _, _, phi_scan = net(img, coords, None, None, bt,
                                                   node_xyz=_nx, node_features=_nf)
                    modal_phi = batch['modal_phi'].to(args.device)
                    phi_exc_c = phi_exc.clone()
                    for i in range(int(bt.max().item()) + 1):
                        m = (bt == i)
                        dot = torch.sum(phi_scan[m] * modal_phi[m], dim=(0, 2))
                        phi_exc_c[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                    phi_exc = phi_exc_c
                r = net(img, coords, frequencies, phi_exc, bt,
                        node_xyz=_nx, node_features=_nf)
                if isinstance(r, tuple):
                    prediction = r[0]
                    if omega_true is not None:
                        omega_pred_val = r[1].detach().cpu()  # 已是 rad/s
                        omega_errs.append((omega_pred_val - omega_true).abs())
                else:
                    prediction = r
                pred_out = prediction.detach().cpu()
                tgt_out = target.detach().cpu()
                if pred_out.ndim == 3 and tgt_out.ndim == 4:
                    tgt_out = tgt_out.reshape(-1, *tgt_out.shape[2:])
                predictions.append(pred_out)
                outputs.append(tgt_out)

    try:
        return torch.cat(predictions, dim=0), torch.cat(outputs, dim=0), omega_errs
    except RuntimeError:
        return predictions, outputs, omega_errs


def _evaluate(prediction, output, omega_errs, logger, epoch, verbose=True, phase1=False):
    results = {}
    # Phase1: 只评估 ω，跳过 FRF 指标 (FRF 尚未训练，指标无意义)
    if not phase1:
        if isinstance(prediction, list):
            asinh_mse_vals = [F.mse_loss(p, o).item() for p, o in zip(prediction, output)]
            results["loss (MSE)"] = np.mean(asinh_mse_vals)
            mae_list, mape_list = [], []
            for p_asinh, o_asinh in zip(prediction, output):
                p_phys = p_asinh; o_phys = o_asinh
                p_amp = torch.sqrt(p_phys[..., 0]**2 + p_phys[..., 1]**2 + 1e-8)
                o_amp = torch.sqrt(o_phys[..., 0]**2 + o_phys[..., 1]**2 + 1e-8)
                mae_list.append(F.l1_loss(p_amp, o_amp).item())
                mape_list.append((torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0)
            results["Amplitude MAE"] = np.mean(mae_list)
            results["Amplitude MAPE (%)"] = np.mean(mape_list)
        else:
            if prediction.shape != output.shape:
                output = output.reshape(prediction.shape)
            results["loss (MSE)"] = F.mse_loss(prediction, output).item()
            if prediction.ndim >= 3 and prediction.shape[-1] == 2:
                p_phys = prediction; o_phys = output
                p_amp = torch.sqrt(p_phys[..., 0]**2 + p_phys[..., 1]**2 + 1e-8)
                o_amp = torch.sqrt(o_phys[..., 0]**2 + o_phys[..., 1]**2 + 1e-8)
                results["Amplitude MAE"] = F.l1_loss(p_amp, o_amp).item()
                results["Amplitude MAPE (%)"] = (torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0
    if omega_errs:
        results["ω_MAE (rad/s)"] = torch.cat([e.flatten() for e in omega_errs]).mean().item()
    if verbose:
        for key, val in results.items():
            _log(f"{key} = {val:4.4f}" if isinstance(val, float) else f"{key} = {val:4.4}", logger)
    return results


def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best"):
    os.makedirs(savepath, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(savepath, name))


def _log(msg, logger):
    if logger and hasattr(logger, 'info'):
        logger.info(msg)
    else:
        print(msg)
