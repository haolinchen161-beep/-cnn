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
from .losses import modal_loss, frf_loss
from .augmentations import create_augmenter


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler, logger=None, start_epoch=0):
    """
    GrooveTransFRF 两阶段训练循环。

    阶段1 (0 ~ 动态解锁):      全解冻纯模态, 仅modal_loss
    阶段2 (动态解锁 ~ total):  模态+FRF×50
    
    
    
    """
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    total_epochs = config.get('epochs', 2000)

    frf_weight = config.get('frf_loss_weight', 50.0)

    # 数据增强器
    augmenter = create_augmenter(config)

    # 损失日志
    import csv
    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "loss_log.csv")
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['轮次', '训练损失', 'ω%', 'ζ%', 'φMSE', 'w占比%', 'z占比%', 'phi占比%', 'FRF占比%', '验证MSE', '幅值MAE', '幅值MAPE%', '学习率'])

    phase2_unlocked = False
    unlock_epoch = start_epoch  # 防止断点续训报错

    try:
      for epoch in range(start_epoch, total_epochs):
        losses, omega_losses, zeta_losses = [], [], []
        weighted_w_losses, weighted_z_losses, weighted_p_losses = [], [], []

        # ---- 两阶段判定 (动态解锁: ω<1% 或 epoch>600) ----
        in_phase1 = not phase2_unlocked
        in_phase2 = phase2_unlocked

        # ---- 阶段切换 ----
        if in_phase1 and epoch == 0:
            _log("=== 阶段1: 全解冻纯模态, ω<0.5% 或 epoch>1000 解锁 FRF ===", logger)
        elif in_phase2 and epoch > 0 and not getattr(net, '_phase2_logged', False):
            _log(f"=== 阶段2: FRF 联合训练 (第 {epoch} 轮解锁) ===", logger)
            net._phase2_logged = True
            lowest = np.inf

        for batch in dataloader:
            optimizer.zero_grad()

            # 数据增强
            if augmenter is not None:
                augmenter.train()
                batch = augmenter(batch)

            img = batch['image_tensor'].to(args.device)
            coords = batch['query_coords'].to(args.device)
            batch_idx_t = batch['batch'].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                if in_phase2:
                    # 阻尼退火: alpha 从 10 线性衰减到 1 (200轮内)
                    phase2_epoch = epoch - unlock_epoch
                    alpha = max(1.0, 10.0 - 9.0 * phase2_epoch / 200.0)
                    frequencies = batch['frequencies'].to(args.device)
                    phi_exc = batch.get('modal_phi_exc')
                    phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
                    # φ_exc 符号对齐
                    if phi_exc is not None:
                        with torch.no_grad():
                            _, _, _, phi_scan = net(img, coords, frequencies, None, batch_idx_t)
                        modal_phi = batch['modal_phi'].to(args.device)
                        phi_exc_corrected = phi_exc.clone()
                        b_idx = batch_idx_t
                        B = phi_exc.shape[0]
                        if b_idx is not None:
                            for i in range(int(b_idx.max().item()) + 1):
                                mask = (b_idx == i)
                                dot = torch.sum(phi_scan[mask] * modal_phi[mask], dim=0)
                                phi_exc_corrected[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                        else:
                            N_per = phi_scan.shape[0] // B
                            phi_s = phi_scan.view(B, N_per, -1)
                            mp_s = modal_phi.view(B, N_per, -1)
                            for i in range(B):
                                dot = torch.sum(phi_s[i] * mp_s[i], dim=0)
                                phi_exc_corrected[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                        phi_exc = phi_exc_corrected
                    frf_pred, omega_pred, zeta_pred, phi_pred = net(img, coords, frequencies, phi_exc, batch_idx_t, alpha=alpha)
                    loss_m, l_w, l_z, l_p = modal_loss(
                        omega_pred, batch['modal_omega_norm'].to(args.device),
                        zeta_pred, batch['modal_zeta'].to(args.device),
                        phi_pred, batch['modal_phi'].to(args.device),
                        batch_idx=batch_idx_t)
                    raw_frf = frf_loss(frf_pred, batch['point_frf'].to(args.device))
                    phase2_ep = epoch - unlock_epoch
                    current_frf_w = 0.05 * min(1.0, phase2_ep / 20.0)  # 预热20轮, 目标0.05
                    loss = loss_m + current_frf_w * raw_frf
                else:
                    _, omega_pred, zeta_pred, phi_pred = net(img, coords, None, None, batch_idx_t)
                    loss_m, l_w, l_z, l_p = modal_loss(
                        omega_pred, batch['modal_omega_norm'].to(args.device),
                        zeta_pred, batch['modal_zeta'].to(args.device),
                        phi_pred, batch['modal_phi'].to(args.device),
                        batch_idx=batch_idx_t)
                    loss = loss_m

            losses.append(loss.detach().cpu().item())
            omega_target = batch['modal_omega_norm'].to(args.device)
            omega_rel_err = torch.abs(omega_pred - omega_target) / (omega_target + 1e-8)
            omega_losses.append(omega_rel_err.mean().detach().cpu().item())
            zeta_target = batch['modal_zeta'].to(args.device)
            zeta_rel_err = torch.abs(zeta_pred - zeta_target) / (zeta_target + 1e-8)
            zeta_losses.append(zeta_rel_err.mean().detach().cpu().item())
            weighted_w_losses.append(l_w.detach().cpu().item())
            weighted_z_losses.append(l_z.detach().cpu().item())
            weighted_p_losses.append(l_p.detach().cpu().item())

            scaler.scale(loss).backward()

            # 分组件梯度裁剪
            _apply_gradient_clip(net, config)

            scaler.step(optimizer)
            scaler.update()

        mean_loss = np.mean(losses)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        raw_w = np.mean(omega_losses) if omega_losses else 0
        raw_z = np.mean(zeta_losses) if zeta_losses else 0
        wgt_w = np.mean(weighted_w_losses) if weighted_w_losses else 0
        wgt_z = np.mean(weighted_z_losses) if weighted_z_losses else 0
        wgt_p = np.mean(weighted_p_losses) if weighted_p_losses else 0
        omega_pct = raw_w * 100
        zeta_pct  = raw_z * 100
        phi_mse = wgt_p if wgt_p > 0 else 0  # raw MSE (phi_weight=1)
        omega_share = wgt_w / mean_loss * 100 if mean_loss > 0 else 0
        zeta_share  = wgt_z / mean_loss * 100 if mean_loss > 0 else 0
        phi_share   = (wgt_p * 100.0) / mean_loss * 100 if mean_loss > 0 else 0
        frf_s = 0 if in_phase1 else (100 - omega_share - zeta_share - phi_share)
        _log(f"Epoch {epoch:4d} | w={omega_pct:.1f}% z={zeta_pct:.1f}% phiMSE={phi_mse:.3f} | 占比 w{omega_share:.0f}% z{zeta_share:.0f}% phi{phi_share:.0f}% frf{frf_s:.0f}% | total={mean_loss:.2e}", logger)

        # 动态解锁: ω误差 < 0.5% 才进入Phase 2
        if not phase2_unlocked and (omega_pct < 0.5 or epoch > 1000):
            trigger = 'ω<0.5%' if omega_pct < 0.5 else f'epoch>{1000}'
            phase2_unlocked = True
            unlock_epoch = epoch
            _log(f">>> {trigger} 动态解锁 Phase2! <<<", logger)

        lr = optimizer.param_groups[0]['lr']
        val_freq = config.get('validation_frequency', 5)
        if epoch % val_freq == 0 or epoch % int(total_epochs / 10) == 0:
            save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last")
            if in_phase1:
                # Phase 1 不训 FRF, 跳过 FRF 评估, 仅记录 ω_MAE
                val_results = evaluate(args, config, net, valloader, logger, epoch)
                omega_mae = val_results.get("ω_MAE (rad/s)", -1)
                _log(f"Epoch {epoch:4d} | ω_MAE={omega_mae:.1f} rad/s (Phase1: FRF metrics skipped)", logger)
                log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_pct:.3f}', f'{zeta_pct:.3f}', f'{phi_mse:.2f}', f'{omega_share:.1f}', f'{zeta_share:.1f}', f'{phi_share:.1f}', '0', '', '', '', f'{lr:.2e}'])
            else:
                val_results = evaluate(args, config, net, valloader, logger, epoch)
                val_loss = val_results["loss (MSE)"]
                frf_share = 100 - omega_share - zeta_share - phi_share
                log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_pct:.3f}', f'{zeta_pct:.3f}', f'{phi_mse:.2f}', f'{omega_share:.1f}', f'{zeta_share:.1f}', f'{phi_share:.1f}', f'{frf_share:.1f}', f'{val_loss:.4f}',
                                     f'{val_results.get("Amplitude MAE", 0):.4f}',
                                     f'{val_results.get("Amplitude MAPE (%)", 0):.2f}',
                                     f'{lr:.2e}'])
            log_file.flush()
            use_val_metric = not in_phase1
            if use_val_metric:
                val_loss = val_results["loss (MSE)"]
                best_metric = val_loss
                metric_name = "val_loss"
                fmt = ".6f"
            else:
                # Phase1: 用验证 ω_MAE 选最优 (比 train_loss 更直接)
                best_metric = val_results.get("ω_MAE (rad/s)", mean_loss)
                metric_name = "ω_MAE"
                fmt = ".1f"
            if best_metric < lowest:
                _log(f"best model ({metric_name}={best_metric:{fmt}})", logger)
                save_model(args.dir, epoch, net, optimizer, best_metric)
                lowest = best_metric
        else:
            frf_s = 0 if in_phase1 else (100 - omega_share - zeta_share - phi_share)
            log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_pct:.3f}', f'{zeta_pct:.3f}', f'{phi_mse:.2f}', f'{omega_share:.1f}', f'{zeta_share:.1f}', f'{phi_share:.1f}', f'{frf_s:.1f}', '', '', '', f'{lr:.2e}'])

        if epoch == (total_epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            if os.path.exists(path):
                net.load_state_dict(torch.load(path, map_location='cpu')["model_state_dict"])
            evaluate(args, config, net, valloader, logger, epoch, verbose=True)

    finally:
        log_file.close()

    return net


def _apply_gradient_clip(net, config):
    grad_clip = config.get('optimizer', {}).get('gradient_clip')
    if grad_clip is None:
        return
    _clip_module(net, 'encoder', 3.0)
    _clip_module(net, 'micro_decoder', 5.0)
    _clip_module(net, 'macro_decoder', 2.0)
def _clip_module(net, prefix, max_norm):
    """按参数名前缀裁剪梯度。"""
    params = [p for name, p in net.named_parameters()
              if name.startswith(prefix + '.') and p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True):
    """验证/测试评估"""
    prediction, output, omega_errs = _generate_preds(args, config, net, dataloader)
    results = _evaluate(prediction, output, omega_errs, logger, epoch, verbose)
    return results


def _generate_preds(args, config, net, dataloader):
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

            if isinstance(frequencies, list):
                for i, freqs_i in enumerate(frequencies):
                    m = (bt == i)
                    img_i = img[i:i+1]; c_i = coords[m].unsqueeze(0)
                    bt_i = torch.zeros(m.sum(), dtype=torch.long, device=args.device)
                    pe_i = phi_exc[i:i+1].to(args.device) if phi_exc is not None else None
                    if pe_i is not None:
                        with torch.no_grad():
                            _, _, _, phi_scan = net(img_i, c_i, freqs_i.unsqueeze(0).to(args.device), None, bt_i)
                        dot = torch.sum(phi_scan.squeeze(0) * batch['modal_phi'].to(args.device)[m], dim=0)
                        pe_i = pe_i * torch.sign(dot + 1e-8).unsqueeze(0)
                    r = net(img_i, c_i, freqs_i.unsqueeze(0).to(args.device), pe_i, bt_i)
                    if isinstance(r, tuple):
                        predictions.append(r[0].squeeze(0).cpu())
                        if omega_true is not None:
                            omega_errs.append((r[1].cpu() * 25000.0 - omega_true[i]).abs())
                    else:
                        predictions.append(r.squeeze(0).cpu())
                    outputs.append(target[i].cpu())
            else:
                target = target.to(args.device)
                frequencies = frequencies.to(args.device)
                phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
                if phi_exc is not None:
                    with torch.no_grad():
                        _, _, _, phi_scan = net(img, coords, frequencies, None, bt)
                    modal_phi = batch['modal_phi'].to(args.device)
                    phi_exc_c = phi_exc.clone()
                    for i in range(int(bt.max().item()) + 1):
                        m = (bt == i)
                        dot = torch.sum(phi_scan[m] * modal_phi[m], dim=0)
                        phi_exc_c[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                    phi_exc = phi_exc_c
                r = net(img, coords, frequencies, phi_exc, bt)
                if isinstance(r, tuple):
                    prediction = r[0]
                    if omega_true is not None:
                        omega_errs.append((r[1].detach().cpu() * 25000.0 - omega_true).abs())
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
    """评估: asinh→物理空间, 计算幅值 MAE 和百分比 MAPE."""
    if isinstance(prediction, list):
        asinh_mse_vals = [F.mse_loss(p, o).item() for p, o in zip(prediction, output)]
        results = {"loss (MSE)": np.mean(asinh_mse_vals)}
        mae_list, mape_list = [], []
        for p_asinh, o_asinh in zip(prediction, output):
            p_phys = p_asinh  # FRF已是线性物理量
            o_phys = o_asinh
            p_amp = torch.sqrt(p_phys[..., 0]**2 + p_phys[..., 1]**2 + 1e-8)
            o_amp = torch.sqrt(o_phys[..., 0]**2 + o_phys[..., 1]**2 + 1e-8)
            mae_list.append(F.l1_loss(p_amp, o_amp).item())
            mape_list.append((torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0)
        results["Amplitude MAE"] = np.mean(mae_list)
        results["Amplitude MAPE (%)"] = np.mean(mape_list)
    else:
        results = {}
        # 兜底: 确保 prediction 和 output 形状一致
        if prediction.shape != output.shape:
            output = output.reshape(prediction.shape)
        results["loss (MSE)"] = F.mse_loss(prediction, output).item()
        if prediction.ndim >= 3 and prediction.shape[-1] == 2:
            p_phys = prediction
            o_phys = output
            p_amp = torch.sqrt(p_phys[..., 0]**2 + p_phys[..., 1]**2 + 1e-8)
            o_amp = torch.sqrt(o_phys[..., 0]**2 + o_phys[..., 1]**2 + 1e-8)
            results["Amplitude MAE"] = F.l1_loss(p_amp, o_amp).item()
            results["Amplitude MAPE (%)"] = (torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0

    # ω误差
    if omega_errs:
        results["ω_MAE (rad/s)"] = torch.cat([e.flatten() for e in omega_errs]).mean().item()

    if verbose:
        for key, val in results.items():
            _log(f"{key} = {val:4.4f}" if isinstance(val, float) else f"{key} = {val:4.4}", logger)

    return results


def _evaluate(prediction, output, omega_errs, logger, epoch, verbose=True):
    if isinstance(prediction, list):
        asinh_mse_vals = [F.mse_loss(p, o).item() for p, o in zip(prediction, output)]
        results = {"loss (MSE)": np.mean(asinh_mse_vals)}
        mae_list, mape_list = [], []
        for p_asinh, o_asinh in zip(prediction, output):
            p_phys = p_asinh  # FRF已是线性物理量; o_phys = o_asinh
            p_amp = torch.sqrt(p_phys[..., 0]**2 + p_phys[..., 1]**2 + 1e-8)
            o_amp = torch.sqrt(o_phys[..., 0]**2 + o_phys[..., 1]**2 + 1e-8)
            mae_list.append(F.l1_loss(p_amp, o_amp).item())
            mape_list.append((torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0)
        results["Amplitude MAE"] = np.mean(mae_list)
        results["Amplitude MAPE (%)"] = np.mean(mape_list)
    else:
        results = {}
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
    """简易日志: 若 logger 可用则用 logger，否则 print"""
    if logger and hasattr(logger, 'info'):
        logger.info(msg)
    else:
        print(msg)
