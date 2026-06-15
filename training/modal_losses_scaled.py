import torch
import torch.nn.functional as F


def frequency_loss(pred, target):
    fp = pred / (2.0 * torch.pi)
    ft = target / (2.0 * torch.pi)
    rel = (fp - ft) / ft.clamp_min(1e-6)
    loss = F.smooth_l1_loss(torch.log(fp.clamp_min(1e-6)), torch.log(ft.clamp_min(1e-6)))
    return loss, {"freq_mae_hz": torch.abs(fp - ft).mean().detach(), "freq_mape_percent": (torch.abs(rel).mean() * 100.0).detach()}


def align_phi(pred, target, batch):
    out = torch.empty_like(target)
    n = int(batch.max().item()) + 1 if batch.numel() else 0
    for i in range(n):
        m = batch == i
        dot = torch.sum(pred[m] * target[m], dim=(0, 2))
        s = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
        out[m] = target[m] * s.view(1, -1, 1)
    return out


def mac(a, b):
    num = torch.sum(a * b, dim=(0, 2)) ** 2
    den = torch.sum(a ** 2, dim=(0, 2)) * torch.sum(b ** 2, dim=(0, 2))
    return num / den.clamp_min(1e-12)


def phi_terms(pred, target, batch, weight=None):
    if weight is None:
        weight = torch.ones(pred.shape[0], dtype=pred.dtype, device=pred.device)
    else:
        weight = weight.to(pred.device, dtype=pred.dtype)
    mse_list, scale_list, mac_list = [], [], []
    n = int(batch.max().item()) + 1 if batch.numel() else 0
    for i in range(n):
        m = batch == i
        p, t = pred[m], target[m]
        w = weight[m].view(-1, 1, 1)
        w = w / w.mean().clamp_min(1e-8)
        t_std = torch.std(t.transpose(0, 1).reshape(t.shape[1], -1), dim=1).clamp_min(1e-8)
        p_std = torch.std(p.transpose(0, 1).reshape(p.shape[1], -1), dim=1).clamp_min(1e-8)
        mse_list.append(torch.mean(w * ((p - t) / t_std.view(1, -1, 1)) ** 2))
        scale_list.append(torch.mean(torch.abs(torch.log(p_std / t_std))))
        mac_list.append(mac(p, t))
    return torch.stack(mse_list).mean(), torch.stack(scale_list).mean(), torch.stack(mac_list, dim=0)


def modal_loss(out, batch, freq_weight=1.0, phi_weight=1.0, mac_weight=5.0, scale_weight=1.0):
    lf, fm = frequency_loss(out["omega"], batch["modal_omega_phys"])
    target = align_phi(out["phi"], batch["modal_phi"], batch["batch"])
    lmse, lscale, macs = phi_terms(out["phi"], target, batch["batch"], batch.get("node_weight"))
    lphi = lmse + scale_weight * lscale + mac_weight * (1.0 - macs.mean())
    loss = freq_weight * lf + phi_weight * lphi
    return loss, {
        "loss": loss.detach(), "loss_freq": lf.detach(), "loss_phi": lphi.detach(),
        "phi_mse": lmse.detach(), "phi_scale": lscale.detach(), "phi_mac": macs.mean().detach(),
        "phi_mac_mode1": macs[:, 0].mean().detach(), "phi_mac_mode2": macs[:, 1].mean().detach(), "phi_mac_mode3": macs[:, 2].mean().detach(),
        **fm,
    }
