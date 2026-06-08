"""可微模态叠加层。

网络预测模态参数，本模块无需可学习参数即可重建方向性节点 FRF：

    H_qf(Ω) = Σ_k φ_q^a · φ_f^b / (ω_k² - Ω² + j·2·ζ_k·ω_k·Ω)

其中 ``a`` 为响应方向，``b`` 为力激励方向。
例如 ``H_YY`` 表示 Y 向响应、Y 向激励。

ANSYS 生成器将物理柔度乘以 ``AMPLITUDE_SCALE``；
本解码器保持相同约定以兼容目标值。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalFRFDecoder(nn.Module):
    """无参数模态叠加解码器，支持任意方向的 FRF 重建。

    本层不假设特定的激励或响应方向。
    调用者提供 ``phi_response``（响应方向的模态振型）和
    ``phi_force_exc``（激励方向在激励节点处的模态值）。
    """

    def __init__(self, amp_scale: float = 500000.0, clamp_denominator: bool = True):
        super().__init__()
        self.amp_scale = amp_scale
        self.clamp_denominator = clamp_denominator

    def forward(self,
                phi_response: torch.Tensor,
                phi_force_exc: torch.Tensor,
                omega: torch.Tensor,
                zeta: torch.Tensor,
                frequencies_hz: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """为拼接的变长节点批次重建 FRF。

        Args:
            phi_response:  (total_N, K)，预测或目标模态振型（沿**响应**方向）。
            phi_force_exc: (B, K)，激励节点在**激励**方向上的模态值。
            omega:         (B, K)，固有圆频率，单位 rad/s。
            zeta:          (B, K)，模态阻尼比。
            frequencies_hz:(B, F)，物理频率，单位 Hz。
            batch:         (total_N,)，每个节点所属的图索引。

        Returns:
            (total_N, F, 2) 复数 FRF，格式为 [实部, 虚部]。
        """
        total_n, n_modes = phi_response.shape
        batch = batch.long()
        _, n_freq = frequencies_hz.shape
        omega_q = 2.0 * torch.pi * frequencies_hz

        frf_re = phi_response.new_zeros(total_n, n_freq)
        frf_im = phi_response.new_zeros(total_n, n_freq)

        for k in range(n_modes):
            wk = omega[:, k]
            zk = zeta[:, k]
            # 模态参与因子：每节点响应值 × 激励点激励方向值
            pk = phi_response[:, k] * phi_force_exc[batch, k]

            dw = wk.unsqueeze(1) ** 2 - omega_q ** 2
            gm = 2.0 * zk.unsqueeze(1) * wk.unsqueeze(1) * omega_q
            denom = dw ** 2 + gm ** 2 + 1e-6
            if self.clamp_denominator:
                denom = torch.clamp(denom, min=1.0)

            h_re = self.amp_scale * dw / denom
            h_im = -self.amp_scale * gm / denom
            frf_re = frf_re + pk.unsqueeze(-1) * h_re[batch]
            frf_im = frf_im + pk.unsqueeze(-1) * h_im[batch]

        return torch.stack([frf_re, frf_im], dim=-1)


# 向后兼容别名，供旧模块使用。
PhysicsDecoder = ModalFRFDecoder
