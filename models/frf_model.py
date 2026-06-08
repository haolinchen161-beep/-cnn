"""
frf_model.py — 模型构建入口。

当前主模型已从 2.5D CNN-UNet 切换为 MeshGraphNet/GNN。
旧 UNetPhysicsModel 代码保留用于历史对比，但 build_geometric_model 默认不再构建 CNN。
"""

from .meshgraphnet_frf_model import MeshGraphFRFModel


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    """构建 GNN/MeshGraphNet 模态 FRF 代理模型。"""
    kwargs = {}
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return MeshGraphFRFModel(**kwargs)
