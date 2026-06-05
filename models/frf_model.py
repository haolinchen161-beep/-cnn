"""
frf_model.py — 模型构建入口。
"""
from .unet_physics_model import UNetPhysicsModel


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    kwargs = {}
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return UNetPhysicsModel(**kwargs)
