"""Model construction entrypoint for the FEM-aware MeshGraphNet branch."""

from .meshgraphnet_frf_model import MeshGraphFRFModel


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    kwargs = {}
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return MeshGraphFRFModel(**kwargs)
