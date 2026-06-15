"""Model construction entrypoint for the FEM-aware MeshGraphNet branch."""

from .meshgraphnet_frf_model import MeshGraphFRFModel


DEFAULT_NODE_FEATURE_DIM = 26
DEFAULT_EDGE_FEATURE_DIM = 4


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    kwargs = {
        "node_in_dim": DEFAULT_NODE_FEATURE_DIM,
        "edge_in_dim": DEFAULT_EDGE_FEATURE_DIM,
    }
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return MeshGraphFRFModel(**kwargs)
