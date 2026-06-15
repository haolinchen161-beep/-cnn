from .modal_meshgraphnet import MeshModalNet


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    kwargs = {}
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return MeshModalNet(**kwargs)
