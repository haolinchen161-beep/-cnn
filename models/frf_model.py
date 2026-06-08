"""Model factory for the Transolver modal-FRF network."""
from .transolver_modal_model import TransolverModalFRF


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    kwargs = {}
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return TransolverModalFRF(**kwargs)
