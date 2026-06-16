from .modal_losses_scaled import frequency_loss, modal_loss, weighted_phi_z_terms
from .modal_trainer_simple import evaluate_modal, run_epoch, train_modal

# Backward-compatible aliases for older sample scripts.
train = train_modal
evaluate = evaluate_modal

__all__ = [
    "frequency_loss",
    "modal_loss",
    "weighted_phi_z_terms",
    "train_modal",
    "evaluate_modal",
    "run_epoch",
    "train",
    "evaluate",
]
