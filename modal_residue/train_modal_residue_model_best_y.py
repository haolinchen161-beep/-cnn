from __future__ import annotations

"""Entry wrapper that selects the best checkpoint by residue asinh validation loss.

The base trainer already computes the right validation quantities for the
signed-asinh residue target.  This wrapper changes only the checkpoint score:

    score = val_Y_rms + 0.05 * val_w10_rms + 0.001 * val_A_vis_mean

This prevents best_model.pt from being selected mainly by frequency error or
by unstable physical-space percentage errors.
"""

from typing import Dict, Tuple

import modal_residue.train_modal_residue_model as _base


def modal_score(metrics: Dict[str, Tuple[float, float, float]], best_a_weight: float) -> float:
    """Residue-first validation score for signed-asinh training.

    The original base score was mostly frequency-driven.  For the current
    residue target, choose the checkpoint primarily by Y=asinh(A/s_mode)
    validation RMS, with small penalties for frequency and physical A error.
    """
    w_rms = float(metrics.get("w10_triplet", (0.0, 0.0, 0.0))[2])
    a_vis_mean = float(metrics.get("A_vis_triplet", (0.0, 0.0, 0.0))[0])
    y_triplet = metrics.get("Y_smooth_l1_triplet")
    if y_triplet is None:
        # Fallback for older metric dictionaries that do not contain Y loss.
        a_vis_rms = float(metrics.get("A_vis_triplet", (0.0, 0.0, 0.0))[2])
        return float(w_rms + float(best_a_weight) * a_vis_rms)
    y_rms = float(y_triplet[2])
    return float(y_rms + 0.05 * w_rms + 0.001 * a_vis_mean)


_base.modal_score = modal_score
main = _base.main


if __name__ == "__main__":
    main()
