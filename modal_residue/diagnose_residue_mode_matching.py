from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow both:
#   python -m modal_residue.diagnose_residue_mode_matching
# and:
#   python modal_residue/diagnose_residue_mode_matching.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from modal_residue import train_modal_residue_model as base
from modal_residue import train_modal_residue_bottom_model as bottom


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _linear_assignment_max(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Maximize score[row, col]. Uses scipy if available, greedy fallback otherwise."""
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-score)
        return row.astype(np.int64), col.astype(np.int64)
    except Exception:
        score_work = np.asarray(score, dtype=np.float64).copy()
        n_row, n_col = score_work.shape
        used_r = set()
        used_c = set()
        rows = []
        cols = []
        for _ in range(min(n_row, n_col)):
            best = None
            best_val = -np.inf
            for r in range(n_row):
                if r in used_r:
                    continue
                for c in range(n_col):
                    if c in used_c:
                        continue
                    val = score_work[r, c]
                    if val > best_val:
                        best_val = val
                        best = (r, c)
            if best is None:
                break
            r, c = best
            used_r.add(r)
            used_c.add(c)
            rows.append(r)
            cols.append(c)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def residue_cosine_matrix(A_pred: np.ndarray, A_true: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """
    A_pred: [nodes, modes_pred]
    A_true: [nodes, modes_true]
    Return signed cosine similarity matrix [modes_pred, modes_true].
    """
    P = np.asarray(A_pred, dtype=np.float64)
    T = np.asarray(A_true, dtype=np.float64)
    Pn = np.sqrt(np.sum(P * P, axis=0) + eps)
    Tn = np.sqrt(np.sum(T * T, axis=0) + eps)
    return (P.T @ T) / (Pn[:, None] * Tn[None, :])


def build_model_from_ckpt(ckpt: Dict, device: torch.device):
    ckpt_args = ckpt.get("args", {}) or {}
    hidden = _as_int(ckpt_args.get("hidden", 64), 64)
    gnn_layers = _as_int(ckpt_args.get("gnn_layers", 2), 2)
    model = base.MeshGraphModalResidueNet(
        int(ckpt["node_in_dim"]),
        int(ckpt["edge_in_dim"]),
        int(ckpt["n_modes"]),
        hidden=hidden,
        gnn_layers=gnn_layers,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def diagnose_split(args, model, stats: Dict[str, np.ndarray], device: torch.device) -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray]]:
    ds = bottom.H5Split(args.data_dir, args.split)
    if args.max_samples > 0:
        ds.keys = ds.keys[: min(args.max_samples, len(ds.keys))]

    rows: List[Dict[str, float]] = []
    n_modes = int(stats["omega_log_mean"].shape[0])
    pair_counts = np.zeros((n_modes, n_modes), dtype=np.int64)  # pred row -> true col
    diag_abs_all = []
    match_abs_all = []
    diag_signed_all = []
    match_signed_all = []
    freq_rel_diag_all = []
    freq_rel_match_all = []

    for i in range(len(ds)):
        t = bottom.to_tensors(ds[i], stats, device)
        q = bottom.select_query(t, args.eval_query_nodes, args.target_region, random_sample=False)
        if q.numel() < args.min_nodes:
            continue

        omega_n, Y_n = bottom.forward_model(model, t, q)
        omega_pred, A_pred_t = base.denorm_outputs(omega_n, Y_n, stats, clamp=args.asinh_clamp)
        A_true_t = t["A"][q].float()

        A_pred = A_pred_t.detach().cpu().numpy()
        A_true = A_true_t.detach().cpu().numpy()
        omega_p = omega_pred.detach().cpu().numpy().reshape(-1)
        omega_t = t["omega"].detach().cpu().numpy().reshape(-1)

        signed_cos = residue_cosine_matrix(A_pred, A_true)
        abs_cos = np.abs(signed_cos)

        freq_rel = np.abs(omega_p[:, None] - omega_t[None, :]) / np.maximum(np.abs(omega_t[None, :]), 1e-12)
        if args.freq_match_weight > 0:
            score = abs_cos - float(args.freq_match_weight) * np.minimum(freq_rel, float(args.freq_match_clip))
        else:
            score = abs_cos
        row_ind, col_ind = _linear_assignment_max(score)

        assign = np.full(n_modes, -1, dtype=np.int64)
        for r, c in zip(row_ind, col_ind):
            if 0 <= r < n_modes and 0 <= c < n_modes:
                assign[r] = c
                pair_counts[r, c] += 1

        diag_abs = np.diag(abs_cos)
        diag_signed = np.diag(signed_cos)
        match_abs = np.full(n_modes, np.nan, dtype=np.float64)
        match_signed = np.full(n_modes, np.nan, dtype=np.float64)
        freq_rel_diag = np.diag(freq_rel)
        freq_rel_match = np.full(n_modes, np.nan, dtype=np.float64)
        for r in range(n_modes):
            c = assign[r]
            if c >= 0:
                match_abs[r] = abs_cos[r, c]
                match_signed[r] = signed_cos[r, c]
                freq_rel_match[r] = freq_rel[r, c]

        diag_abs_all.append(diag_abs)
        match_abs_all.append(match_abs)
        diag_signed_all.append(diag_signed)
        match_signed_all.append(match_signed)
        freq_rel_diag_all.append(freq_rel_diag)
        freq_rel_match_all.append(freq_rel_match)

        sample_row: Dict[str, float] = {
            "sample": i,
            "n_target_nodes": int(q.numel()),
            "diag_abs_mean": float(np.nanmean(diag_abs)),
            "matched_abs_mean": float(np.nanmean(match_abs)),
            "match_gain_mean": float(np.nanmean(match_abs - diag_abs)),
            "diag_signed_mean": float(np.nanmean(diag_signed)),
            "matched_signed_mean": float(np.nanmean(match_signed)),
            "n_swapped_pred_channels": int(np.sum(assign != np.arange(n_modes))),
            "assignment_pred_to_true_1based": " ".join(str(int(c + 1)) if c >= 0 else "0" for c in assign),
        }
        for r in range(n_modes):
            sample_row[f"diag_abs_m{r+1:02d}"] = float(diag_abs[r])
            sample_row[f"matched_abs_m{r+1:02d}"] = float(match_abs[r])
            sample_row[f"diag_signed_m{r+1:02d}"] = float(diag_signed[r])
            sample_row[f"matched_signed_m{r+1:02d}"] = float(match_signed[r])
            sample_row[f"matched_true_m{r+1:02d}"] = int(assign[r] + 1) if assign[r] >= 0 else 0
            sample_row[f"freq_rel_diag_m{r+1:02d}_pct"] = float(freq_rel_diag[r] * 100.0)
            sample_row[f"freq_rel_match_m{r+1:02d}_pct"] = float(freq_rel_match[r] * 100.0)
        rows.append(sample_row)

    arrays = {
        "pair_counts": pair_counts,
        "diag_abs": np.asarray(diag_abs_all, dtype=np.float64),
        "match_abs": np.asarray(match_abs_all, dtype=np.float64),
        "diag_signed": np.asarray(diag_signed_all, dtype=np.float64),
        "match_signed": np.asarray(match_signed_all, dtype=np.float64),
        "freq_rel_diag": np.asarray(freq_rel_diag_all, dtype=np.float64),
        "freq_rel_match": np.asarray(freq_rel_match_all, dtype=np.float64),
    }
    return rows, arrays


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_pair_count_csv(path: Path, pair_counts: np.ndarray) -> None:
    rows = []
    n_modes = pair_counts.shape[0]
    for r in range(n_modes):
        row = {"pred_mode": r + 1}
        for c in range(n_modes):
            row[f"true_mode_{c+1:02d}"] = int(pair_counts[r, c])
        rows.append(row)
    write_csv(path, rows)


def make_summary(arrays: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    pair_counts = arrays["pair_counts"]
    n_modes = pair_counts.shape[0]
    diag_abs = arrays["diag_abs"]
    match_abs = arrays["match_abs"]
    diag_signed = arrays["diag_signed"]
    match_signed = arrays["match_signed"]
    freq_rel_diag = arrays["freq_rel_diag"]
    freq_rel_match = arrays["freq_rel_match"]
    rows = []
    n_samples = max(1, diag_abs.shape[0])
    for r in range(n_modes):
        most_common_true = int(np.argmax(pair_counts[r]) + 1) if pair_counts.size else 0
        fixed_count = int(pair_counts[r, r]) if r < pair_counts.shape[1] else 0
        rows.append({
            "pred_mode": r + 1,
            "fixed_abs_mean": float(np.nanmean(diag_abs[:, r])) if diag_abs.size else float("nan"),
            "matched_abs_mean": float(np.nanmean(match_abs[:, r])) if match_abs.size else float("nan"),
            "match_gain_mean": float(np.nanmean(match_abs[:, r] - diag_abs[:, r])) if match_abs.size else float("nan"),
            "fixed_signed_mean": float(np.nanmean(diag_signed[:, r])) if diag_signed.size else float("nan"),
            "matched_signed_mean": float(np.nanmean(match_signed[:, r])) if match_signed.size else float("nan"),
            "diag_sign_negative_rate_pct": float(np.nanmean(diag_signed[:, r] < 0.0) * 100.0) if diag_signed.size else float("nan"),
            "swap_rate_pct": float((1.0 - fixed_count / n_samples) * 100.0),
            "most_common_matched_true_mode": most_common_true,
            "most_common_count": int(np.max(pair_counts[r])) if pair_counts.size else 0,
            "freq_rel_diag_mean_pct": float(np.nanmean(freq_rel_diag[:, r]) * 100.0) if freq_rel_diag.size else float("nan"),
            "freq_rel_match_mean_pct": float(np.nanmean(freq_rel_match[:, r]) * 100.0) if freq_rel_match.size else float("nan"),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("modal_residue/data_modal_residue_fixedclamp300"))
    p.add_argument("--checkpoint", type=Path, default=Path("runs/modal_residue_bottom_asinh_fixedclamp300/best_model.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/modal_residue_bottom_asinh_fixedclamp300/mode_matching_diagnostic"))
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--target-region", choices=["bottom", "all"], default="bottom")
    p.add_argument("--eval-query-nodes", type=int, default=0, help="0 means all target-region nodes.")
    p.add_argument("--min-nodes", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--asinh-clamp", type=float, default=20.0)
    p.add_argument("--freq-match-weight", type=float, default=0.0, help="Optional penalty for frequency mismatch in assignment score.")
    p.add_argument("--freq-match-clip", type=float, default=0.20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    stats = ckpt["stats"]
    model = build_model_from_ckpt(ckpt, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, arrays = diagnose_split(args, model, stats, device)
    write_csv(args.out_dir / f"{args.split}_residue_matching_samples.csv", rows)
    write_csv(args.out_dir / f"{args.split}_residue_matching_summary.csv", make_summary(arrays))
    write_pair_count_csv(args.out_dir / f"{args.split}_residue_matching_pair_counts.csv", arrays["pair_counts"])

    global_summary = {
        "split": args.split,
        "n_samples_used": len(rows),
        "target_region": args.target_region,
        "eval_query_nodes": args.eval_query_nodes,
        "diag_abs_mean": float(np.nanmean(arrays["diag_abs"])) if arrays["diag_abs"].size else None,
        "matched_abs_mean": float(np.nanmean(arrays["match_abs"])) if arrays["match_abs"].size else None,
        "match_gain_mean": float(np.nanmean(arrays["match_abs"] - arrays["diag_abs"])) if arrays["match_abs"].size else None,
        "diag_signed_mean": float(np.nanmean(arrays["diag_signed"])) if arrays["diag_signed"].size else None,
        "matched_signed_mean": float(np.nanmean(arrays["match_signed"])) if arrays["match_signed"].size else None,
    }
    with open(args.out_dir / f"{args.split}_residue_matching_global_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(global_summary, indent=2, ensure_ascii=False))
    print(f"Saved diagnostic files to: {args.out_dir}")


if __name__ == "__main__":
    main()
