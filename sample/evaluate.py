from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from data.dataset import TransolverModalDataset, collate_mesh_batch
from models import build_geometric_model
from training.trainer import evaluate

CONFIG = {
    'frf_loss_weight': 0.02,
    'omega_loss_weight': 1.0,
    'zeta_loss_weight': 10.0,
    'phi_loss_weight': 3.0,
}

class Args:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Transolver Modal')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'ansys', 'data'))
    parser.add_argument('--split', default='test.h5')
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), 'output_transolver_modal'))
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--slices', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no-edges', action='store_true')
    parser.add_argument('--response-dir', default='Z', choices=['X', 'Y', 'Z'])
    parser.add_argument('--force-dir', default='Z', choices=['X', 'Y', 'Z'])
    return parser.parse_args()

def main():
    cli = parse_args()
    ckpt_path = cli.checkpoint or os.path.join(cli.output_dir, 'checkpoint_best_modal')

    dataset = TransolverModalDataset([cli.split], data_dir=cli.data_dir, use_edges=not cli.no_edges, require_frf=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mesh_batch,
    )

    first = dataset[0]
    model = build_geometric_model({
        'in_dim': first['node_features'].shape[1],
        'hidden_dim': cli.hidden_dim,
        'n_layers': cli.layers,
        'n_heads': cli.heads,
        'n_slices': cli.slices,
        'dropout': cli.dropout,
        'n_modes': first['modal_omega'].shape[0],
        'use_edge_stem': not cli.no_edges,
        'amp_scale': 500000.0,
        'response_direction': cli.response_dir,
        'force_direction': cli.force_dir,
    }, {}).to(cli.device)

    ckpt = torch.load(ckpt_path, map_location=cli.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    args = Args()
    args.device = cli.device
    args.dir = cli.output_dir
    args.fp16 = False

    print('=' * 72)
    print('Transolver Modal evaluation')
    print('=' * 72)
    print('Checkpoint:', ckpt_path)
    print('Epoch:', ckpt.get('epoch', 'NA'), 'Loss:', ckpt.get('loss', -1))
    print('Params:', sum(p.numel() for p in model.parameters()))
    print('Samples:', len(dataset))

    metrics = evaluate(args, CONFIG, model, loader, verbose=True, phase1=False)
    print('\nSummary:')
    for key, value in metrics.items():
        print(key, ':', value)

if __name__ == '__main__':
    main()
