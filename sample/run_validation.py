"""
UNetPhysicsModel 模态参数预测训练 — ANSYS 凹槽工件 (2.5D CNN).
用法: F:\pytorch_cuda12\python.exe sample/run_validation.py
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore', message='Detected call of')
warnings.filterwarnings('ignore', message='To get the last learning rate')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from models import build_geometric_model
from training import train, evaluate, modal_loss

CONFIG = {
    'epochs': 2000,
    'validation_frequency': 5,

    # 两阶段
    'phase1_epochs': 300,    # CNN收敛快, 300轮够
    'frf_loss_weight': 50.0,
    'zeta_loss_weight': 0.0,

    'freq_min': 1.0, 'freq_max': 5000.0,
    'data_path_train': ['train.h5'],
    'data_path_val': ['val.h5'],
    'data_path_test': ['test.h5'],

    'augmentation': {
        'enabled': False,  # 图像数据暂不增强
    },

    'optimizer': {
        'name': 'AdamW',
        'kwargs': {'lr': 0.001, 'weight_decay': 0.0001, 'betas': (0.9, 0.999)},
        'gradient_clip': 2.0,
        'gradient_clip_transolver': 3.0,
        'gradient_clip_head_phi': 5.0,
        'gradient_clip_modal': 2.0,
    },
}

MODEL_CFG = {
    'encoder_kwargs': {
        'in_ch': 6, 'hidden': 512, 'n_modes': 3,
        'amp_scale': 500000.0, 'freq_min': 1.0, 'freq_max': 5000.0,
    },
    'decoder_kwargs': {},
}


class SimpleArgs:
    def __init__(self):
        self.batch_size = 8  # CNN 显存友好, 可增大batch
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fp16 = False
        self.dir = os.path.join(os.path.dirname(__file__), "output")
        self.debug = False


def main():
    print("=" * 60)
    print("UNetPhysicsModel (2.5D CNN) — ANSYS 3D 凹槽工件")
    print("=" * 60)
    args = SimpleArgs()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data")
    print(f"Device: {args.device}, Batch: {args.batch_size}")

    # 数据
    print("\n--- Step 1: DataLoader ---")
    from data.dataset import GeometricHDF5Dataset, collate_geometry_batch
    trainset = GeometricHDF5Dataset(['train.h5'], CONFIG, data_dir=data_dir, normalization=True, test=False)
    valset = GeometricHDF5Dataset(['val.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_geometry_batch, generator=gen)
    valloader = torch.utils.data.DataLoader(valset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=0, collate_fn=collate_geometry_batch)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=0, collate_fn=collate_geometry_batch)

    batch = next(iter(trainloader))
    print(f"  Train: {len(trainset)} samples, {len(trainloader)} batches")
    print(f"  Image: {batch['image_tensor'].shape}, coords: {batch['query_coords'].shape}")

    # 模型
    print("\n--- Step 2: Model ---")
    net = build_geometric_model(MODEL_CFG['encoder_kwargs'], MODEL_CFG['decoder_kwargs']).to(args.device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Params: {total_params:,}")

    # 前向测试
    print("\n--- Step 3: Forward test ---")
    net.eval()
    with torch.no_grad():
        img = batch['image_tensor'].to(args.device)
        coords = batch['query_coords'].to(args.device)
        batch_idx = batch['batch'].to(args.device)
        phi_exc = batch.get('modal_phi_exc')
        phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
        frf_p, op, zp, pp = net(img, coords, batch['frequencies'].to(args.device), phi_exc, batch_idx)
    print(f"  FRF={list(frf_p.shape)}, omega={list(op.shape)}, phi={list(pp.shape)}")

    # 初始Loss
    print("\n--- Step 4: Initial Loss ---")
    with torch.no_grad():
        init_loss, _, _, _ = modal_loss(op, batch['modal_omega_norm'].to(args.device),
            zp, batch['modal_zeta'].to(args.device),
            pp, batch['modal_phi'].to(args.device), batch_idx=batch_idx)
    print(f"  Init loss: {init_loss.item():.0f}")

    # 训练
    print("\n--- Step 5: Train ---")
    optimizer = torch.optim.AdamW(net.parameters(), **CONFIG['optimizer']['kwargs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    start_epoch = 0
    ckpt_path = os.path.join(args.dir, "checkpoint_last")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resume from epoch {start_epoch}")

    print(f"  Training {CONFIG['epochs']} epochs...")
    t0 = time.time()
    net = train(args, CONFIG, MODEL_CFG, net, trainloader, optimizer, valloader, scheduler, logger=None, start_epoch=start_epoch)
    elapsed = time.time() - t0
    print(f"  Done, {elapsed:.0f}s")

    # 验证
    print("\n--- Step 6: Evaluate ---")
    best_path = os.path.join(args.dir, "checkpoint_best")
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=args.device)["model_state_dict"])
    results = evaluate(args, CONFIG, net, testloader, verbose=True)
    print(f"\n{'='*60}")
    print(f"Done | Device:{args.device} | Params:{total_params:,} | Time:{elapsed:.0f}s")
    print(f"Test MSE:{results.get('loss (asinh-MSE)', -1):.4f}")
    return 0


if __name__ == '__main__':
    exit(main())
