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

    # 阶段控制
    'enable_phase2': True,           # 开启 FRF 联合训练
    'phase2_min_epoch': 200,         # 200 轮模态预训练后进 Phase2
    'zeta_warmup_epochs': 40,        # 前40轮 zeta_w=0 (防 spike)

    # 模态损失权重 (trainer 内部: omega 在 Hz 空间, zeta 在 log 空间, phi 归一化后 MSE+MAC+std)
    'omega_loss_weight': 1.0,        # 频率损失权重 (Hz-space smooth_l1)
    'zeta_loss_weight': 10.0,        # 阻尼损失权重 (log-space smooth_l1)
    'phi_loss_weight': 3.0,          # 振型损失权重 (归一化 MSE + MAC + std)

    # FRF 弱约束: dB空间 MSE×0.5 ≈ 10-20 损失贡献 (总损失 ~200-400 的 5-10%)
    'frf_loss_weight': 0.5,
    'frf_warmup_epochs': 50,
    'teacher_anneal_epochs': 200,       # Teacher-Forced ω 退火周期: α 1.0→0.0

    'freq_min': 1.0, 'freq_max': 5000.0,
    'data_path_train': ['train.h5'],
    'data_path_val': ['val.h5'],
    'data_path_test': ['test.h5'],

    'augmentation': {
        'enabled': False,
    },

    'optimizer': {
        'name': 'AdamW',
        'kwargs': {'lr': 0.001, 'weight_decay': 0.001, 'betas': (0.9, 0.999)},
        'gradient_clip': 2.0,
    },
}

MODEL_CFG = {
    'encoder_kwargs': {
        'in_ch': 6, 'hidden': 768, 'n_modes': 3,
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
    data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data_2")
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
        nx = batch.get('node_xyz'); nf = batch.get('node_features')
        nx = nx.to(args.device) if nx is not None else None
        nf = nf.to(args.device) if nf is not None else None
        phi_exc = batch.get('modal_phi_exc')
        phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
        frf_p, omega_p, log_z, zeta_p, phi_p = net(
            img, coords, batch['frequencies'].to(args.device), phi_exc, batch_idx,
            node_xyz=nx, node_features=nf)
    print(f"  FRF={list(frf_p.shape)}, omega_phys={list(omega_p.shape)}, phi={list(phi_p.shape)}")
    print(f"  omega_phys[0] rad/s: {omega_p[0].tolist()}")
    print(f"  freq_hz[0]: {[f'{w/(2*torch.pi):.1f}' for w in omega_p[0].tolist()]}")

    # 初始Loss
    print("\n--- Step 4: Initial Loss ---")
    with torch.no_grad():
        init_loss, l_w, l_z, l_p, mac_val = modal_loss(
            omega_p, batch['modal_omega_phys'].to(args.device),
            log_z, batch['modal_zeta'].to(args.device),
            phi_p, batch['modal_phi'].to(args.device),
            batch_idx=batch_idx,
            omega_weight=1.0, zeta_weight=0.0, phi_weight=3.0)
    mac_str = '/'.join(f'{x:.3f}' for x in mac_val.tolist())
    print(f"  Init loss: {init_loss.item():.0f} MAC=[{mac_str}]")
    print(f"  ω pred[0] Hz: {[f'{x/(2*torch.pi):.0f}' for x in omega_p[0].tolist()]}")
    print(f"  ω true[0] Hz: {[f'{x/(2*torch.pi):.0f}' for x in batch['modal_omega_phys'][0].tolist()]}")

    # 训练
    print("\n--- Step 5: Train ---")
    optimizer = torch.optim.AdamW(net.parameters(), **CONFIG['optimizer']['kwargs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=1, eta_min=1e-6)
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
    print(f"Test MSE:{results.get('loss (MSE)', -1):.4f}")
    return 0


if __name__ == '__main__':
    exit(main())
