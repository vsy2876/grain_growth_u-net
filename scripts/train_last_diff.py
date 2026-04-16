# train.py
# Channel-Stacked Video Diffusion — HPC Training Script
# Architecture: 1 frame in → 3 future frames out (one-shot, Δt=100 per frame)
# Optimised for H200 / H100 / A100 with bfloat16, torch.compile, TF32
#
# Changes vs previous version:
#   - NUM_PAST_FRAMES removed (hardcoded to 1 in model)
#   - Dataset simplified: anchor → [T+100, T+200, T+300]
#   - Adaptive SSIM weight schedule (0 → 0.15 over epochs 20-60)
#   - Anchor noise augmentation (50% chance, std 0-0.05) for rollout robustness
#   - gt_clean_images passed to loss for SSIM computation
#   - Updated loss call to match new 7-return signature

import os
os.environ["CC"]  = "gcc"
os.environ["CXX"] = "g++"

import re
import math
import random
import signal
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from last_diff import GrainDiffusionUNet, PhysicsInformedLoss, NoiseScheduler

# ==============================================================================
# CONFIG
# ==============================================================================
ROOT_DIR        = "/home/hice1/vyadav68/scratch/grain_growth/dataset/grain_images"
CHECKPOINT_DIR  = "./checkpoints_diff_last"
FINAL_MODEL_DIR = "./final_model_diff_last"
VIZ_DIR         = "./visualizations_diff_last"

IMAGE_SIZE      = 512
BATCH_SIZE      = 8
NUM_WORKERS     = 14
NUM_EPOCHS      = 80
SAVE_EVERY      = 5
LOG_EVERY       = 100
VISUALIZE_EVERY = 500

# One-shot multi-frame: predict T+100, T+200, T+300 from a single anchor frame
NUM_FUTURE_FRAMES = 3
DELTA_T           = 100          # physical timestep between consecutive frames

BASE_CHANNELS   = 64
CONTEXT_DIM     = 256
EMBED_DIM       = 128
PHYSICS_WEIGHT  = 0.05
LEARNING_RATE   = 1e-4
NUM_TIMESTEPS   = 1000

# SSIM loss schedule: ramp from 0 → SSIM_MAX_WEIGHT between epochs SSIM_WARMUP and SSIM_RAMP_END
SSIM_MAX_WEIGHT = 0.10
SSIM_WARMUP     = 50             # epoch at which SSIM loss starts
SSIM_RAMP_END   = 90             # epoch at which SSIM loss reaches max weight

RESUME_TRAINING = True

AMP_DTYPE = torch.bfloat16

# ==============================================================================
# H200 / H100 / A100 SPEED FLAGS
# ==============================================================================
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
mp.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device       : {device}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Total Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"AMP dtype    : {AMP_DTYPE}")

# ==============================================================================
# SSIM WEIGHT SCHEDULER
# ==============================================================================
def get_ssim_weight(epoch):
    """
    Returns the SSIM loss weight for the current epoch.

    Epoch 0  → SSIM_WARMUP:  0.0          (pure diffusion loss — stable early training)
    Epoch SSIM_WARMUP → SSIM_RAMP_END:    linear ramp 0 → SSIM_MAX_WEIGHT
    Epoch SSIM_RAMP_END+:  SSIM_MAX_WEIGHT (held constant)

    Rationale: introducing SSIM too early destabilises diffusion training because
    the denoised estimate x0 is very noisy at high diffusion timesteps.
    Starting from epoch 20 means the model has already learned reasonable structure.
    """
#    if epoch < SSIM_WARMUP:
#        return 0.0
#    elif epoch < SSIM_RAMP_END:
#        progress = (epoch - SSIM_WARMUP) / (SSIM_RAMP_END - SSIM_WARMUP)
#        return SSIM_MAX_WEIGHT * progress
#    else:
#        return SSIM_MAX_WEIGHT
    if epoch < 30:
        return 0.0
    elif epoch < 50:
        return 0.10 * (epoch - 30) / 20.0
    else:
        return 0.10

# ==============================================================================
# TRANSFORMS
# ==============================================================================
grain_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # → [-1, 1]
])

boundary_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# ==============================================================================
# DATASET
# ==============================================================================
def seed_worker(worker_id):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


class PottsDataset(Dataset):
    """
    Builds (anchor → future triplet) samples from SPPARKS grain growth images.

    For each run, for each valid anchor timestep T:
        Input:   T          (single anchor frame, 3 channels)
        Targets: T+100, T+200, T+300  (stacked, 9 channels)
        Boundaries: T+100, T+200, T+300 boundary maps (stacked, 3 channels)

    Valid anchor timesteps: 0, 100, 200, ..., 700  (T+300 must be ≤ 1000)
    This gives 8 samples per run × ~1000 runs = ~8000 total samples.
    """

    ANCHOR_TIMESTEPS = list(range(0, 701, DELTA_T))   # [0, 100, ..., 700]

    def __init__(self, root_dir, num_future_frames=NUM_FUTURE_FRAMES):
        self.root_dir         = root_dir
        self.num_future_frames = num_future_frames
        self.samples          = []

        # Discover all runs from timestep_0 directory
        t0_dir       = os.path.join(root_dir, "timestep_0")
        run_registry = {}
        for fname in sorted(os.listdir(t0_dir)):
            if not fname.endswith("_rgb.png"):
                continue
            m = re.match(r"run_(\d+)_temp_([\d.]+)_timestep_0_rgb\.png", fname)
            if m:
                run_registry[int(m.group(1))] = float(m.group(2))

        print(f"Runs discovered: {len(run_registry)}")

        for run_id, temperature in run_registry.items():
            for anchor_t in self.ANCHOR_TIMESTEPS:
                target_ts = [anchor_t + DELTA_T * (i + 1) for i in range(num_future_frames)]

                # All targets must be within the simulation range
                if target_ts[-1] > 1000:
                    continue

                def rgb_path(t):
                    return os.path.join(
                        root_dir, f"timestep_{t}",
                        f"run_{run_id}_temp_{temperature:.3f}_timestep_{t}_rgb.png"
                    )

                def bnd_path(t):
                    return os.path.join(
                        root_dir, f"timestep_{t}",
                        f"run_{run_id}_temp_{temperature:.3f}_timestep_{t}_boundary.png"
                    )

                anchor_p   = rgb_path(anchor_t)
                target_ps  = [rgb_path(t)  for t in target_ts]
                bnd_ps     = [bnd_path(t)  for t in target_ts]

                if not os.path.exists(anchor_p):
                    continue
                if not all(os.path.exists(p) for p in target_ps):
                    continue
                if not all(os.path.exists(p) for p in bnd_ps):
                    continue

                self.samples.append({
                    "anchor_path":         anchor_p,
                    "target_grain_paths":  target_ps,
                    "target_bnd_paths":    bnd_ps,
                    "temperature":         temperature,
                    "anchor_t":            anchor_t,
                    "target_ts":           target_ts,
                    # target_step = total physical jump from anchor to last target
                    "time_jump":           float(target_ts[-1] - anchor_t),
                })

        print(f"Valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Anchor frame (single frame, 3 channels)
        anchor = grain_transform(Image.open(s["anchor_path"]).convert("RGB"))

        # Target frames stacked → (F*3, H, W)
        target_tensors = [
            grain_transform(Image.open(p).convert("RGB"))
            for p in s["target_grain_paths"]
        ]
        target_stacked = torch.cat(target_tensors, dim=0)

        # Boundary maps stacked → (F, H, W)  — 1=boundary, 0=interior
        bnd_tensors = []
        for p in s["target_bnd_paths"]:
            b = boundary_transform(Image.open(p).convert("L"))
            bnd_tensors.append(1.0 - b)                    # invert: 1=boundary
        bnd_stacked = torch.cat(bnd_tensors, dim=0)        # (F, H, W)

        return {
            "anchor":          anchor,                      # (3, H, W)
            "target_stacked":  target_stacked,              # (F*3, H, W)
            "bnd_stacked":     bnd_stacked,                 # (F, H, W)
            "temperature":     torch.tensor(s["temperature"], dtype=torch.float32),
            "time_jump":       torch.tensor(s["time_jump"],   dtype=torch.float32),
        }

# ==============================================================================
# VISUALISATION
# ==============================================================================
def sobel_edges_np(tensor_img):
    gray   = tensor_img.mean(dim=0, keepdim=True).unsqueeze(0)
    dev    = tensor_img.device
    sx     = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=dev).view(1,1,3,3)
    sy     = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=dev).view(1,1,3,3)
    ex     = F.conv2d(gray, sx, padding=1)
    ey     = F.conv2d(gray, sy, padding=1)
    edges  = torch.sqrt(ex**2 + ey**2)
    return (edges / (edges.max() + 1e-6)).squeeze().cpu().numpy()


def visualize_predictions(anchor, target_stacked, pred_stacked,
                           bnd_stacked, pred_bnd_raw, epoch, batch_idx, viz_dir, time_jump):
    """Show the last frame in the predicted sequence for clarity."""

    # Last future frame (channels -3:)
    gt_last   = target_stacked[:, -3:, :, :]
    pred_last = pred_stacked[:,   -3:, :, :]
    bnd_last  = bnd_stacked[:,    -1:, :, :]       # (B, 1, H, W)
    raw_last  = pred_bnd_raw[0, -1, :, :].detach().cpu().float()
    pred_bnd_display = (raw_last / (raw_last.max() + 1e-6)).numpy()
    gt_edges  = sobel_edges_np(gt_last[0].detach().float().clamp(0, 1))
    t_val     = int(time_jump[0].item())

    def to_rgb(t):
        return t[0].detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()

    def to_gray(t):
        return t[0].detach().cpu().float().clamp(0, 1).squeeze(0).numpy()

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    fig.suptitle(
        f"Epoch {epoch+1} | Batch {batch_idx} | Δt total={t_val} (Last Frame = T+{t_val})",
        fontsize=14, fontweight='bold'
    )
    items = [
        ("Anchor Frame",          to_rgb(anchor),         None),
        (f"GT Last Frame",        to_rgb(gt_last),         None),
        (f"Pred Last Frame",      to_rgb(pred_last),       None),
        ("GT Boundary Map",       to_gray(bnd_last),       "gray"),
        ("Pred Boundary (Sobel)", pred_bnd_display,        "hot"),
        ("GT Boundary (Sobel)",   gt_edges,                "hot"),
    ]
    for ax, (title, img, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(viz_dir, f"viz_ep{epoch+1:04d}_b{batch_idx:05d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz] → {path}")

# ==============================================================================
# CHECKPOINT
# ==============================================================================
def save_checkpoint(epoch, model, optimizer, lr_scheduler, train_losses, val_losses, label="epoch"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{label}_{epoch:04d}.pt")
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }, path)
    print(f"  [Checkpoint] → {path}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    os.makedirs(CHECKPOINT_DIR,  exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR,         exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_ds  = PottsDataset(ROOT_DIR, num_future_frames=NUM_FUTURE_FRAMES)
    val_size = int(0.1 * len(full_ds))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])
    print(f"Train: {trn_size} | Val: {val_size}")

    loader_kwargs = dict(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2, worker_init_fn=seed_worker
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GrainDiffusionUNet(
        num_future_frames=NUM_FUTURE_FRAMES,
        base_channels=BASE_CHANNELS,
        context_dim=CONTEXT_DIM,
        embed_dim=EMBED_DIM,
    ).to(device)

    loss_fn      = PhysicsInformedLoss(physics_weight=PHYSICS_WEIGHT).to(device)
    scheduler    = NoiseScheduler(num_timesteps=NUM_TIMESTEPS)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    use_scaler = (AMP_DTYPE == torch.float16)
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    train_losses = []
    val_losses   = []

    if RESUME_TRAINING:
        ckpts = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
        if ckpts:
            latest = os.path.join(CHECKPOINT_DIR, ckpts[-1])
            ckpt   = torch.load(latest, map_location=device)
            # Strip torch.compile prefix if present
            state  = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
            model.load_state_dict(state)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            lr_scheduler.load_state_dict(ckpt["scheduler_state"])
            train_losses = ckpt.get("train_losses", [])
            val_losses   = ckpt.get("val_losses",   [])
            start_epoch  = ckpt["epoch"] + 1
            print(f"Resumed from {latest} (epoch {start_epoch})")

            # Reset LR scheduler for Phase 2 — cosine over the remaining 50 epochs
            # Don't use the loaded scheduler state, it's already at eta_min
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=1e-6
            )
            # Also warm up the LR slightly for Phase 2 fine-tuning
            for pg in optimizer.param_groups:
                pg['lr'] = 3e-5   # lower than Phase 1 (1e-4), but not dead

    model = torch.compile(model)

    # Warmup compile with a dummy forward pass
    print("Warming up torch.compile...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        _ = model(
            noisy_image   = torch.randn(2, NUM_FUTURE_FRAMES*3, IMAGE_SIZE, IMAGE_SIZE, device=device),
            initial_image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device),
            diff_step     = torch.randint(0, NUM_TIMESTEPS, (2,), device=device),
            target_step   = torch.full((2,), float(NUM_FUTURE_FRAMES * DELTA_T), device=device),
            temperature   = torch.full((2,), 0.5, device=device),
        )
    print("Warmup complete.")

    # ── Signal handler ────────────────────────────────────────────────────────
    interrupted = False
    def handle_signal(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n[SIGNAL {sig}] Saving emergency checkpoint...")
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):

        current_ssim_weight = get_ssim_weight(epoch)

        model.train()
        epoch_train_loss = 0.0
        stopped_early    = False

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Train]")):
            if interrupted:
                save_checkpoint(epoch, model, optimizer, lr_scheduler,
                                train_losses, val_losses, "emergency")
                stopped_early = True
                break

            anchor   = batch["anchor"].to(device)           # (B, 3, H, W)
            target   = batch["target_stacked"].to(device)   # (B, F*3, H, W)
            bnd      = batch["bnd_stacked"].to(device)      # (B, F, H, W)
            temp     = batch["temperature"].to(device)
            t_jump   = batch["time_jump"].to(device)

            # ── Anchor noise augmentation ─────────────────────────────────
            # 50% of batches: add small Gaussian noise to the anchor frame.
            # Trains the model to tolerate slightly imperfect inputs,
            # which improves robustness during any autoregressive use.
            if random.random() < 0.5:
                std    = random.uniform(0.0, 0.05)
                anchor = (anchor + torch.randn_like(anchor) * std).clamp(-1, 1)

            B         = target.size(0)
            diff_step = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)

            noisy_target, actual_noise = scheduler.add_noise(target, diff_step, device)

            # ── Forward ───────────────────────────────────────────────────
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                predicted_noise = model(
                    noisy_image   = noisy_target,
                    initial_image = anchor,
                    diff_step     = diff_step,
                    target_step   = t_jump,
                    temperature   = temp,
                )
                
                # 1. Compute the raw denoised image
                pred_clean_raw = scheduler.remove_noise(noisy_target, predicted_noise, diff_step, device)
                
                # 2. THE GRADIENT GATE
                # Isolate the images where t >= 300 and sever their gradients to protect the physics loss
                grad_mask = (diff_step < 300).float().view(-1, 1, 1, 1)
                pred_clean_safe = (pred_clean_raw * grad_mask) + (pred_clean_raw.detach() * (1.0 - grad_mask))

                # 3. Compute loss using the gradient-safe tensor
                (loss, diff_l, bound_l, energy_l,
                temporal_l, ssim_l, pred_bnd_raw) = loss_fn(
                    pred_noise         = predicted_noise,
                    actual_noise       = actual_noise,
                    pred_clean_images  = pred_clean_safe,    # <--- Feed the safe tensor here!
                    gt_clean_images    = target,
                    gt_boundary_images = bnd,
                    diff_step          = diff_step,
                    ssim_weight        = current_ssim_weight, 
                )
                
                # loss    = torch.nn.functional.mse_loss(predicted_noise, actual_noise)
                # diff_l  = loss
                # bound_l = torch.tensor(0.0, device=device)
                # energy_l= torch.tensor(0.0, device=device)
                # temporal_l = torch.tensor(0.0, device=device)
                # ssim_l  = torch.tensor(0.0, device=device)

            # ── Backward ──────────────────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

            if batch_idx % LOG_EVERY == 0:
                print(
                    f"  Ep{epoch+1} B{batch_idx:04d} | "
                    f"Total:{loss.item():.4f} | "
                    f"Diff:{diff_l.item():.4f} | "
                    f"Bound:{bound_l.item():.4f} | "
                    f"Energy:{energy_l.item():.4f} | "
                    f"Temporal:{temporal_l.item():.4f} | "
                    f"SSIM_loss:{ssim_l.item():.4f} (w={current_ssim_weight:.3f})"
                )

            if batch_idx % VISUALIZE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    visualize_predictions(
                        anchor, target, pred_clean_safe,
                        bnd, pred_bnd_raw,
                        epoch, batch_idx, VIZ_DIR, t_jump
                    )
                model.train()

        if stopped_early:
            print("Training interrupted. Exiting.")
            break

        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        epoch_val_loss = 0.0
        val_stopped    = False

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val]"):
                if interrupted:
                    save_checkpoint(epoch, model, optimizer, lr_scheduler,
                                    train_losses, val_losses, "emergency")
                    val_stopped = True
                    break

                anchor = batch["anchor"].to(device)
                target = batch["target_stacked"].to(device)
                bnd    = batch["bnd_stacked"].to(device)
                temp   = batch["temperature"].to(device)
                t_jump = batch["time_jump"].to(device)

                B         = target.size(0)
                diff_step = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)
                noisy_target, actual_noise = scheduler.add_noise(target, diff_step, device)

                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    predicted_noise = model(
                        noisy_image   = noisy_target,
                        initial_image = anchor,
                        diff_step     = diff_step,
                        target_step   = t_jump,
                        temperature   = temp,
                    )
                    
                    # 1. Compute raw denoised image
                    pred_clean_raw = scheduler.remove_noise(noisy_target, predicted_noise, diff_step, device)
                    
                    # 2. Apply the exact same mask so val loss metrics match training loss metrics
                    grad_mask = (diff_step < 300).float().view(-1, 1, 1, 1)
                    pred_clean_safe = (pred_clean_raw * grad_mask) + (pred_clean_raw.detach() * (1.0 - grad_mask))

                    # 3. Compute metric using the safe tensor
                    (loss, *_) = loss_fn(
                        pred_noise         = predicted_noise,
                        actual_noise       = actual_noise,
                        pred_clean_images  = pred_clean_safe,
                        gt_clean_images    = target,
                        gt_boundary_images = bnd,
                        diff_step          = diff_step,
                        ssim_weight        = 0.0,  # Keep SSIM off for validation
                    )

                    # loss    = torch.nn.functional.mse_loss(predicted_noise, actual_noise)
                    # diff_l  = loss
                    # bound_l = torch.tensor(0.0, device=device)
                    # energy_l= torch.tensor(0.0, device=device)
                    # temporal_l = torch.tensor(0.0, device=device)
                    # ssim_l  = torch.tensor(0.0, device=device)
                    # epoch_val_loss += loss.item()

        if val_stopped:
            print("Interrupted during validation. Exiting.")
            break

        avg_val = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val)
        lr_scheduler.step()

        print(
            f"\nEpoch {epoch+1:03d} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"LR: {lr_scheduler.get_last_lr()[0]:.6f} | "
            f"SSIM weight: {current_ssim_weight:.3f}\n"
        )

        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(epoch + 1, model, optimizer, lr_scheduler, train_losses, val_losses)

    # ── Loss Curve ────────────────────────────────────────────────────────────
    n = min(len(train_losses), len(val_losses))
    if n > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses[:n], label="Train")
        plt.plot(val_losses[:n],   label="Val")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend(); plt.grid(True); plt.tight_layout()
        curve_path = os.path.join(FINAL_MODEL_DIR, "loss_curve.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss curve → {curve_path}")

    # ── Save Final Model ──────────────────────────────────────────────────────
    final_path = os.path.join(FINAL_MODEL_DIR, "grain_diffusion_final.pt")
    torch.save({
        "epoch":           NUM_EPOCHS,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "hyperparameters": {
            "num_future_frames": NUM_FUTURE_FRAMES,
            "delta_t":           DELTA_T,
            "base_channels":     BASE_CHANNELS,
            "context_dim":       CONTEXT_DIM,
            "embed_dim":         EMBED_DIM,
            "physics_weight":    PHYSICS_WEIGHT,
            "ssim_max_weight":   SSIM_MAX_WEIGHT,
            "ssim_warmup_epoch": SSIM_WARMUP,
            "ssim_ramp_end":     SSIM_RAMP_END,
            "learning_rate":     LEARNING_RATE,
            "num_timesteps":     NUM_TIMESTEPS,
            "image_size":        IMAGE_SIZE,
            "batch_size":        BATCH_SIZE,
            "num_epochs":        NUM_EPOCHS,
            "amp_dtype":         str(AMP_DTYPE),
        }
    }, final_path)
    print(f"Final model → {final_path}")

    weights_path = os.path.join(FINAL_MODEL_DIR, "grain_diffusion_weights_only.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights only → {weights_path}")


if __name__ == "__main__":
    main()