# train_autoregressive.py
# Grain Growth Conditional Diffusion Model — Autoregressive Retraining
# Key change: Dataset filtered to time_jump=100 only for autoregressive specialization

import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
import re
import math
import itertools
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

from model import GrainDiffusionUNet, PhysicsInformedLoss, NoiseScheduler


# ==============================================================================
# CONFIG
# ==============================================================================
ROOT_DIR        = "grain_images"
CHECKPOINT_DIR  = "./checkpoints_2"        # separate from original checkpoints
FINAL_MODEL_DIR = "./final_model_2"
VIZ_DIR         = "./visualizations_2"

IMAGE_SIZE      = 512
BATCH_SIZE      = 32
NUM_WORKERS     = 14
NUM_EPOCHS      = 100
SAVE_EVERY      = 5
LOG_EVERY       = 100
VISUALIZE_EVERY = 500

BASE_CHANNELS   = 64
CONTEXT_DIM     = 256
EMBED_DIM       = 128
PHYSICS_WEIGHT  = 0.1
LEARNING_RATE   = 1e-4
NUM_TIMESTEPS   = 1000

# KEY CHANGE: fixed time jump for autoregressive training
TIME_JUMP       = 100

# Set to path of your best existing checkpoint to finetune instead of training from scratch
# e.g. "./final_model/grain_diffusion_final.pt" or "./checkpoints/checkpoint_epoch_0100.pt"
# Set to None to train from scratch
FINETUNE_FROM   = "./final_model_1/grain_diffusion_final.pt"

RESUME_TRAINING = False  # Set True to resume from CHECKPOINT_DIR checkpoints

AMP_DTYPE       = torch.bfloat16


# ==============================================================================
# SPEED FLAGS
# ==============================================================================
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True
mp.set_sharing_strategy('file_system')


# ==============================================================================
# DEVICE
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device       : {device}")
if torch.cuda.is_available():
    print(f"GPU Name     : {torch.cuda.get_device_name(0)}")
    print(f"Total Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"AMP dtype    : {AMP_DTYPE}")


# ==============================================================================
# TRANSFORMS
# ==============================================================================
grain_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

boundary_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ==============================================================================
# DATASET — AUTOREGRESSIVE VERSION
# KEY CHANGE: only consecutive pairs with time_jump=100
# i.e. (t=0→100), (t=100→200), ..., (t=900→1000)
# This gives the model 10x more focused training on single-step prediction
# ==============================================================================
class PottsDatasetAutoregressive(Dataset):
    def __init__(self, root_dir, time_jump=100):
        self.pairs    = []
        self.time_jump = time_jump

        # Build consecutive pairs: (0,100), (100,200), ..., (900,1000)
        timesteps = list(range(0, 1001, time_jump))
        consecutive_pairs = [(timesteps[i], timesteps[i+1]) for i in range(len(timesteps)-1)]

        t0_dir       = os.path.join(root_dir, "timestep_0")
        run_registry = {}

        for fname in sorted(os.listdir(t0_dir)):
            if not fname.endswith("_rgb.png"):
                continue
            match = re.match(r"run_(\d+)_temp_([\d.]+)_timestep_0_rgb\.png", fname)
            if match:
                run_registry[int(match.group(1))] = float(match.group(2))

        print(f"Runs discovered : {len(run_registry)}")
        print(f"Time jump       : {time_jump} (consecutive pairs only)")
        print(f"Pairs per run   : {len(consecutive_pairs)}")

        for run_id, temperature in run_registry.items():
            for t_init, t_target in consecutive_pairs:
                init_grain = os.path.join(
                    root_dir, f"timestep_{t_init}",
                    f"run_{run_id}_temp_{temperature:.3f}_timestep_{t_init}_rgb.png"
                )
                target_grain = os.path.join(
                    root_dir, f"timestep_{t_target}",
                    f"run_{run_id}_temp_{temperature:.3f}_timestep_{t_target}_rgb.png"
                )
                target_boundary = os.path.join(
                    root_dir, f"timestep_{t_target}",
                    f"run_{run_id}_temp_{temperature:.3f}_timestep_{t_target}_boundary.png"
                )
                if all(os.path.exists(p) for p in [init_grain, target_grain, target_boundary]):
                    self.pairs.append({
                        "init_grain":      init_grain,
                        "target_grain":    target_grain,
                        "target_boundary": target_boundary,
                        "temperature":     temperature,
                        "time_jump":       float(time_jump),
                        "t_initial":       t_init,
                        "t_target":        t_target,
                    })

        expected = len(run_registry) * len(consecutive_pairs)
        print(f"Valid pairs    : {len(self.pairs)}  (expected {expected})")
        print(f"Missing pairs  : {expected - len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p            = self.pairs[idx]
        init_image   = grain_transform(Image.open(p["init_grain"]).convert("RGB"))
        target_image = grain_transform(Image.open(p["target_grain"]).convert("RGB"))
        boundary     = boundary_transform(Image.open(p["target_boundary"]).convert("L"))
        boundary     = 1.0 - boundary
        return {
            "initial_image":  init_image,
            "target_image":   target_image,
            "boundary_image": boundary,
            "temperature":    torch.tensor(p["temperature"], dtype=torch.float32),
            "time_jump":      torch.tensor(p["time_jump"],   dtype=torch.float32),
        }


# ==============================================================================
# VISUALISATION
# ==============================================================================
def sobel_edges_np(tensor_img):
    gray = tensor_img.mean(dim=0, keepdim=True).unsqueeze(0)
    device = tensor_img.device
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)
    ex = F.conv2d(gray, sx, padding=1)
    ey = F.conv2d(gray, sy, padding=1)
    edges = torch.sqrt(ex**2 + ey**2)
    return (edges / (edges.max() + 1e-6)).squeeze().cpu().numpy()


def visualize_predictions(initial_image, target_image, pred_clean,
                           boundary_image, pred_boundaries_raw,
                           epoch, batch_idx, viz_dir, time_jump):
    def to_numpy_rgb(t):
        return t[0].detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()

    def to_numpy_gray(t):
        return t[0].detach().cpu().float().clamp(0, 1).squeeze(0).numpy()

    raw = pred_boundaries_raw[0].detach().cpu().float()
    pred_boundary_display = (raw / (raw.max() + 1e-6)).squeeze(0).numpy()

    t_val = int(time_jump[0].item())

    gt_edges   = sobel_edges_np(target_image[0].detach().float().clamp(0, 1))
    pred_edges = sobel_edges_np(pred_clean[0].detach().float().clamp(0, 1))

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    fig.suptitle(f"Epoch {epoch+1} | Batch {batch_idx} | Δt={t_val}", fontsize=14)

    titles = [
        "Initial Image (t=0)",
        f"Ground Truth (x_0)\nΔt = {t_val}",
        f"Predicted (x̂_0)\nΔt = {t_val}",
        "GT Boundary Map",
        "Predicted Boundary (Sobel)",
        "GT Boundary (Sobel)",
    ]
    images = [
        to_numpy_rgb(initial_image),
        to_numpy_rgb(target_image),
        to_numpy_rgb(pred_clean),
        to_numpy_gray(boundary_image),
        pred_edges,
        gt_edges,
    ]
    cmaps = [None, None, None, "gray", "hot", "hot"]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(viz_dir, f"viz_ep{epoch+1:04d}_b{batch_idx:05d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz saved] → {save_path}")


# ==============================================================================
# CHECKPOINT
# ==============================================================================
def save_checkpoint(epoch, model, optimizer, lr_scheduler,
                    train_losses, val_losses, label="epoch"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{label}_{epoch:04d}.pt")
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }, ckpt_path)
    print(f"  [Checkpoint saved] → {ckpt_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    os.makedirs(CHECKPOINT_DIR,  exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR,         exist_ok=True)

    # ---- Dataset ----
    full_dataset = PottsDatasetAutoregressive(ROOT_DIR, time_jump=TIME_JUMP)
    val_size     = int(0.1 * len(full_dataset))
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    # ---- Model ----
    model = GrainDiffusionUNet(
        in_channels=6, out_channels=3,
        base_channels=BASE_CHANNELS,
        context_dim=CONTEXT_DIM,
        embed_dim=EMBED_DIM
    ).to(device)

    loss_fn      = PhysicsInformedLoss(physics_weight=PHYSICS_WEIGHT).to(device)
    scheduler    = NoiseScheduler(num_timesteps=NUM_TIMESTEPS)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    use_scaler = (AMP_DTYPE == torch.float16)
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # ---- Weight loading (before compile) ----
    start_epoch  = 0
    train_losses = []
    val_losses   = []

    if RESUME_TRAINING:
        # Resume from autoregressive checkpoints
        ckpt_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
        if ckpt_files:
            latest = os.path.join(CHECKPOINT_DIR, ckpt_files[-1])
            ckpt   = torch.load(latest, map_location=device)
            state_dict = {k.replace("_orig_mod.", ""): v
                          for k, v in ckpt["model_state"].items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            lr_scheduler.load_state_dict(ckpt["scheduler_state"])
            train_losses = ckpt.get("train_losses", [])
            val_losses   = ckpt.get("val_losses",   [])
            start_epoch  = ckpt["epoch"] + 1
            print(f"Resumed from {latest} at epoch {start_epoch}")

    elif FINETUNE_FROM is not None:
        # KEY OPTION: finetune from existing general model weights
        # This is faster than training from scratch since the model already
        # understands grain structure — it just needs to specialize at Δt=100
        print(f"Finetuning from: {FINETUNE_FROM}")
        ckpt       = torch.load(FINETUNE_FROM, map_location=device)
        state_dict = {k.replace("_orig_mod.", ""): v
                      for k, v in ckpt["model_state"].items()}
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {FINETUNE_FROM}")
        # Note: optimizer and scheduler start fresh — we want to finetune
        # at potentially lower LR, not resume the original training trajectory

    # ---- Compile ----
    model = torch.compile(model)

    print("Warming up torch.compile...")
    with torch.no_grad():
        dummy_noisy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        dummy_init  = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        dummy_step  = torch.randint(0, NUM_TIMESTEPS, (2,), device=device)
        dummy_time  = torch.ones(2, device=device) * float(TIME_JUMP)
        dummy_temp  = torch.ones(2, device=device) * 0.5
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            _ = model(
                noisy_image   = dummy_noisy,
                initial_image = dummy_init,
                diff_step     = dummy_step,
                target_step   = dummy_time,
                temperature   = dummy_temp
            )
    print("Warmup complete.")

    # ---- Interruption handler ----
    interrupted = False
    def handle_signal(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n[SIGNAL {sig}] Saving emergency checkpoint...")
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    # ---- Training Loop ----
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_train_loss  = 0.0
        train_interrupted = False

        for batch_idx, batch in enumerate(tqdm(train_loader,
                                               desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Train]")):
            if interrupted:
                save_checkpoint(epoch, model, optimizer, lr_scheduler,
                                train_losses, val_losses, "emergency")
                train_interrupted = True
                break

            initial_image  = batch["initial_image"].to(device)
            target_image   = batch["target_image"].to(device)
            boundary_image = batch["boundary_image"].to(device)
            temperature    = batch["temperature"].to(device)
            time_jump      = batch["time_jump"].to(device)   # always 100.0

            B         = target_image.size(0)
            diff_step = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)

            noisy_target, actual_noise = scheduler.add_noise(target_image, diff_step, device)

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                predicted_noise = model(
                    noisy_image   = noisy_target,
                    initial_image = initial_image,
                    diff_step     = diff_step,
                    target_step   = time_jump,
                    temperature   = temperature
                )
                pred_clean = scheduler.remove_noise(
                    noisy_target, predicted_noise, diff_step, device
                )
                loss, diff_l, bound_l, energy_l, pred_boundaries_raw = loss_fn(
                    predicted_noise, actual_noise, pred_clean, boundary_image
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

            if batch_idx % LOG_EVERY == 0:
                print(f"  Ep{epoch+1} B{batch_idx:04d} | "
                      f"Total:{loss.item():.4f} | "
                      f"Diff:{diff_l.item():.4f} | "
                      f"Bound:{bound_l.item():.4f} | "
                      f"Energy:{energy_l.item():.4f}")

            if batch_idx % VISUALIZE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    visualize_predictions(
                        initial_image, target_image, pred_clean,
                        boundary_image, pred_boundaries_raw,
                        epoch, batch_idx, VIZ_DIR, time_jump
                    )
                model.train()

        if train_interrupted:
            print("Interrupted. Exiting.")
            break

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        epoch_val_loss  = 0.0
        val_interrupted = False

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val]"):
                if interrupted:
                    save_checkpoint(epoch, model, optimizer, lr_scheduler,
                                    train_losses, val_losses, "emergency")
                    val_interrupted = True
                    break

                initial_image  = batch["initial_image"].to(device)
                target_image   = batch["target_image"].to(device)
                boundary_image = batch["boundary_image"].to(device)
                temperature    = batch["temperature"].to(device)
                time_jump      = batch["time_jump"].to(device)

                B         = target_image.size(0)
                diff_step = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)

                noisy_target, actual_noise = scheduler.add_noise(target_image, diff_step, device)

                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    predicted_noise = model(
                        noisy_image   = noisy_target,
                        initial_image = initial_image,
                        diff_step     = diff_step,
                        target_step   = time_jump,
                        temperature   = temperature
                    )
                    pred_clean = scheduler.remove_noise(
                        noisy_target, predicted_noise, diff_step, device
                    )
                    loss, _, _, _, _ = loss_fn(
                        predicted_noise, actual_noise, pred_clean, boundary_image
                    )

                epoch_val_loss += loss.item()

        if val_interrupted:
            print("Interrupted during validation. Exiting.")
            break

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        lr_scheduler.step()

        print(f"\nEpoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}\n")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(epoch + 1, model, optimizer, lr_scheduler,
                            train_losses, val_losses)

    # ---- Plot loss curves ----
    plot_n = min(len(train_losses), len(val_losses))
    if plot_n > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses[:plot_n], label="Train Loss")
        plt.plot(val_losses[:plot_n],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Autoregressive Training Loss (Δt=100)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(FINAL_MODEL_DIR, "loss_curve.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"Loss curve saved → {FINAL_MODEL_DIR}/loss_curve.png")

    # ---- Save Final Model ----
    final_ckpt_path = os.path.join(FINAL_MODEL_DIR, "grain_diffusion_ar_final.pt")
    torch.save({
        "epoch":           NUM_EPOCHS,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "hyperparameters": {
            "base_channels":  BASE_CHANNELS,
            "context_dim":    CONTEXT_DIM,
            "embed_dim":      EMBED_DIM,
            "physics_weight": PHYSICS_WEIGHT,
            "learning_rate":  LEARNING_RATE,
            "num_timesteps":  NUM_TIMESTEPS,
            "image_size":     IMAGE_SIZE,
            "batch_size":     BATCH_SIZE,
            "num_epochs":     NUM_EPOCHS,
            "amp_dtype":      str(AMP_DTYPE),
            "time_jump":      TIME_JUMP,
        }
    }, final_ckpt_path)
    print(f"Final model saved → {final_ckpt_path}")

    weights_path = os.path.join(FINAL_MODEL_DIR, "grain_diffusion_ar_weights_only.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights only saved → {weights_path}")


if __name__ == "__main__":
    main()