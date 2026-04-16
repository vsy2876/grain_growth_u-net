# train.py
# Grain Growth Conditional Diffusion Model — HPC Training Script
# Optimized for H200 / H100 / A100 with bfloat16 AMP, torch.compile, TF32

import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
import re
import math
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
matplotlib.use('Agg')   # Non-interactive backend for HPC (no display)
import matplotlib.pyplot as plt

from diff_lstm import GrainDiffusionUNet, PhysicsInformedLoss, NoiseScheduler

# ==============================================================================
# CONFIG — Edit these before launching
# ==============================================================================
ROOT_DIR        = "grain_images"
CHECKPOINT_DIR  = "./checkpoints_diff_lstm"
FINAL_MODEL_DIR = "./final_model_diff_lstm"
VIZ_DIR         = "./visualizations_model_diff_lstm"

IMAGE_SIZE      = 512
BATCH_SIZE      = 8
NUM_WORKERS     = 14
NUM_EPOCHS      = 100
SAVE_EVERY      = 5
LOG_EVERY       = 100
VISUALIZE_EVERY = 500

SEQ_LENGTH      = 3      # Number of past frames for the ConvLSTM sequence window

BASE_CHANNELS   = 64
CONTEXT_DIM     = 256
EMBED_DIM       = 128
PHYSICS_WEIGHT  = 0.1
LEARNING_RATE   = 1e-4
NUM_TIMESTEPS   = 1000

RESUME_TRAINING = False   # Set True to resume from latest checkpoint

# bfloat16 is preferred over float16 on H200/H100/A100 (no overflow, same speed)
AMP_DTYPE       = torch.bfloat16

# ==============================================================================
# H200 / H100 / A100 GLOBAL SPEED FLAGS
# ==============================================================================
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32  = True   # TF32 matmul
torch.backends.cudnn.allow_tf32        = True   # TF32 convolutions
torch.backends.cudnn.benchmark         = True   # Auto-tune conv algorithms
mp.set_sharing_strategy('file_system') # Bypass /dev/shm limit on HPC

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
# DATASET: SLIDING WINDOW SEQUENCES
# ==============================================================================
def seed_worker(worker_id):
    # This specifically limits PyTorch CPU ops to 1 thread per worker
    torch.set_num_threads(1)
    
    # (Optional but recommended) Ensure NumPy also doesn't spawn rogue threads
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

class PottsDataset(Dataset):
    def __init__(self, root_dir, seq_length=SEQ_LENGTH):
        self.samples = []
        self.seq_length = seq_length
        timesteps = list(range(0, 1001, 100)) # [0, 100, 200, ..., 1000]
        
        # Create sliding windows: e.g., if seq_length=3, windows are 4 frames long
        # Window example: [0, 100, 200, 300] -> Past=[0,100,200], Target=[300]
        windows = [timesteps[i : i + seq_length + 1] for i in range(len(timesteps) - seq_length)]

        t0_dir = os.path.join(root_dir, "timestep_0")
        run_registry = {}

        for fname in sorted(os.listdir(t0_dir)):
            if not fname.endswith("_rgb.png"):
                continue
            match = re.match(r"run_(\d+)_temp_([\d.]+)_timestep_0_rgb\.png", fname)
            if match:
                run_registry[int(match.group(1))] = float(match.group(2))

        print(f"Runs discovered : {len(run_registry)}")

        for run_id, temperature in run_registry.items():
            for window in windows:
                past_ts = window[:-1]
                target_t = window[-1]
                
                # Generate paths for all past frames
                past_paths = [
                    os.path.join(root_dir, f"timestep_{t}", f"run_{run_id}_temp_{temperature:.3f}_timestep_{t}_rgb.png")
                    for t in past_ts
                ]
                
                target_grain = os.path.join(
                    root_dir, f"timestep_{target_t}",
                    f"run_{run_id}_temp_{temperature:.3f}_timestep_{target_t}_rgb.png"
                )
                target_boundary = os.path.join(
                    root_dir, f"timestep_{target_t}",
                    f"run_{run_id}_temp_{temperature:.3f}_timestep_{target_t}_boundary.png"
                )
                
                # Check if all files in the sequence exist
                all_exist = all(os.path.exists(p) for p in past_paths) and os.path.exists(target_grain) and os.path.exists(target_boundary)
                
                if all_exist:
                    self.samples.append({
                        "past_grain_paths": past_paths,
                        "target_grain":     target_grain,
                        "target_boundary":  target_boundary,
                        "temperature":      temperature,
                        "time_jump":        float(target_t - past_ts[-1]),
                        "t_target":         target_t,
                    })

        print(f"Valid sequence windows : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. Load and stack past frames sequence
        past_tensors = []
        for path in s["past_grain_paths"]:
            img = grain_transform(Image.open(path).convert("RGB"))
            past_tensors.append(img)
            
        past_frames = torch.stack(past_tensors) # Shape: (S, 3, H, W)
        
        # 2. Immediate initial image is the LAST frame in the history sequence
        init_image = past_frames[-1] 
        
        # 3. Target and Boundary images
        target_image = grain_transform(Image.open(s["target_grain"]).convert("RGB"))
        boundary = boundary_transform(Image.open(s["target_boundary"]).convert("L"))
        boundary = 1.0 - boundary   # Invert: 1=boundary, 0=interior
        
        return {
            "past_frames":    past_frames,
            "initial_image":  init_image,
            "target_image":   target_image,
            "boundary_image": boundary,
            "temperature":    torch.tensor(s["temperature"], dtype=torch.float32),
            "time_jump":      torch.tensor(s["time_jump"],   dtype=torch.float32),
        }

# ==============================================================================
# VISUALISATION (6-Panel Plot)
# ==============================================================================
def sobel_edges_np(tensor_img):
    """Helper function to calculate Sobel edges for the ground truth RGB image."""
    gray = tensor_img.mean(dim=0, keepdim=True).unsqueeze(0)
    device = tensor_img.device
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)
    ex = F.conv2d(gray, sx, padding=1)
    ey = F.conv2d(gray, sy, padding=1)
    edges = torch.sqrt(ex**2 + ey**2)
    return (edges / (edges.max() + 1e-6)).squeeze().cpu().numpy()

def visualize_predictions(initial_image, target_image, pred_clean,
                           boundary_image, pred_boundaries_raw, epoch, batch_idx, viz_dir, time_jump):
    def to_numpy_rgb(t):
        return t[0].detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()

    def to_numpy_gray(t):
        return t[0].detach().cpu().float().clamp(0, 1).squeeze(0).numpy()

    t_val = int(time_jump[0].item())

    # 1. Get Predicted Boundaries (Directly from the loss function's Sobel pass)
    raw = pred_boundaries_raw[0].detach().cpu().float()
    pred_boundary_display = (raw / (raw.max() + 1e-6)).squeeze(0).numpy()

    # 2. Get Ground Truth Boundaries (Running Sobel on the GT target image)
    gt_edges = sobel_edges_np(target_image[0].detach().float().clamp(0, 1))

    # Create 6 subplots
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    fig.suptitle(f"Epoch {epoch+1} | Batch {batch_idx} | Δt={t_val} (Sliding Window)", fontsize=16, fontweight='bold')

    titles = [
        "Initial Image (Anchor)",
        f"Ground Truth (x_0)\nΔt = {t_val}",
        f"Predicted (x̂_0)\nΔt = {t_val}",
        "GT Boundary Map (True)",
        "Predicted Boundary (Sobel)",
        "GT Boundary (Sobel)"
    ]
    images = [
        to_numpy_rgb(initial_image),
        to_numpy_rgb(target_image),
        to_numpy_rgb(pred_clean),
        to_numpy_gray(boundary_image),
        pred_boundary_display, # Panel 5: AI's edge prediction
        gt_edges               # Panel 6: Mathematical true edges
    ]
    cmaps = [None, None, None, "gray", "hot", "hot"]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(viz_dir, f"viz_ep{epoch+1:04d}_b{batch_idx:05d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz saved] → {save_path}")

# ==============================================================================
# CHECKPOINT
# ==============================================================================
def save_checkpoint(epoch, model, optimizer, lr_scheduler, train_losses, val_losses, label="epoch"):
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
    full_dataset  = PottsDataset(ROOT_DIR, seq_length=SEQ_LENGTH)
    val_size      = int(0.1 * len(full_dataset))
    train_size    = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    # Update your DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        worker_init_fn=seed_worker  # <--- Added here
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        worker_init_fn=seed_worker  # <--- Added here
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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    use_scaler = (AMP_DTYPE == torch.float16)
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # ---- Resume ----
    start_epoch  = 0
    train_losses = []
    val_losses   = []

    if RESUME_TRAINING:
        ckpt_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
        if ckpt_files:
            latest   = os.path.join(CHECKPOINT_DIR, ckpt_files[-1])
            ckpt     = torch.load(latest, map_location=device)
            
            state_dict = ckpt["model_state"]
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("_orig_mod.", "")
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            
            optimizer.load_state_dict(ckpt["optimizer_state"])
            lr_scheduler.load_state_dict(ckpt["scheduler_state"])
            train_losses = ckpt.get("train_losses", [])
            val_losses   = ckpt.get("val_losses",   [])
            start_epoch  = ckpt["epoch"] + 1
            print(f"Resumed from {latest} at epoch {start_epoch}")

    model = torch.compile(model)

    print("Warming up torch.compile with dummy forward pass...")
    with torch.no_grad():
        dummy_noisy  = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        dummy_init   = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        dummy_past   = torch.randn(2, SEQ_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE, device=device) # Added
        dummy_step   = torch.randint(0, NUM_TIMESTEPS, (2,), device=device)
        dummy_time   = torch.ones(2, device=device) * 100.0
        dummy_temp   = torch.ones(2, device=device) * 0.5
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            _ = model(
                noisy_image   = dummy_noisy,
                initial_image = dummy_init,
                diff_step     = dummy_step,
                target_step   = dummy_time,
                temperature   = dummy_temp,
                past_frames   = dummy_past,      # Added
                precomputed_momentum = None
            )
    print("torch.compile warmup complete.")

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

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Train]")):
            if interrupted:
                save_checkpoint(epoch, model, optimizer, lr_scheduler, train_losses, val_losses, "emergency")
                train_interrupted = True
                break

            past_frames    = batch["past_frames"].to(device)
            initial_image  = batch["initial_image"].to(device)
            target_image   = batch["target_image"].to(device)
            boundary_image = batch["boundary_image"].to(device)
            temperature    = batch["temperature"].to(device)
            time_jump      = batch["time_jump"].to(device)

            B         = target_image.size(0)
            diff_step = torch.randint(0, NUM_TIMESTEPS, (B,), device=device)

            noisy_target, actual_noise = scheduler.add_noise(target_image, diff_step, device)

            # ---- AMP Forward Pass ----
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                predicted_noise = model(
                    noisy_image   = noisy_target,
                    initial_image = initial_image,
                    diff_step     = diff_step,
                    target_step   = time_jump,
                    temperature   = temperature,
                    past_frames   = past_frames,          # Added sequence tensor
                    precomputed_momentum = None
                )
                pred_clean = scheduler.remove_noise(noisy_target, predicted_noise, diff_step, device)
                loss, diff_l, bound_l, energy_l, pred_boundaries_raw = loss_fn(
                    predicted_noise, actual_noise, pred_clean, boundary_image
                )

            # ---- Backward Pass with GradScaler ----
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Critical for LSTM
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
        epoch_val_loss    = 0.0
        val_interrupted   = False 

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [Val]"):
                if interrupted:
                    save_checkpoint(epoch, model, optimizer, lr_scheduler, train_losses, val_losses, "emergency")
                    val_interrupted = True
                    break

                past_frames    = batch["past_frames"].to(device)
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
                        temperature   = temperature,
                        past_frames   = past_frames,          # Added sequence tensor
                        precomputed_momentum = None
                    )
                    pred_clean = scheduler.remove_noise(noisy_target, predicted_noise, diff_step, device)
                    loss, _, _, _, _ = loss_fn(predicted_noise, actual_noise, pred_clean, boundary_image)

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
            save_checkpoint(epoch + 1, model, optimizer, lr_scheduler, train_losses, val_losses)

    # ---- Plot loss curves ----
    plot_n = min(len(train_losses), len(val_losses))
    if plot_n > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses[:plot_n], label="Train Loss")
        plt.plot(val_losses[:plot_n],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(FINAL_MODEL_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss curve saved → {FINAL_MODEL_DIR}/loss_curve.png")

    # ---- Save Final Model ----
    final_ckpt_path = os.path.join(FINAL_MODEL_DIR, "grain_diffusion_final.pt")
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
            "seq_length":     SEQ_LENGTH,
        }
    }, final_ckpt_path)
    print(f"Final model saved → {final_ckpt_path}")

    weights_path = os.path.join(FINAL_MODEL_DIR, "grain_diffusion_weights_only.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights only saved → {weights_path}")

if __name__ == "__main__":
    main()