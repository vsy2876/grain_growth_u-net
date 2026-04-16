# model.py
# Channel-Stacked Video Diffusion U-Net for Potts Model Grain Growth
# Architecture: 1 frame in → 3 future frames out simultaneously (one-shot)
# Changes vs previous version:
#   - num_past_frames hardcoded to 1 (ablations proved multi-frame adds nothing)
#   - past_encoder simplified to 4 layers, 3 input channels
#   - Frame index embeddings added before final_conv (gives each output frame identity)
#   - PhysicsInformedLoss: added SSIM loss + cross-frame monotonic energy penalty
#   - NoiseScheduler: added ddim_step for fast inference

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from pytorch_msssim import ssim as compute_ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("[WARNING] pytorch_msssim not found. SSIM loss disabled. Install with: pip install pytorch-msssim")


# ==============================================================================
# BUILDING BLOCKS
# ==============================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        device = x.device
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


class ContextMLP(nn.Module):
    """
    Fuses three conditioning signals into a single context vector:
      - diff_step:    the diffusion noise level (0-999)
      - target_step:  the physical time jump being predicted (e.g. 300)
      - temperature:  thermodynamic temperature of the run
    """
    def __init__(self, embed_dim, context_dim):
        super().__init__()
        self.diff_step_embed   = SinusoidalEmbedding(embed_dim)
        self.target_step_embed = SinusoidalEmbedding(embed_dim)
        self.temp_embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, context_dim), nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )

    def forward(self, diff_step, target_step, temperature):
        d_emb    = self.diff_step_embed(diff_step.float())
        t_emb    = self.target_step_embed(target_step.float())
        temp_emb = self.temp_embed(temperature.unsqueeze(1))
        combined = torch.cat([d_emb, t_emb, temp_emb], dim=1)
        return self.mlp(combined)


class AdaGNBlock(nn.Module):
    """Adaptive Group Normalisation — injects context into feature maps via FiLM."""
    def __init__(self, in_channels, context_dim, num_groups=8):
        super().__init__()
        self.norm         = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, affine=False)
        self.context_proj = nn.Linear(context_dim, in_channels * 2)

    def forward(self, x, context):
        x_norm    = self.norm(x)
        film      = self.context_proj(context)
        gamma, beta = film.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta  = beta.unsqueeze(2).unsqueeze(3)
        return x_norm * (1 + gamma) + beta


class ResBlock(nn.Module):
    """Residual block with two AdaGN-conditioned convolutions."""
    def __init__(self, in_channels, out_channels, context_dim, num_groups=8):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adagn1 = AdaGNBlock(out_channels, context_dim, num_groups)
        self.adagn2 = AdaGNBlock(out_channels, context_dim, num_groups)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, context):
        residual = self.residual_conv(x)
        x = F.silu(self.adagn1(self.conv1(x), context))
        x = F.silu(self.adagn2(self.conv2(x), context))
        return x + residual


# ==============================================================================
# MAIN MODEL
# ==============================================================================

class GrainDiffusionUNet(nn.Module):
    """
    One-shot multi-frame diffusion U-Net.

    Input:
        - noisy_image:   (B, num_future_frames*3, H, W)  — noisy canvas of future frames
        - initial_image: (B, 3, H, W)                    — clean anchor frame (most recent past)
        - diff_step:     (B,)                             — diffusion timestep
        - target_step:   (B,)                             — physical time jump to last target
        - temperature:   (B,)                             — run temperature
        - past_frame / precomputed_context               — single past frame for bottleneck

    Output:
        - (B, num_future_frames*3, H, W)  — predicted noise for all future frames
          Channels [0:3]=T+100, [3:6]=T+200, [6:9]=T+300, each with its own frame embedding.
    """

    def __init__(self, num_future_frames=3, base_channels=64, context_dim=256, embed_dim=128):
        super().__init__()
        self.num_future_frames = num_future_frames
        ch = base_channels

        self.context_mlp = ContextMLP(embed_dim, context_dim)

        # ── Past Frame Encoder ────────────────────────────────────────────────
        # Takes the single anchor frame (3 channels).
        # 4 stride-2 layers: 512→256→128→64→32, matching the U-Net bottleneck spatial size.
        self.past_encoder = nn.Sequential(
            nn.Conv2d(3,      ch*2, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch*2,   ch*4, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch*4,   ch*8, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch*8,   ch*8, kernel_size=4, stride=2, padding=1), nn.SiLU(),
        )

        # ── Bottleneck Fusion ─────────────────────────────────────────────────
        # Merges noisy bottleneck features with past frame context.
        self.bottleneck_fusion = nn.Conv2d(ch*8 + ch*8, ch*8, kernel_size=1)

        # ── Main Encoder ──────────────────────────────────────────────────────
        # Input = noisy future canvas (F*3) + clean anchor (3)
        unet_in = num_future_frames * 3 + 3
        self.enc1 = ResBlock(unet_in, ch,     context_dim)
        self.enc2 = ResBlock(ch,      ch*2,   context_dim)
        self.enc3 = ResBlock(ch*2,    ch*4,   context_dim)
        self.enc4 = ResBlock(ch*4,    ch*8,   context_dim)

        self.down1 = nn.Conv2d(ch,    ch,    kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(ch*2,  ch*2,  kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(ch*4,  ch*4,  kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(ch*8,  ch*8,  kernel_size=4, stride=2, padding=1)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ResBlock(ch*8, ch*8, context_dim)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(ch*8, ch*8, kernel_size=4, stride=2, padding=1)
        self.dec4 = ResBlock(ch*8 + ch*8, ch*8, context_dim)
        self.up3  = nn.ConvTranspose2d(ch*8, ch*4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(ch*4 + ch*4, ch*4, context_dim)
        self.up2  = nn.ConvTranspose2d(ch*4, ch*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(ch*2 + ch*2, ch*2, context_dim)
        self.up1  = nn.ConvTranspose2d(ch*2, ch,   kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(ch + ch,     ch,   context_dim)

        # ── Per-Frame Output Head ─────────────────────────────────────────────
        # Each future frame gets its own learned identity embedding (ch-dim vector).
        # The same final_conv is applied to (d1 + frame_embed[i]) for each frame i.
        # This tells the decoder which of the 3 output frames it is producing,
        # so T=300, T=400, T=500 each see a different input — closing the SSIM gap
        # that previously opened up across the output sequence.
        self.frame_embed = nn.Embedding(num_future_frames, ch)
        # self.final_conv  = nn.Conv2d(ch, 3, kernel_size=1)   # outputs 3 channels per frame
        # ── Per-Frame Output Head ─────────────────────────────────────────────        
        # Replace the linear 1x1 conv with a non-linear head
        self.frame_decoder = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, 3, kernel_size=1)
        )

    # ─────────────────────────────────────────────────────────────────────────
    def get_past_context(self, past_frame):
        """Pre-compute bottleneck context from anchor frame once per inference call."""
        return self.past_encoder(past_frame)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, noisy_image, initial_image, diff_step, target_step, temperature,
                past_frame=None, precomputed_context=None):
        """
        Args
        ----
        noisy_image         : (B, F*3, H, W)
        initial_image       : (B, 3, H, W)   — clean anchor (also used as past_frame)
        diff_step           : (B,)
        target_step         : (B,)            — physical jump to last target (e.g. 300)
        temperature         : (B,)
        past_frame          : (B, 3, H, W)    — training path: compute context from scratch
        precomputed_context : (B, C, h, w)    — inference path: pre-computed context
        """
        # 1. Context vector (scalar conditioning for all ResBlocks via AdaGN)
        context = self.context_mlp(diff_step, target_step, temperature)

        # 2. Main encoder — processes noisy future canvas + clean anchor
        x  = torch.cat([noisy_image, initial_image], dim=1)
        e1 = self.enc1(x, context)
        e2 = self.enc2(self.down1(e1), context)
        e3 = self.enc3(self.down2(e2), context)
        e4 = self.enc4(self.down3(e3), context)
        b_noisy = self.down4(e4)

        # 3. Past frame context
        if precomputed_context is not None:
            past_ctx = precomputed_context                  # inference: pre-computed
        elif past_frame is not None:
            past_ctx = self.past_encoder(past_frame)        # training: compute now
        else:
            # Fallback: use anchor directly (same frame, valid since num_past_frames=1)
            past_ctx = self.past_encoder(initial_image)

        # 4. Fuse bottleneck
        b_fused = torch.cat([b_noisy, past_ctx], dim=1)
        b_fused = F.silu(self.bottleneck_fusion(b_fused))
        b = self.bottleneck(b_fused, context)

        # 5. Decoder
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1), context)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), context)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), context)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), context)

        # 6. Per-frame output with frame identity embeddings
        # d1: (B, ch, H, W). For each future frame i, add its learned embedding
        # then project to 3 RGB channels. Concatenate all frames along channel dim.
        frame_idx   = torch.arange(self.num_future_frames, device=d1.device)
        frame_embs  = self.frame_embed(frame_idx)           # (F, ch)
        frame_embs  = frame_embs.unsqueeze(-1).unsqueeze(-1)  # (F, ch, 1, 1)

        frame_outputs = []
        for i in range(self.num_future_frames):
            feat = d1 + frame_embs[i].unsqueeze(0)         # (B, ch, H, W)
            # frame_outputs.append(self.final_conv(feat))     # (B, 3, H, W)
            # Pass through the NON-LINEAR decoder
            frame_outputs.append(self.frame_decoder(feat))

        return torch.cat(frame_outputs, dim=1)              # (B, F*3, H, W)


# ==============================================================================
# PHYSICS-INFORMED LOSS
# ==============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Multi-component loss for grain growth diffusion.

    Components:
      1. Diffusion noise MSE         — core denoising objective
      2. Boundary MSE                — Sobel edges of pred vs GT boundary map
      3. Energy penalty (one-way)    — penalise predicted energy > GT energy
      4. Temporal monotonic penalty  — penalise Energy(frame i+1) > Energy(frame i)
      5. SSIM loss (optional)        — structural similarity, weight ramped during training

    Args
    ----
    physics_weight    : weight applied to (boundary + energy + temporal) block
    energy_threshold  : sigmoid midpoint for soft boundary binarisation
    energy_temperature: sigmoid steepness
    """

    def __init__(self, physics_weight=0.1, energy_threshold=0.1, energy_temperature=0.05):
        super().__init__()
        self.mse              = nn.MSELoss()
        self.physics_weight   = physics_weight
        self.energy_threshold = energy_threshold
        self.energy_temp      = energy_temperature

        sobel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        gaussian = torch.tensor([[1., 2., 1.],
                                  [2., 4., 2.],
                                  [1., 2., 1.]]) / 16.0

        self.register_buffer('sobel_x',  sobel_x)
        self.register_buffer('sobel_y',  sobel_y)
        self.register_buffer('gaussian', gaussian.view(1, 1, 3, 3))

    # ─────────────────────────────────────────────────────────────────────────
    def _sobel_energy(self, rgb_frames):
        """
        Compute normalised Sobel boundary energy for a batch of RGB frames.

        Args:  rgb_frames : (N, 3, H, W) in any value range
        Returns: boundaries_norm : (N, 1, H, W) in [0, 1]
        """
        gray = (0.299 * rgb_frames[:, 0]
              + 0.587 * rgb_frames[:, 1]
              + 0.114 * rgb_frames[:, 2]).unsqueeze(1)
        gray   = F.conv2d(gray, self.gaussian, padding=1)
        edge_x = F.conv2d(gray, self.sobel_x,  padding=1)
        edge_y = F.conv2d(gray, self.sobel_y,  padding=1)
        mag    = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return mag / (mag.max() + 1e-6)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, pred_noise, actual_noise,
                pred_clean_images, gt_clean_images, gt_boundary_images, diff_step,
                ssim_weight=0.0):
        """
        Args
        ----
        pred_noise        : (B, F*3, H, W)
        actual_noise      : (B, F*3, H, W)
        pred_clean_images : (B, F*3, H, W)   — x0 estimated from denoising step
        gt_clean_images   : (B, F*3, H, W)   — ground-truth future frames
        gt_boundary_images: (B, F,   H, W)   — pre-computed boundary maps (1=boundary)
        ssim_weight       : float             — ramped from 0→0.15 during training

        Returns
        -------
        total_loss, diff_loss, boundary_loss, energy_penalty,
        temporal_penalty, ssim_loss, pred_boundaries_raw (B, F, H, W)
        """
        B, C_pred, H, W = pred_clean_images.shape
        num_frames = C_pred // 3

        # ── 1. Diffusion noise MSE ─────────────────────────────────────────
        diff_loss = self.mse(pred_noise, actual_noise)

        # ── 2. Reshape frames into batch dimension ─────────────────────────
        pred_flat = pred_clean_images.view(B * num_frames, 3, H, W)  # (B*F, 3, H, W)
        gt_flat   = gt_clean_images.view(B * num_frames, 3, H, W)
        gt_bnd    = gt_boundary_images.view(B * num_frames, 1, H, W)

        # ── 3. Sobel edges on predictions ─────────────────────────────────
        pred_bnd_norm = self._sobel_energy(pred_flat)                 # (B*F, 1, H, W)

        # ── 4. Boundary MSE ────────────────────────────────────────────────
        boundary_loss = self.mse(pred_bnd_norm, gt_bnd)

        # ── 5. Per-frame energy scalar (soft boundary density) ─────────────
        pred_bnd_soft = torch.sigmoid(
            (pred_bnd_norm - self.energy_threshold) / self.energy_temp
        )
        gt_bnd_soft = torch.sigmoid(
            (gt_bnd - self.energy_threshold) / self.energy_temp
        )

        pred_energy = pred_bnd_soft.mean(dim=[1, 2, 3])              # (B*F,)
        gt_energy   = gt_bnd_soft.mean(dim=[1, 2, 3])                # (B*F,)

        # ── 6. One-way energy penalty (pred must not exceed GT) ────────────
        energy_penalty = F.relu(pred_energy - gt_energy).mean()

        # ── 7. Cross-frame monotonic energy penalty ────────────────────────
        # Enforce Energy(T+200) ≤ Energy(T+100) ≤ Energy(T+300) within each prediction.
        # This directly encodes the 2nd Law across the output sequence.
        pred_energy_seq = pred_energy.view(B, num_frames)            # (B, F)
        temporal_penalty = torch.tensor(0.0, device=pred_clean_images.device)
        for i in range(num_frames - 1):
            temporal_penalty = temporal_penalty + F.relu(
                pred_energy_seq[:, i + 1] - pred_energy_seq[:, i]
            ).mean()

        # ── 8. SSIM loss (ramped in during training) ───────────────────────
        ssim_loss = torch.tensor(0.0, device=pred_clean_images.device)
        
        # Create a strict mask: Only compute SSIM for samples where diff_step < 200
        valid_mask = (diff_step < 200).float()
        
        if ssim_weight > 0.0 and SSIM_AVAILABLE and valid_mask.sum() > 0:
            pred_01 = (pred_flat.clamp(-1, 1) + 1) / 2              
            gt_01   = (gt_flat.clamp(-1, 1)   + 1) / 2
            
            # Calculate raw SSIM for the batch without averaging yet
            raw_ssim_loss = 1.0 - compute_ssim(pred_01, gt_01, data_range=1.0, size_average=False)
            
            # Apply the mask so we only penalize low-noise predictions
            masked_ssim = raw_ssim_loss * valid_mask.repeat_interleave(num_frames)
            
            # Only average over the valid samples
            ssim_loss = masked_ssim.sum() / (valid_mask.sum() * num_frames)

        # ── 9. Total loss ──────────────────────────────────────────────────
        physics_block = boundary_loss + energy_penalty + temporal_penalty
        total_loss    = diff_loss + self.physics_weight * physics_block + (ssim_weight * ssim_loss)

        # Reshape raw boundaries for visualisation (B, F, H, W)
        pred_bnd_raw_vis = pred_bnd_norm.view(B, num_frames, H, W)

        return (total_loss, diff_loss, boundary_loss,
                energy_penalty, temporal_penalty, ssim_loss,
                pred_bnd_raw_vis)


# ==============================================================================
# NOISE SCHEDULER
# ==============================================================================

class NoiseScheduler:
    """
    DDPM noise schedule with DDIM sampling support.

    Use ddpm_step for training-time inference checks.
    Use ddim_step (50 steps, eta=0) for fast evaluation — no retraining needed.
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T             = num_timesteps
        self.betas         = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas        = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean_image, t, device):
        alpha_bar  = self.alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
        noise      = torch.randn_like(clean_image)
        noisy      = alpha_bar.sqrt() * clean_image + (1 - alpha_bar).sqrt() * noise
        return noisy, noise

    def remove_noise(self, noisy_image, predicted_noise, t, device):
        alpha_bar  = self.alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
        pred_clean = (noisy_image - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        return pred_clean.clamp(-1, 1)

    def ddpm_step(self, x_t, pred_noise, t, device):
        """Standard DDPM reverse step — stochastic."""
        alpha_t     = self.alphas.to(device)[t]
        alpha_bar_t = self.alpha_cumprod.to(device)[t]
        beta_t      = self.betas.to(device)[t]
        z           = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return (1 / alpha_t.sqrt()) * (
            x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise
        ) + beta_t.sqrt() * z

    def ddim_step(self, x_t, pred_noise, t, t_prev, device, eta=0.0):
        """
        DDIM reverse step — deterministic when eta=0.

        Use with 50 evenly-spaced timesteps for ~17x faster inference:

            steps = torch.linspace(999, 0, 50, dtype=torch.long)
            for i, t in enumerate(steps):
                t_prev = steps[i+1] if i+1 < len(steps) else torch.tensor(-1)
                pred_noise = model(...)
                x_t = scheduler.ddim_step(x_t, pred_noise, t, t_prev, device)
        """
        ac     = self.alpha_cumprod.to(device)
        a_t    = ac[t]
        a_prev = ac[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        x0_pred = (x_t - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        dir_xt  = (1 - a_prev).sqrt() * pred_noise
        return a_prev.sqrt() * x0_pred + dir_xt