# model.py
# Channel-Stacked Video Diffusion U-Net for Potts Model Grain Growth
# Feature: Multi-frame Sequence Prediction via Bottleneck Spatial Conditioning (No LSTM)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# [EXISTING BLOCKS] SinusoidalEmbedding, ContextMLP, AdaGNBlock, ResBlock
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
    def __init__(self, embed_dim, context_dim):
        super().__init__()
        self.diff_step_embed   = SinusoidalEmbedding(embed_dim)
        self.target_step_embed = SinusoidalEmbedding(embed_dim)
        self.temp_embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )

    def forward(self, diff_step, target_step, temperature):
        d_emb    = self.diff_step_embed(diff_step.float())
        t_emb    = self.target_step_embed(target_step.float())
        temp_emb = self.temp_embed(temperature.unsqueeze(1))
        combined = torch.cat([d_emb, t_emb, temp_emb], dim=1)
        return self.mlp(combined)

class AdaGNBlock(nn.Module):
    def __init__(self, in_channels, context_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, affine=False)
        self.context_proj = nn.Linear(context_dim, in_channels * 2)

    def forward(self, x, context):
        x_norm = self.norm(x)
        film_params = self.context_proj(context)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta  = beta.unsqueeze(2).unsqueeze(3)
        return x_norm * (1 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim, num_groups=8):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adagn1 = AdaGNBlock(out_channels, context_dim, num_groups)
        self.adagn2 = AdaGNBlock(out_channels, context_dim, num_groups)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, context):
        residual = self.residual_conv(x)
        x = F.silu(self.adagn1(self.conv1(x), context))
        x = F.silu(self.adagn2(self.conv2(x), context))
        return x + residual


# ==============================================================================
# UPDATED: CHANNEL-STACKED VIDEO DIFFUSION U-NET
# ==============================================================================
class GrainDiffusionUNet(nn.Module):
    def __init__(self, num_past_frames=3, num_future_frames=3, base_channels=64, 
                 context_dim=256, embed_dim=128):
        super().__init__()
        self.context_mlp = ContextMLP(embed_dim, context_dim)
        ch = base_channels
        
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames

        # ---- PAST FRAME ENCODER (Dynamic Channels) ----
        # Takes stacked past frames (e.g., 3 frames = 9 channels)
        past_in_channels = num_past_frames * 3
        self.past_encoder = nn.Sequential(
            nn.Conv2d(past_in_channels, ch, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1), nn.SiLU()
        )

        # ---- BOTTLENECK FUSION (No LSTM) ----
        self.bottleneck_fusion = nn.Conv2d(ch * 8 + ch * 8, ch * 8, kernel_size=1)

        # ---- MAIN ENCODER ----
        # In Channels = (Future Frames * 3) for the noisy canvas + 3 for the clean anchor skip connection
        unet_in_channels = (num_future_frames * 3) + 3
        
        self.enc1 = ResBlock(unet_in_channels, ch,     context_dim)
        self.enc2 = ResBlock(ch,          ch * 2, context_dim)
        self.enc3 = ResBlock(ch * 2,      ch * 4, context_dim)
        self.enc4 = ResBlock(ch * 4,      ch * 8, context_dim)

        self.down1 = nn.Conv2d(ch,     ch,     kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(ch * 4, ch * 4, kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1)

        # ---- BOTTLENECK ----
        self.bottleneck = ResBlock(ch * 8, ch * 8, context_dim)

        # ---- DECODER ----
        self.up4  = nn.ConvTranspose2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1)
        self.dec4 = ResBlock(ch * 8 + ch * 8, ch * 8, context_dim)
        self.up3  = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(ch * 4 + ch * 4, ch * 4, context_dim)
        self.up2  = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(ch * 2 + ch * 2, ch * 2, context_dim)
        self.up1  = nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(ch + ch, ch, context_dim)

        # Outputs the stacked predicted frames (e.g., 3 frames = 9 channels)
        self.final_conv = nn.Conv2d(ch, num_future_frames * 3, kernel_size=1)
    
    def get_past_context(self, past_frames_stacked):
        """Extracts spatial context from stacked frames ONCE for the inference loop"""
        # past_frames_stacked shape: (B, num_past_frames*3, H, W)
        return self.past_encoder(past_frames_stacked)

    def forward(self, noisy_image, initial_image, diff_step, target_step, temperature, 
                past_frames_stacked=None, precomputed_context=None):
        """
        Args:
            noisy_image:   (B, F*3, H, W) - noisy target sequence (F frames stacked)
            initial_image: (B, 3, H, W)   - clean immediate previous frame (Anchor)
            diff_step:     (B,)           - diffusion noise timestep (t)
            target_step:   (B,)           - physical target jump
            temperature:   (B,)           - thermodynamic temperature
            past_frames_stacked: (B, P*3, H, W) - Sequence of past P frames stacked
            precomputed_context: (B, C, H, W)   - Pre-calculated context state
        """
        # 1. Build Context
        context = self.context_mlp(diff_step, target_step, temperature)

        # 2. Main Encoder (Process the multi-frame noisy canvas + 1 anchor)
        x = torch.cat([noisy_image, initial_image], dim=1) 
        e1 = self.enc1(x, context)
        e2 = self.enc2(self.down1(e1), context)
        e3 = self.enc3(self.down2(e2), context)
        e4 = self.enc4(self.down3(e3), context)
        b_noisy = self.down4(e4)

        # 3. Handle Spatial Context Injection
        if precomputed_context is not None:
            # FAST PATH: Inference loop
            past_context = precomputed_context
        elif past_frames_stacked is not None:
            # TRAINING PATH: Calculate from scratch
            past_context = self.past_encoder(past_frames_stacked)
        else:
            raise ValueError("Must provide either past_frames_stacked or precomputed_context")

        # 4. Fuse Context into Bottleneck
        b_fused = torch.cat([b_noisy, past_context], dim=1)
        b_fused = F.silu(self.bottleneck_fusion(b_fused))
        b = self.bottleneck(b_fused, context)

        # 5. Decoder (Expand back to multi-frame image space)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1), context)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), context)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), context)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), context)

        return self.final_conv(d1)

# ==============================================================================
# 6. MULTI-FRAME PHYSICS-INFORMED LOSS
# ==============================================================================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, physics_weight=0.1, energy_threshold=0.1, energy_temperature=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight
        self.energy_threshold = energy_threshold
        self.energy_temperature = energy_temperature

        sobel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        gaussian = torch.tensor([[1., 2., 1.],
                                  [2., 4., 2.],
                                  [1., 2., 1.]]) / 16.0
        self.register_buffer('gaussian', gaussian.view(1, 1, 3, 3))

    def forward(self, pred_noise, actual_noise, pred_clean_images, gt_boundary_images):
        """
        Handles stacked frames! (B, F*3, H, W)
        """
        # ── 1. Diffusion noise MSE loss (Works on stacked channels) ────────────
        diff_loss = self.mse(pred_noise, actual_noise)

        # ── 2. Reshape to treat multiple frames as a massive batch ─────────────
        B, C_pred, H, W = pred_clean_images.shape
        num_frames = C_pred // 3  # FIXED: Renamed from F to avoid shadowing torch.nn.functional
        
        # Flatten batch and frames: (B*num_frames, 3, H, W)
        pred_clean_reshaped = pred_clean_images.view(B * num_frames, 3, H, W)
        gt_boundaries_reshaped = gt_boundary_images.view(B * num_frames, 1, H, W)

        # ── 3. RGB → Grayscale ─────────────────────────────────────────────────
        gray = (0.299 * pred_clean_reshaped[:, 0] +
                0.587 * pred_clean_reshaped[:, 1] +
                0.114 * pred_clean_reshaped[:, 2]).unsqueeze(1)
        
        gray_smooth = F.conv2d(gray, self.gaussian, padding=1) # Now F is safely PyTorch again!

        # ── 4. Sobel edge magnitudes ───────────────────────────────────────────
        edge_x = F.conv2d(gray_smooth, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray_smooth, self.sobel_y, padding=1)
        pred_boundaries_raw = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        # ── 5. Boundary MSE loss ───────────────────────────────────────────────
        pred_boundaries_norm = pred_boundaries_raw / (pred_boundaries_raw.max() + 1e-6)
        boundary_loss = self.mse(pred_boundaries_norm, gt_boundaries_reshaped)

        # ── 6. Energy penalty ──────────────────────────────────────────────────
        pred_boundaries_soft = torch.sigmoid(
            (pred_boundaries_norm - self.energy_threshold) / self.energy_temperature
        ) 

        pred_energy    = pred_boundaries_soft.mean(dim=[1, 2, 3]) 
        gt_energy      = gt_boundaries_reshaped.mean(dim=[1, 2, 3])

        energy_penalty = torch.nn.functional.relu(pred_energy - gt_energy).mean()

        # ── 7. Total loss ──────────────────────────────────────────────────────
        total_loss = diff_loss + self.physics_weight * (boundary_loss + energy_penalty)

        # Reshape the raw boundaries back for visualization
        pred_boundaries_raw_orig_shape = pred_boundaries_raw.view(B, num_frames, H, W)

        return total_loss, diff_loss, boundary_loss, energy_penalty, pred_boundaries_raw_orig_shape

# ==============================================================================
# 7. NOISE SCHEDULER (Unchanged)
# ==============================================================================
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean_image, t, device):
        alpha_bar = self.alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(clean_image)
        noisy_image = (alpha_bar.sqrt() * clean_image) + ((1 - alpha_bar).sqrt() * noise)
        return noisy_image, noise

    def remove_noise(self, noisy_image, predicted_noise, t, device):
        alpha_bar = self.alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
        pred_clean = (noisy_image - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        return pred_clean.clamp(-1, 1)
    
    def ddpm_step(self, x_t, pred_noise, t, device):
        alpha_t = self.alphas.to(device)[t]
        alpha_bar_t = self.alpha_cumprod.to(device)[t]
        beta_t = self.betas.to(device)[t]
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t_prev = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise) + beta_t.sqrt() * z
        return x_t_prev