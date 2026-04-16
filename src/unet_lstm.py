# model.py
# Conditional Diffusion U-Net for Potts Model Grain Growth Prediction
# Feature: Includes Bottleneck ConvLSTM for Temporal Momentum Tracking

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# [EXISTING BLOCKS] SinusoidalEmbedding, ContextMLP, AdaGNBlock, ResBlock
# (Kept exactly as you wrote them)
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
# NEW: CONVOLUTIONAL LSTM CELL
# ==============================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3)):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        
        # Gates: 4 * hidden_dim for Input, Forget, Cell, Output
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim, 
            out_channels=4 * hidden_dim, 
            kernel_size=kernel_size, 
            padding=padding
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        h, w = image_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))

# ==============================================================================
# UPDATED: GRAIN DIFFUSION U-NET
# ==============================================================================
class GrainDiffusionUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base_channels=64, 
                 context_dim=256, embed_dim=128):
        super().__init__()
        self.context_mlp = ContextMLP(embed_dim, context_dim)
        ch = base_channels

        # ---- NEW: CONTEXT-FREE PAST FRAME ENCODER ----
        # Shrinks clean past frames down to bottleneck size without adding noise conditioning
        self.past_encoder = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1), nn.SiLU()
        )

        # ---- NEW: BOTTLENECK CONV-LSTM ----
        self.conv_lstm = ConvLSTMCell(input_dim=ch * 8, hidden_dim=ch * 8)
        self.bottleneck_fusion = nn.Conv2d(ch * 8 + ch * 8, ch * 8, kernel_size=1)

        # ---- MAIN ENCODER ----
        self.enc1 = ResBlock(in_channels, ch,     context_dim)
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

        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=1)
    
    def get_momentum(self, past_frames):
        """Extracts topological momentum ONCE for the inference loop"""
        B, S, _, H, W = past_frames.shape
        past_frames_flat = past_frames.view(B * S, 3, H, W)
        past_latents_flat = self.past_encoder(past_frames_flat)
        _, C_lat, H_lat, W_lat = past_latents_flat.shape
        past_latents = past_latents_flat.view(B, S, C_lat, H_lat, W_lat)

        h_state, c_state = self.conv_lstm.init_hidden(B, (H_lat, W_lat), past_frames.device)
        for t in range(S):
            h_state, c_state = self.conv_lstm(past_latents[:, t], (h_state, c_state))
        return h_state

    def forward(self, noisy_image, initial_image, diff_step, target_step, temperature, 
                past_frames=None, precomputed_momentum=None):
        """
        Args:
            noisy_image:   (B, 3, H, W)   - noisy target grain structure (at step t)
            initial_image: (B, 3, H, W)   - clean immediate previous frame (for skip connections)
            diff_step:     (B,)           - diffusion noise timestep (t)
            target_step:   (B,)           - physical target jump (e.g. 100.0)
            temperature:   (B,)           - thermodynamic temperature
            past_frames:   (B, S, 3, H, W)- Sequence of past S frames (used during training)
            precomputed_momentum: (B, C, H, W) - Pre-calculated LSTM state (used during inference)
        """
        device = noisy_image.device
        
        # 1. Build Context (Global physical conditions)
        context = self.context_mlp(diff_step, target_step, temperature)

        # 2. Main Encoder (Process the noisy canvas)
        x = torch.cat([noisy_image, initial_image], dim=1) 
        e1 = self.enc1(x, context)
        e2 = self.enc2(self.down1(e1), context)
        e3 = self.enc3(self.down2(e2), context)
        e4 = self.enc4(self.down3(e3), context)
        b_noisy = self.down4(e4)  # Shape: (B, ch*8, H/16, W/16)

        # 3. Handle Temporal Momentum (The LSTM injection)
        if precomputed_momentum is not None:
            # FAST PATH: Used during the 1000-step inference loop
            momentum_tensor = precomputed_momentum
            
        elif past_frames is not None:
            # TRAINING PATH: Calculate momentum from scratch
            B, S, _, H, W = past_frames.shape
            past_frames_flat = past_frames.view(B * S, 3, H, W)
            past_latents_flat = self.past_encoder(past_frames_flat)
            _, C_lat, H_lat, W_lat = past_latents_flat.shape
            past_latents = past_latents_flat.view(B, S, C_lat, H_lat, W_lat)

            h_state, c_state = self.conv_lstm.init_hidden(B, (H_lat, W_lat), device)
            for t in range(S):
                h_state, c_state = self.conv_lstm(past_latents[:, t], (h_state, c_state))
            momentum_tensor = h_state
            
        else:
            raise ValueError("Must provide either past_frames (training) or precomputed_momentum (inference)")

        # 4. Fuse Momentum into Bottleneck
        b_fused = torch.cat([b_noisy, momentum_tensor], dim=1) # (B, ch*16, H/16, W/16)
        b_fused = F.silu(self.bottleneck_fusion(b_fused))      # Back to (B, ch*8, H/16, W/16)
        
        b = self.bottleneck(b_fused, context)

        # 5. Decoder (Expand back to image space)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1), context)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), context)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), context)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), context)

        return self.final_conv(d1)

# ==============================================================================
# 6. PHYSICS-INFORMED LOSS
#    Diffusion MSE Loss + Boundary Accuracy Loss + Energy Monotonicity Penalty
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

    def forward(self, pred_noise, actual_noise, pred_clean_image, gt_boundary_image):
        """
        Args:
            pred_noise:        (B, 3, H, W) - U-Net predicted noise
            actual_noise:      (B, 3, H, W) - actual noise added during forward diffusion
            pred_clean_image:  (B, 3, H, W) - reconstructed clean grain structure
            gt_boundary_image: (B, 1, H, W) - ground truth boundary map (1=boundary, 0=interior)
        Returns:
            total_loss, diff_loss, boundary_loss, energy_penalty, pred_boundaries_raw
        """

        # ── 1. Diffusion noise MSE loss ────────────────────────────────────────
        diff_loss = self.mse(pred_noise, actual_noise)

        # ── 2. RGB → Grayscale ─────────────────────────────────────────────────
        gray = (0.299 * pred_clean_image[:, 0] +
                0.587 * pred_clean_image[:, 1] +
                0.114 * pred_clean_image[:, 2]).unsqueeze(1)        # (B, 1, H, W)
        
        # ── 3A. Gaussian blur to suppress noise before edge detection ───────────────
        gray_smooth = F.conv2d(gray, self.gaussian, padding=1)

        # ── 3B. Sobel edge magnitudes (raw, continuous, unnormalized) ───────────
        edge_x = F.conv2d(gray_smooth, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray_smooth, self.sobel_y, padding=1)
        pred_boundaries_raw = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # (B, 1, H, W)

        # ── 4. Boundary MSE loss (normalized to [0,1] to match GT scale) ───────
        pred_boundaries_norm = pred_boundaries_raw / (pred_boundaries_raw.max() + 1e-6)
        boundary_loss = self.mse(pred_boundaries_norm, gt_boundary_image)

        # ── 5. Energy penalty (soft binarization so both sides are comparable) ─
        # Soft binarize the Sobel output → each pixel is now "how boundary-like
        # is this pixel" in [0,1], same quantity as gt_boundary_image
        # threshold=0.1 separates noise from real edges
        # temperature=0.05 controls sharpness of the soft step
        pred_boundaries_soft = torch.sigmoid(
            (pred_boundaries_norm - self.energy_threshold) / self.energy_temperature
        )                                                           # (B, 1, H, W) ∈ [0,1]

        # Both are now fractions of boundary pixels — directly comparable
        pred_energy    = pred_boundaries_soft.mean(dim=[1, 2, 3])  # (B,)
        gt_energy      = gt_boundary_image.mean(dim=[1, 2, 3])     # (B,)

        # Only penalize when predicted boundary fraction EXCEEDS ground truth
        # (Potts model: boundary length must not increase)
        energy_penalty = F.relu(pred_energy - gt_energy).mean()

        # ── 6. Total loss ──────────────────────────────────────────────────────
        total_loss = diff_loss + self.physics_weight * (boundary_loss + energy_penalty)

        # Return pred_boundaries_raw so training loop can visualize without recomputing Sobel
        return total_loss, diff_loss, boundary_loss, energy_penalty, pred_boundaries_raw



# ==============================================================================
# 7. NOISE SCHEDULER
#    Handles forward diffusion (add noise) and reverse diffusion (remove noise)
#    Used in the training and inference loops in your .ipynb
# ==============================================================================
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean_image, t, device):
        """
        Forward diffusion: adds noise to clean_image at diffusion step t.
        Returns: (noisy_image, actual_noise)
        """
        alpha_bar = self.alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(clean_image)
        noisy_image = (alpha_bar.sqrt() * clean_image) + ((1 - alpha_bar).sqrt() * noise)
        return noisy_image, noise

    def remove_noise(self, noisy_image, predicted_noise, t, device):
        """
        Reverse diffusion: removes one step of noise using the model's prediction.
        Returns: predicted clean image clamped to [-1, 1]
        """
        alpha_bar = self.alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
        pred_clean = (noisy_image - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        return pred_clean.clamp(-1, 1)
    
    def ddpm_step(self, x_t, pred_noise, t, device):
        """Mathematical reverse-step for the inference loop"""
        alpha_t = self.alphas.to(device)[t]
        alpha_bar_t = self.alpha_cumprod.to(device)[t]
        beta_t = self.betas.to(device)[t]
        
        # We don't add random noise on the very last step (t=0)
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        
        x_t_prev = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise) + beta_t.sqrt() * z
        return x_t_prev
