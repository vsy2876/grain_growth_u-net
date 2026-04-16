# model.py
# Conditional Diffusion U-Net for Potts Model Grain Growth Prediction
# Inputs: Initial Grain Image (3ch RGB) + Temperature (scalar) + Target Timestep (scalar)
# Output: Predicted Noise (3ch RGB)

from matplotlib.pyplot import gray
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. SINUSOIDAL EMBEDDING
#    Converts a scalar timestep into a rich high-dimensional vector
# ==============================================================================
class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B,) float tensor
        device = x.device
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)   # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # (B, embed_dim)
        return emb


# ==============================================================================
# 2. CONTEXT MLP
#    Combines diffusion_step, target_step, and temperature into one Context Vector
# ==============================================================================
class ContextMLP(nn.Module):
    def __init__(self, embed_dim, context_dim):
        super().__init__()
        self.diff_step_embed   = SinusoidalEmbedding(embed_dim)
        self.target_step_embed = SinusoidalEmbedding(embed_dim)

        # Temperature: raw scalar -> embedding via MLP
        self.temp_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Fuse all 3 embeddings into one Context Vector
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )

    def forward(self, diff_step, target_step, temperature):
        # All inputs: (B,) tensors
        d_emb    = self.diff_step_embed(diff_step.float())          # (B, embed_dim)
        t_emb    = self.target_step_embed(target_step.float())      # (B, embed_dim)
        temp_emb = self.temp_embed(temperature.unsqueeze(1))        # (B, embed_dim)

        combined = torch.cat([d_emb, t_emb, temp_emb], dim=1)      # (B, embed_dim * 3)
        return self.mlp(combined)                                    # (B, context_dim)


# ==============================================================================
# 3. AdaGN BLOCK (Adaptive Group Normalization)
#    Injects the Context Vector into image feature maps via scale (gamma) + shift (beta)
# ==============================================================================
class AdaGNBlock(nn.Module):
    def __init__(self, in_channels, context_dim, num_groups=8):
        super().__init__()
        # Normalize feature maps first (no learnable affine — AdaGN handles that)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, affine=False)
        # Projects context vector to gamma + beta (size = 2 * channels)
        self.context_proj = nn.Linear(context_dim, in_channels * 2)

    def forward(self, x, context):
        # x:       (B, C, H, W) — image feature maps
        # context: (B, context_dim) — the context vector
        x_norm = self.norm(x)

        # Generate gamma and beta from context
        film_params = self.context_proj(context)              # (B, C*2)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)               # (B, C, 1, 1)
        beta  = beta.unsqueeze(2).unsqueeze(3)                # (B, C, 1, 1)

        # AdaGN formula: normalized_features * (1 + gamma) + beta
        return x_norm * (1 + gamma) + beta


# ==============================================================================
# 4. RESIDUAL BLOCK WITH AdaGN
#    Basic building block of the U-Net encoder and decoder
# ==============================================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim, num_groups=8):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adagn1 = AdaGNBlock(out_channels, context_dim, num_groups)
        self.adagn2 = AdaGNBlock(out_channels, context_dim, num_groups)

        # Match channel dimensions for residual connection if needed
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, context):
        residual = self.residual_conv(x)

        x = F.silu(self.adagn1(self.conv1(x), context))
        x = F.silu(self.adagn2(self.conv2(x), context))

        return x + residual


# ==============================================================================
# 5. GRAIN DIFFUSION U-NET
#    Full U-Net: Encoder -> Bottleneck -> Decoder with Skip Connections
#    6-channel input (InstructPix2Pix): [3ch noisy target | 3ch initial image]
# ==============================================================================
class GrainDiffusionUNet(nn.Module):
    def __init__(self,
                 in_channels=6,       # 3ch noisy target + 3ch initial image
                 out_channels=3,      # 3ch predicted noise
                 base_channels=64,    # Feature map width at first encoder block
                 context_dim=256,     # Context Vector size
                 embed_dim=128):      # Sinusoidal embedding dimension
        super().__init__()

        self.context_mlp = ContextMLP(embed_dim, context_dim)

        ch = base_channels

        # ---- ENCODER ----
        self.enc1 = ResBlock(in_channels, ch,     context_dim)   # Full res
        self.enc2 = ResBlock(ch,          ch * 2, context_dim)   # Half res
        self.enc3 = ResBlock(ch * 2,      ch * 4, context_dim)   # Quarter res
        self.enc4 = ResBlock(ch * 4,      ch * 8, context_dim)   # Eighth res

        # Downsampling (stride-2 convolutions)
        self.down1 = nn.Conv2d(ch,     ch,     kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(ch * 4, ch * 4, kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1)

        # ---- BOTTLENECK ----
        self.bottleneck = ResBlock(ch * 8, ch * 8, context_dim)

        # ---- DECODER ----
        # Each block input = upsampled features + skip connection (hence doubled channels)
        self.up4  = nn.ConvTranspose2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1)
        self.dec4 = ResBlock(ch * 8 + ch * 8, ch * 8, context_dim)

        self.up3  = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(ch * 4 + ch * 4, ch * 4, context_dim)

        self.up2  = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(ch * 2 + ch * 2, ch * 2, context_dim)

        self.up1  = nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(ch + ch, ch, context_dim)

        # Final 1x1 conv to produce the 3-channel noise prediction
        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, noisy_image, initial_image, diff_step, target_step, temperature):
        """
        Args:
            noisy_image:   (B, 3, H, W) - noisy target grain structure
            initial_image: (B, 3, H, W) - clean initial grain structure at t=0
            diff_step:     (B,)         - diffusion noise timestep (0 to 999)
            target_step:   (B,)         - physical simulation target timestep (e.g. 500.0)
            temperature:   (B,)         - thermodynamic temperature of this simulation run

        Returns:
            (B, 3, H, W) - predicted noise
        """

        # Step 1: Build the Context Vector from all scalar conditions
        context = self.context_mlp(diff_step, target_step, temperature)  # (B, context_dim)

        # Step 2: Concatenate images — InstructPix2Pix style (6-channel input)
        x = torch.cat([noisy_image, initial_image], dim=1)               # (B, 6, H, W)

        # Step 3: ENCODER (save outputs for skip connections)
        e1 = self.enc1(x,              context)                          # (B, ch,    H,    W)
        e2 = self.enc2(self.down1(e1), context)                          # (B, ch*2,  H/2,  W/2)
        e3 = self.enc3(self.down2(e2), context)                          # (B, ch*4,  H/4,  W/4)
        e4 = self.enc4(self.down3(e3), context)                          # (B, ch*8,  H/8,  W/8)

        # Step 4: BOTTLENECK
        b  = self.bottleneck(self.down4(e4), context)                    # (B, ch*8,  H/16, W/16)

        # Step 5: DECODER (upsample and concatenate skip connections)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1), context)   # (B, ch*8,  H/8,  W/8)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), context)   # (B, ch*4,  H/4,  W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), context)   # (B, ch*2,  H/2,  W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), context)   # (B, ch,    H,    W)

        # Step 6: Output predicted noise
        return self.final_conv(d1)                                        # (B, 3, H, W)


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
