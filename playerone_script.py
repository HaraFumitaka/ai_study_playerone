import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Conv3DBlock(nn.Module):
    """3D Convolutional Block for motion and camera encoding"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MotionEncoder(nn.Module):
    """
    Motion Encoder for encoding human motion parameters
    Takes motion parameters and outputs latent representations
    """
    def __init__(self, input_dim: int, hidden_dims: list = [64, 128, 256, 384, 512, 640, 768, 896]):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        # 8 layers of 3D convolutions as mentioned in paper
        for hidden_dim in hidden_dims:
            layers.append(Conv3DBlock(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = hidden_dims[-1]
    
    def forward(self, motion_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_params: (B, C, T, H, W) - motion parameters
        Returns:
            motion_latent: (B, C_out, T, H, W)
        """
        return self.encoder(motion_params)


class CameraEncoder(nn.Module):
    """
    Camera Encoder using Plücker ray parameterization
    Encodes camera extrinsics (rotation-only) for view-change information
    """
    def __init__(self, input_dim: int = 6, hidden_dims: list = [64, 128, 256, 384, 512, 640, 768, 896]):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(Conv3DBlock(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = hidden_dims[-1]
    
    def rodrigues_to_rotation_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation vector to rotation matrix using Rodrigues formula
        Args:
            v: (B, T, 3) rotation vectors
        Returns:
            R: (B, T, 3, 3) rotation matrices
        """
        # Normalize rotation axis
        theta = torch.norm(v, dim=-1, keepdim=True)
        u = v / (theta + 1e-8)
        
        # Construct cross product matrix
        B, T, _ = v.shape
        u_cross = torch.zeros(B, T, 3, 3, device=v.device)
        u_cross[..., 0, 1] = -u[..., 2]
        u_cross[..., 0, 2] = u[..., 1]
        u_cross[..., 1, 0] = u[..., 2]
        u_cross[..., 1, 2] = -u[..., 0]
        u_cross[..., 2, 0] = -u[..., 1]
        u_cross[..., 2, 1] = u[..., 0]
        
        # Rodrigues formula: R = I + sin(θ)[u]× + (1-cos(θ))[u]×²
        I = torch.eye(3, device=v.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)
        
        R = I + sin_theta * u_cross + (1 - cos_theta) * torch.matmul(u_cross, u_cross)
        return R
    
    def rotation_to_plucker(self, R: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to Plücker ray representation
        Args:
            R: (B, T, 3, 3) rotation matrices
        Returns:
            plucker: (B, T, 6) Plücker coordinates
        """
        # Extract direction and moment from rotation matrix
        direction = R[..., :3, 2]  # Taking z-axis as direction
        moment = torch.cross(torch.zeros_like(direction), direction, dim=-1)
        plucker = torch.cat([direction, moment], dim=-1)
        return plucker
    
    def forward(self, head_motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_motion: (B, T, 3) - head rotation parameters (θx, θy, θz)
        Returns:
            camera_latent: (B, C_out, T, H, W)
        """
        # Convert to rotation matrix
        R = self.rodrigues_to_rotation_matrix(head_motion)
        
        # Convert to Plücker parameterization
        plucker = self.rotation_to_plucker(R)  # (B, T, 6)
        
        # Reshape for 3D convolution: (B, C, T, H, W)
        B, T, C = plucker.shape
        plucker = plucker.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, 6, T, 1, 1)
        
        return self.encoder(plucker)


class PartDisentangledMotionInjection(nn.Module):
    """
    Part-Disentangled Motion Injection (PMI)
    Splits human motion into three parts: head, hands, body&feet
    Each part is processed by its own motion encoder
    """
    def __init__(self, 
                 head_dim: int = 3,
                 hands_dim: int = 90,  # 45 per hand
                 body_feet_dim: int = 66):
        super().__init__()
        
        # Three separate motion encoders for different body parts
        self.head_encoder = MotionEncoder(head_dim)
        self.hands_encoder = MotionEncoder(hands_dim)
        self.body_feet_encoder = MotionEncoder(body_feet_dim)
        
        # Camera encoder for view alignment
        self.camera_encoder = CameraEncoder()
    
    def forward(self, 
                head_motion: torch.Tensor,
                hands_motion: torch.Tensor,
                body_feet_motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            head_motion: (B, 3, T, H, W)
            hands_motion: (B, 90, T, H, W)
            body_feet_motion: (B, 66, T, H, W)
        Returns:
            motion_latent: (B, 3, T, H, W) - concatenated motion latents
            camera_latent: (B, C, T, H, W) - camera view-change information
        """
        # Encode each body part separately
        head_latent = self.head_encoder(head_motion)
        hands_latent = self.hands_encoder(hands_motion)
        body_feet_latent = self.body_feet_encoder(body_feet_motion)
        
        # Concatenate part-wise latents along channel dimension
        motion_latent = torch.cat([head_latent, hands_latent, body_feet_latent], dim=1)
        
        # Encode camera information from head motion
        # Extract head rotation (assume first 3 channels are rotation params)
        head_rotation = head_motion[:, :3].mean(dim=[3, 4])  # (B, 3, T)
        head_rotation = head_rotation.permute(0, 2, 1)  # (B, T, 3)
        camera_latent = self.camera_encoder(head_rotation)
        
        return motion_latent, camera_latent


class PointMapEncoder(nn.Module):
    """
    Point Map Encoder with Adapter
    Encodes 4D scene point maps for scene-consistent generation
    """
    def __init__(self, input_channels: int = 3, output_channels: int = 64):
        super().__init__()
        
        # Base encoder
        self.encoder = nn.Sequential(
            Conv3DBlock(input_channels, 64),
            Conv3DBlock(64, 128),
            Conv3DBlock(128, 256),
            Conv3DBlock(256, 512)
        )
        
        # Adapter: 5 layers of 3D convolutions
        self.adapter = nn.Sequential(
            Conv3DBlock(512, 256),
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 64),
            Conv3DBlock(64, output_channels),
            nn.Conv3d(output_channels, output_channels, kernel_size=1)
        )
    
    def forward(self, point_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_maps: (B, 3, T, H, W) - point map sequence
        Returns:
            point_latent: (B, 64, T, H, W)
        """
        features = self.encoder(point_maps)
        latent = self.adapter(features)
        return latent


class VAE3DEncoder(nn.Module):
    """3D VAE Encoder for first frame"""
    def __init__(self, in_channels: int = 3, latent_channels: int = 4):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv3d(256, latent_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DiffusionTransformer(nn.Module):
    """
    Simplified Diffusion Transformer (DiT) backbone
    Based on the architecture mentioned in the paper
    """
    def __init__(self, 
                 input_channels: int,
                 hidden_dim: int = 1024,
                 num_layers: int = 28,
                 num_heads: int = 16):
        super().__init__()
        
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, input_channels)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input latent
            t: (B,) timestep
        Returns:
            output: (B, C, T, H, W) denoised latent
        """
        B, C, T, H, W = x.shape
        
        # Reshape to sequence: (B, T*H*W, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)
        
        # Project input
        x = self.input_proj(x)
        
        # Add time embedding
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1)
        x = x + t_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Project output
        x = self.output_proj(x)
        
        # Reshape back: (B, C, T, H, W)
        x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return x


class PlayerOne(nn.Module):
    """
    PlayerOne: Egocentric World Simulator
    
    Main architecture that combines:
    - Part-disentangled Motion Injection (PMI)
    - Scene-frame Reconstruction (SR)
    - Diffusion Transformer
    """
    def __init__(self, 
                 video_channels: int = 4,
                 point_channels: int = 64,
                 dit_hidden_dim: int = 1024,
                 dit_num_layers: int = 28):
        super().__init__()
        
        # First frame encoder
        self.frame_encoder = VAE3DEncoder(in_channels=3, latent_channels=video_channels)
        
        # Part-disentangled motion injection
        self.pmi = PartDisentangledMotionInjection()
        
        # Point map encoder for scene reconstruction
        self.point_encoder = PointMapEncoder(
            input_channels=3, 
            output_channels=point_channels
        )
        
        # Calculate total input channels for DiT
        # frame_latent + motion_latent + video_latent + point_latent
        total_channels = video_channels + 3 + video_channels + point_channels
        
        # Diffusion Transformer
        self.dit = DiffusionTransformer(
            input_channels=total_channels,
            hidden_dim=dit_hidden_dim,
            num_layers=dit_num_layers
        )
        
        self.video_channels = video_channels
        self.point_channels = point_channels
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to latent"""
        # Flow matching noise schedule: zt = σt·z0 + βt·ε
        sigma_t = 1 - t.view(-1, 1, 1, 1, 1)
        beta_t = t.view(-1, 1, 1, 1, 1)
        epsilon = torch.randn_like(x)
        return sigma_t * x + beta_t * epsilon, epsilon
    
    def forward(self, 
                first_frame: torch.Tensor,
                head_motion: torch.Tensor,
                hands_motion: torch.Tensor,
                body_feet_motion: torch.Tensor,
                video_latent: torch.Tensor,
                point_maps: Optional[torch.Tensor] = None,
                timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            first_frame: (B, 3, 1, H, W) - first egocentric frame
            head_motion: (B, 3, T, H, W) - head motion parameters
            hands_motion: (B, 90, T, H, W) - hands motion parameters
            body_feet_motion: (B, 66, T, H, W) - body and feet motion
            video_latent: (B, C_v, T, H, W) - noised video latent
            point_maps: (B, 3, T, H, W) - point map sequence (optional, training only)
            timestep: (B,) - diffusion timestep
            
        Returns:
            predicted_noise: (B, C_v+C_p, T, H, W) if training else (B, C_v, T, H, W)
        """
        B = first_frame.shape[0]
        
        if timestep is None:
            timestep = torch.rand(B, device=first_frame.device)
        
        # 1. Encode first frame
        frame_latent = self.frame_encoder(first_frame)  # (B, C_v, 1, H, W)
        
        # 2. Part-disentangled motion injection
        motion_latent, camera_latent = self.pmi(head_motion, hands_motion, body_feet_motion)
        
        # 3. Inject camera information into video latent
        video_latent_with_camera = video_latent + camera_latent
        
        # 4. Scene-frame reconstruction (if point maps provided)
        if point_maps is not None:
            point_latent = self.point_encoder(point_maps)
            
            # Add noise to point latent
            point_latent_noised, _ = self.add_noise(point_latent, timestep)
            
            # Concatenate all latents
            all_latents = torch.cat([
                frame_latent.expand(-1, -1, video_latent.shape[2], -1, -1),
                motion_latent,
                video_latent_with_camera,
                point_latent_noised
            ], dim=1)
        else:
            # Inference mode: no point maps needed
            all_latents = torch.cat([
                frame_latent.expand(-1, -1, video_latent.shape[2], -1, -1),
                motion_latent,
                video_latent_with_camera,
                torch.zeros(B, self.point_channels, *video_latent.shape[2:], 
                          device=video_latent.device)
            ], dim=1)
        
        # 5. Diffusion transformer denoising
        predicted_noise = self.dit(all_latents, timestep)
        
        return predicted_noise
    
    def training_step(self, 
                     first_frame: torch.Tensor,
                     head_motion: torch.Tensor,
                     hands_motion: torch.Tensor,
                     body_feet_motion: torch.Tensor,
                     clean_video_latent: torch.Tensor,
                     point_maps: torch.Tensor) -> torch.Tensor:
        """
        Training step with joint video-point reconstruction
        
        Returns:
            loss: scalar loss value
        """
        B = first_frame.shape[0]
        device = first_frame.device
        
        # Sample timestep
        t = torch.rand(B, device=device)
        
        # Add noise to video and point latents
        video_noised, video_noise = self.add_noise(clean_video_latent, t)
        
        # Forward pass
        predicted_noise = self.forward(
            first_frame, head_motion, hands_motion, body_feet_motion,
            video_noised, point_maps, t
        )
        
        # Calculate loss (only on video latent part)
        video_pred = predicted_noise[:, :self.video_channels]
        loss = F.mse_loss(video_pred, video_noise)
        
        return loss


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = PlayerOne(
        video_channels=4,
        point_channels=64,
        dit_hidden_dim=1024,
        dit_num_layers=28
    )
    
    # Example inputs
    batch_size = 2
    T = 49  # 6 seconds at 8 FPS
    H, W = 60, 60  # Latent spatial dimensions (480/8)
    
    first_frame = torch.randn(batch_size, 3, 1, H, W)
    head_motion = torch.randn(batch_size, 3, T, H, W)
    hands_motion = torch.randn(batch_size, 90, T, H, W)
    body_feet_motion = torch.randn(batch_size, 66, T, H, W)
    video_latent = torch.randn(batch_size, 4, T, H, W)
    point_maps = torch.randn(batch_size, 3, T, H, W)
    
    # Training mode
    loss = model.training_step(
        first_frame, head_motion, hands_motion, body_feet_motion,
        video_latent, point_maps
    )
    print(f"Training loss: {loss.item():.4f}")
    
    # Inference mode (no point maps needed)
    with torch.no_grad():
        output = model(
            first_frame, head_motion, hands_motion, body_feet_motion,
            video_latent, point_maps=None
        )
    print(f"Output shape: {output.shape}")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
