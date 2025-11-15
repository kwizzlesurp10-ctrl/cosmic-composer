"""
AudioDiffusion: Diffusion model for high-quality audio generation.
Can be used standalone or in hybrid mode with CosmicTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AudioDiffusion(nn.Module):
    """
    Diffusion model for audio generation.
    Can be used for post-processing or standalone generation.
    """
    
    def __init__(
        self,
        audio_dim: int = 128,
        hidden_dim: int = 256,
        num_timesteps: int = 1000
    ):
        """
        Initialize AudioDiffusion model.
        
        Args:
            audio_dim: Audio feature dimension
            hidden_dim: Hidden dimension for diffusion
            num_timesteps: Number of diffusion timesteps
        """
        super().__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(audio_dim + 1, hidden_dim),  # +1 for timestep
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, audio_dim)
        )
        
        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise at timestep t.
        
        Args:
            x: Noisy audio [batch, audio_dim]
            t: Timestep [batch]
            
        Returns:
            Predicted noise [batch, audio_dim]
        """
        # Embed timestep
        t_emb = self.time_embedding(t.float().unsqueeze(-1))
        
        # Concatenate with input
        x_t = torch.cat([x, t.float().unsqueeze(-1) / self.num_timesteps], dim=-1)
        
        # Predict noise
        noise_pred = self.noise_predictor(x_t)
        
        return noise_pred
    
    def sample(self, shape: tuple, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample audio using diffusion process.
        
        Args:
            shape: Output shape [batch, audio_dim]
            device: Device to sample on
            
        Returns:
            Generated audio [batch, audio_dim]
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process
        with torch.no_grad():
            for t in range(self.num_timesteps - 1, -1, -1):
                t_tensor = torch.full((shape[0],), t, device=device)
                noise_pred = self.forward(x, t_tensor)
                
                # Denoise step (simplified DDPM)
                alpha = 1.0 / self.num_timesteps
                x = x - alpha * noise_pred
        
        return x
