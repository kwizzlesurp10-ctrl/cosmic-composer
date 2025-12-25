"""
CosmicTransformer: Transformer-based model for music generation.
Combines text prompts with neural input (e.g., heart rate) to generate audio.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from functools import lru_cache


# Module-level cached tokenization function
@lru_cache(maxsize=1024)
def _cached_tokenize(text: str) -> tuple:
    """Simple tokenization with caching (replace with proper tokenizer in production)."""
    # Convert text to simple integer tokens
    tokens = tuple(hash(c) % 10000 for c in text[:100])  # Limit to 100 chars, use tuple for caching
    if not tokens:
        tokens = (0,)  # At least one token
    return tokens


class CosmicTransformer(nn.Module):
    """
    Transformer-based model for generating music from text prompts and neural inputs.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 2048,
        audio_dim: int = 128,
        sample_rate: int = 44100,
        dropout: float = 0.1
    ):
        """
        Initialize CosmicTransformer.
        
        Args:
            vocab_size: Vocabulary size for text tokenization
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            max_seq_len: Maximum sequence length
            audio_dim: Audio feature dimension
            sample_rate: Audio sample rate
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.audio_dim = audio_dim
        self.sample_rate = sample_rate
        self._device = None  # Cache device for efficient checks
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # Neural input projection (heart rate, etc.)
        self.neural_projection = nn.Linear(10, d_model)  # Support up to 10 neural features
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Audio generation head
        self.audio_decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, audio_dim * 2),  # Generate audio features
            nn.Tanh()
        )
        
        # Final audio synthesis (convert features to waveform)
        self.audio_synthesizer = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, sample_rate // 10)  # Generate chunks of audio
        )
    
    def forward(self, text_tokens: torch.Tensor, neural_input: Dict) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_tokens: Tokenized text input [batch, seq_len]
            neural_input: Dictionary with neural features (e.g., {'heart_rate': 90})
            
        Returns:
            Generated audio tensor [batch, audio_samples]
        """
        batch_size = text_tokens.size(0)
        
        # Embed text
        text_emb = self.text_embedding(text_tokens)  # [batch, seq_len, d_model]
        
        # Process neural input
        neural_features = self._extract_neural_features(neural_input, batch_size)
        neural_emb = self.neural_projection(neural_features)  # [batch, d_model]
        neural_emb = neural_emb.unsqueeze(1)  # [batch, 1, d_model]
        
        # Concatenate neural input as a special token
        x = torch.cat([neural_emb, text_emb], dim=1)  # [batch, seq_len+1, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer(x)  # [batch, seq_len+1, d_model]
        
        # Use the neural token's output for audio generation
        audio_features = self.audio_decoder(encoded[:, 0])  # [batch, audio_dim*2]
        audio_features = audio_features.view(batch_size, -1, self.audio_dim)
        
        # Synthesize audio waveform
        audio = self.audio_synthesizer(audio_features)  # [batch, chunks, samples_per_chunk]
        audio = audio.view(batch_size, -1)  # [batch, total_samples]
        
        return audio
    
    def _get_device(self) -> torch.device:
        """Get model device efficiently with caching."""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    def _extract_neural_features(self, neural_input: Dict, batch_size: int) -> torch.Tensor:
        """Extract neural features into a tensor."""
        features = []
        
        # Heart rate (normalized)
        hr = neural_input.get('heart_rate', 70.0)
        features.append((hr - 60) / 100.0)  # Normalize around 60-160 BPM
        
        # Add other neural features (placeholder for future expansion)
        for i in range(9):
            key = f'feature_{i}'
            features.append(neural_input.get(key, 0.0))
        
        # Create tensor on correct device directly
        device = self._get_device()
        neural_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        neural_tensor = neural_tensor.unsqueeze(0).expand(batch_size, -1)
        
        return neural_tensor
    
    def generate(self, prompt: str, neural_input: Dict, max_length: int = 44100 * 8) -> torch.Tensor:
        """
        Generate audio from text prompt and neural input.
        
        Args:
            prompt: Text prompt describing desired music
            neural_input: Dictionary with neural features
            max_length: Maximum audio length in samples
            
        Returns:
            Generated audio tensor [1, samples]
        """
        self.eval()
        
        # Use cached tokenization
        tokens = list(_cached_tokenize(prompt))
        text_tokens = torch.tensor([tokens], dtype=torch.long, device=self._get_device())
        
        with torch.no_grad():
            audio = self.forward(text_tokens, neural_input)
        
        # Trim or pad to desired length
        if audio.size(1) > max_length:
            audio = audio[:, :max_length]
        elif audio.size(1) < max_length:
            padding = torch.zeros(1, max_length - audio.size(1), device=audio.device)
            audio = torch.cat([audio, padding], dim=1)
        
        return audio.squeeze(0)  # Remove batch dimension
    
    def train_step(self, audio: torch.Tensor, text: torch.Tensor, neural: Dict) -> torch.Tensor:
        """
        Training step.
        
        Args:
            audio: Ground truth audio [batch, samples]
            text: Text tokens [batch, seq_len]
            neural: Neural input dictionary
            
        Returns:
            Loss tensor
        """
        self.train()
        
        # Forward pass
        generated_audio = self.forward(text, neural)
        
        # Simple MSE loss (can be enhanced with perceptual losses)
        loss = F.mse_loss(generated_audio, audio)
        
        return loss
    
    @classmethod
    def load(cls, checkpoint_path: str) -> 'CosmicTransformer':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model instance
        """
        if not os.path.exists(checkpoint_path):
            # Return a new model if checkpoint doesn't exist
            model = cls()
            print(f"Checkpoint not found at {checkpoint_path}, using new model.")
            return model
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = cls()
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
