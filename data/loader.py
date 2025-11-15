"""
AudioTextDataset: Dataset loader for audio-text pairs with neural input.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import numpy as np


class AudioTextDataset(Dataset):
    """
    Dataset for loading audio-text pairs with optional neural input.
    
    Expected dataset structure:
    dataset/
        audio/
            file1.wav
            file2.wav
            ...
        metadata.jsonl  # Each line: {"audio_file": "file1.wav", "text": "description", "neural": {...}}
    """
    
    def __init__(
        self,
        dataset_path: str,
        sample_rate: int = 44100,
        max_audio_length: int = 44100 * 8,  # 8 seconds default
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_path: Path to dataset directory
            sample_rate: Audio sample rate
            max_audio_length: Maximum audio length in samples
            transform: Optional transform to apply
        """
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.transform = transform
        
        # Load metadata
        metadata_file = os.path.join(dataset_path, 'metadata.jsonl')
        if not os.path.exists(metadata_file):
            # Create dummy metadata if file doesn't exist
            self.metadata = self._create_dummy_metadata()
        else:
            self.metadata = self._load_metadata(metadata_file)
        
        self.audio_dir = os.path.join(dataset_path, 'audio')
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir, exist_ok=True)
    
    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Load metadata from JSONL file."""
        metadata = []
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        return metadata
    
    def _create_dummy_metadata(self) -> List[Dict]:
        """Create dummy metadata for testing."""
        return [
            {
                "audio_file": "dummy1.wav",
                "text": "upbeat electronic music",
                "neural": {"heart_rate": 90}
            },
            {
                "audio_file": "dummy2.wav",
                "text": "calm ambient soundscape",
                "neural": {"heart_rate": 65}
            }
        ]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get item from dataset.
        
        Returns:
            tuple: (audio_tensor, text_tokens, neural_dict)
        """
        item = self.metadata[idx]
        
        # Load audio
        audio_file = os.path.join(self.audio_dir, item['audio_file'])
        audio = self._load_audio(audio_file)
        
        # Process text
        text = item.get('text', '')
        text_tokens = self._tokenize_text(text)
        
        # Get neural input
        neural = item.get('neural', {})
        
        # Apply transform if provided
        if self.transform:
            audio = self.transform(audio)
        
        return audio, text_tokens, neural
    
    def _load_audio(self, audio_file: str) -> torch.Tensor:
        """
        Load audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Audio tensor [samples]
        """
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_file)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Trim or pad to max_audio_length
            audio = waveform.squeeze(0)
            if audio.size(0) > self.max_audio_length:
                audio = audio[:self.max_audio_length]
            elif audio.size(0) < self.max_audio_length:
                padding = torch.zeros(self.max_audio_length - audio.size(0))
                audio = torch.cat([audio, padding])
            
            return audio
        
        except Exception as e:
            # Return dummy audio if file doesn't exist
            print(f"Warning: Could not load {audio_file}: {e}. Using dummy audio.")
            return torch.randn(self.max_audio_length) * 0.1
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """
        Simple text tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Token tensor [seq_len]
        """
        # Simple hash-based tokenization (replace with proper tokenizer)
        tokens = [hash(c) % 10000 for c in text[:100]]
        if not tokens:
            tokens = [0]
        return torch.tensor(tokens, dtype=torch.long)
