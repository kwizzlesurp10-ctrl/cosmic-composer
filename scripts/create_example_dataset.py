"""
Create an example dataset for testing Cosmic Composer training.
Generates synthetic audio files and metadata.
"""

import os
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path


def generate_synthetic_audio(
    duration: float = 8.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    noise_level: float = 0.1
) -> torch.Tensor:
    """
    Generate synthetic audio for testing.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        frequency: Base frequency in Hz
        noise_level: Amount of noise to add
        
    Returns:
        Audio tensor [samples]
    """
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a simple tone with harmonics
    audio = torch.sin(2 * np.pi * frequency * t)
    audio += 0.5 * torch.sin(2 * np.pi * frequency * 2 * t)
    audio += 0.25 * torch.sin(2 * np.pi * frequency * 3 * t)
    
    # Add noise
    audio += noise_level * torch.randn_like(audio)
    
    # Normalize
    audio = audio / (audio.abs().max() + 1e-8)
    
    return audio


def create_example_dataset(
    output_dir: str = "example_dataset",
    num_samples: int = 20,
    sample_rate: int = 44100,
    duration: float = 8.0
):
    """
    Create an example dataset with synthetic audio files.
    
    Args:
        output_dir: Output directory for dataset
        num_samples: Number of audio samples to generate
        sample_rate: Audio sample rate
        duration: Duration of each audio sample in seconds
    """
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Example descriptions and neural inputs
    descriptions = [
        "upbeat electronic music with synthesizers",
        "calm ambient soundscape",
        "energetic rock guitar solo",
        "peaceful piano melody",
        "driving techno beat",
        "soft acoustic guitar",
        "intense orchestral piece",
        "relaxing nature sounds",
        "fast-paced drum and bass",
        "melancholic violin solo",
        "happy pop song",
        "dark atmospheric music",
        "bright cheerful melody",
        "mysterious ambient track",
        "powerful orchestral crescendo",
        "gentle lullaby",
        "aggressive metal riff",
        "smooth jazz saxophone",
        "ethereal choir voices",
        "pulsing electronic rhythm"
    ]
    
    # Generate neural inputs (heart rate varies with music style)
    neural_inputs = []
    for i in range(num_samples):
        # Simulate heart rate based on music style
        if i < num_samples // 3:
            hr = np.random.randint(80, 120)  # Upbeat/energetic
        elif i < 2 * num_samples // 3:
            hr = np.random.randint(60, 80)   # Moderate
        else:
            hr = np.random.randint(50, 70)  # Calm/relaxing
        
        neural_inputs.append({"heart_rate": hr})
    
    metadata = []
    
    print(f"Creating example dataset with {num_samples} samples...")
    
    for idx in range(num_samples):
        # Generate synthetic audio with varying frequencies
        base_freq = 220 + (idx % 10) * 44  # Vary frequency
        audio = generate_synthetic_audio(
            duration=duration,
            sample_rate=sample_rate,
            frequency=base_freq,
            noise_level=0.05 + (idx % 5) * 0.02
        )
        
        # Save audio file
        audio_filename = f"example_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename
        torchaudio.save(
            str(audio_path),
            audio.unsqueeze(0),
            sample_rate
        )
        
        # Create metadata entry
        entry = {
            "audio_file": audio_filename,
            "text": descriptions[idx % len(descriptions)],
            "neural": neural_inputs[idx]
        }
        metadata.append(entry)
        
        if (idx + 1) % 5 == 0:
            print(f"  Created {idx + 1}/{num_samples} samples...")
    
    # Write metadata file
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nExample dataset created at: {output_path}")
    print(f"  - {num_samples} audio files in {audio_dir}")
    print(f"  - Metadata file: {metadata_path}")
    print(f"\nYou can now train with:")
    print(f"  python main.py train --dataset {output_path} --epochs 10 --batch-size 4")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Create example dataset for Cosmic Composer")
    parser.add_argument('--output-dir', type=str, default='example_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of audio samples to generate')
    parser.add_argument('--duration', type=float, default=8.0,
                       help='Duration of each audio sample in seconds')
    
    args = parser.parse_args()
    
    create_example_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        duration=args.duration
    )
