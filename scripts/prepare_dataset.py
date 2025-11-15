"""
Dataset preparation script for Cosmic Composer.
Helps organize audio files and create metadata.jsonl file.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import shutil


def create_metadata_from_directory(
    audio_dir: str,
    output_dir: str,
    metadata_file: str = "metadata.jsonl",
    default_neural: Dict = None
):
    """
    Create metadata.jsonl from a directory of audio files.
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Output directory for organized dataset
        metadata_file: Name of metadata file
        default_neural: Default neural input values
    """
    if default_neural is None:
        default_neural = {"heart_rate": 70}
    
    audio_path = Path(audio_dir)
    output_path = Path(output_dir)
    audio_output = output_path / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)
    
    # Supported audio formats
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    metadata = []
    audio_files = [f for f in audio_path.iterdir() 
                   if f.suffix.lower() in audio_extensions]
    
    print(f"Found {len(audio_files)} audio files")
    
    for idx, audio_file in enumerate(audio_files):
        # Copy audio file to organized structure
        new_audio_name = f"audio_{idx:05d}{audio_file.suffix}"
        dest_path = audio_output / new_audio_name
        shutil.copy2(audio_file, dest_path)
        
        # Create metadata entry
        # Try to extract description from filename
        description = audio_file.stem.replace('_', ' ').replace('-', ' ')
        
        entry = {
            "audio_file": new_audio_name,
            "text": description,
            "neural": default_neural.copy()
        }
        metadata.append(entry)
    
    # Write metadata file
    metadata_path = output_path / metadata_file
    with open(metadata_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created dataset at {output_path}")
    print(f"  - {len(metadata)} audio files")
    print(f"  - Metadata file: {metadata_path}")
    return output_path


def create_metadata_from_list(
    audio_files: List[str],
    descriptions: List[str],
    neural_inputs: List[Dict],
    output_dir: str,
    metadata_file: str = "metadata.jsonl"
):
    """
    Create metadata.jsonl from lists of files, descriptions, and neural inputs.
    
    Args:
        audio_files: List of paths to audio files
        descriptions: List of text descriptions (one per audio file)
        neural_inputs: List of neural input dicts (one per audio file)
        output_dir: Output directory for organized dataset
        metadata_file: Name of metadata file
    """
    output_path = Path(output_dir)
    audio_output = output_path / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)
    
    assert len(audio_files) == len(descriptions) == len(neural_inputs), \
        "All lists must have the same length"
    
    metadata = []
    
    for idx, (audio_file, desc, neural) in enumerate(zip(audio_files, descriptions, neural_inputs)):
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"Warning: {audio_file} not found, skipping")
            continue
        
        # Copy audio file
        new_audio_name = f"audio_{idx:05d}{audio_path.suffix}"
        dest_path = audio_output / new_audio_name
        shutil.copy2(audio_path, dest_path)
        
        # Create metadata entry
        entry = {
            "audio_file": new_audio_name,
            "text": desc,
            "neural": neural
        }
        metadata.append(entry)
    
    # Write metadata file
    metadata_path = output_path / metadata_file
    with open(metadata_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created dataset at {output_path}")
    print(f"  - {len(metadata)} audio files")
    print(f"  - Metadata file: {metadata_path}")
    return output_path


def add_neural_data_to_metadata(
    metadata_file: str,
    neural_data: Dict[str, Dict],
    output_file: str = None
):
    """
    Add or update neural data in existing metadata file.
    
    Args:
        metadata_file: Path to existing metadata.jsonl
        neural_data: Dict mapping audio_file to neural input dict
        output_file: Output file path (if None, overwrites input)
    """
    if output_file is None:
        output_file = metadata_file
    
    updated_metadata = []
    
    with open(metadata_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                audio_file = entry['audio_file']
                
                # Update neural data if available
                if audio_file in neural_data:
                    entry['neural'] = neural_data[audio_file]
                
                updated_metadata.append(entry)
    
    # Write updated metadata
    with open(output_file, 'w') as f:
        for entry in updated_metadata:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Updated metadata file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Cosmic Composer")
    parser.add_argument('--mode', choices=['directory', 'list'], required=True,
                       help='Dataset preparation mode')
    parser.add_argument('--audio-dir', type=str,
                       help='Directory containing audio files (for directory mode)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for organized dataset')
    parser.add_argument('--metadata-file', type=str, default='metadata.jsonl',
                       help='Name of metadata file')
    parser.add_argument('--heart-rate', type=int, default=70,
                       help='Default heart rate for neural input')
    
    args = parser.parse_args()
    
    default_neural = {"heart_rate": args.heart_rate}
    
    if args.mode == 'directory':
        if not args.audio_dir:
            parser.error("--audio-dir is required for directory mode")
        create_metadata_from_directory(
            args.audio_dir,
            args.output_dir,
            args.metadata_file,
            default_neural
        )
    else:
        parser.error("List mode not yet implemented via CLI. Use the Python functions directly.")


if __name__ == '__main__':
    main()
