"""
Enhanced training script for CosmicTransformer with validation, checkpointing, and logging.
"""

from models.transformer import CosmicTransformer
from data.loader import AudioTextDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    dataset: AudioTextDataset,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    validation_split: float = 0.2,
    checkpoint_dir: str = "models",
    checkpoint_name: str = "checkpoint.pth",
    resume_from: Optional[str] = None,
    save_every: int = 5,
    device: Optional[str] = None
):
    """
    Train CosmicTransformer model with validation and checkpointing.
    
    Args:
        dataset: AudioTextDataset instance
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
        checkpoint_dir: Directory to save checkpoints
        checkpoint_name: Name of checkpoint file
        resume_from: Path to checkpoint to resume from
        save_every: Save checkpoint every N epochs
        device: Device to train on ('cuda', 'cpu', or None for auto)
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Split dataset into train and validation
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset split: {train_size} training, {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    model = CosmicTransformer()
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (audio, text, neural) in enumerate(pbar):
            # Move to device
            audio = audio.to(device)
            text = text.to(device)
            
            # Forward pass
            loss = model.train_step(audio, text, neural)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for audio, text, neural in pbar:
                audio = audio.to(device)
                text = text.to(device)
                
                loss = model.train_step(audio, text, neural)
                val_loss += loss.item()
                val_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or avg_val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            
            # Save regular checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")
    
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final checkpoint saved to: {checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs': epochs
        }, f, indent=2)
    
    return model, train_losses, val_losses
