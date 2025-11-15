# Training Guide for Cosmic Composer

This guide explains how to train the CosmicTransformer model with your custom datasets.

## Quick Start

### 1. Create an Example Dataset (for testing)

```bash
python scripts/create_example_dataset.py --output-dir example_dataset --num-samples 20
```

This creates a synthetic dataset with 20 audio samples for testing the training pipeline.

### 2. Train the Model

```bash
python main.py train \
    --dataset example_dataset \
    --epochs 10 \
    --batch-size 4
```

## Dataset Structure

Your dataset should be organized as follows:

```
your_dataset/
├── audio/
│   ├── audio_00000.wav
│   ├── audio_00001.wav
│   └── ...
└── metadata.jsonl
```

### Metadata Format

Each line in `metadata.jsonl` should be a JSON object:

```json
{"audio_file": "audio_00000.wav", "text": "upbeat electronic music", "neural": {"heart_rate": 90}}
{"audio_file": "audio_00001.wav", "text": "calm ambient soundscape", "neural": {"heart_rate": 65}}
```

Fields:
- `audio_file`: Filename (must be in the `audio/` subdirectory)
- `text`: Text description of the music
- `neural`: Dictionary with neural input features (e.g., `{"heart_rate": 90}`)

## Preparing Your Own Dataset

### Option 1: From a Directory of Audio Files

If you have a directory with audio files:

```bash
python scripts/prepare_dataset.py \
    --mode directory \
    --audio-dir /path/to/your/audio/files \
    --output-dir my_dataset \
    --heart-rate 75
```

This will:
- Copy all audio files to `my_dataset/audio/`
- Create `metadata.jsonl` with default descriptions extracted from filenames
- Set default heart rate for all samples

### Option 2: Manual Preparation

1. Create dataset directory:
```bash
mkdir my_dataset
mkdir my_dataset/audio
```

2. Copy your audio files to `my_dataset/audio/`

3. Create `my_dataset/metadata.jsonl`:
```bash
# Edit metadata.jsonl
{"audio_file": "file1.wav", "text": "your description", "neural": {"heart_rate": 90}}
{"audio_file": "file2.wav", "text": "another description", "neural": {"heart_rate": 75}}
```

### Option 3: Programmatic Creation

```python
from scripts.prepare_dataset import create_metadata_from_list

audio_files = ["song1.wav", "song2.wav"]
descriptions = ["energetic rock", "calm jazz"]
neural_inputs = [{"heart_rate": 100}, {"heart_rate": 60}]

create_metadata_from_list(
    audio_files,
    descriptions,
    neural_inputs,
    output_dir="my_dataset"
)
```

## Training Options

### Basic Training

```bash
python main.py train \
    --dataset /path/to/dataset \
    --epochs 50 \
    --batch-size 16
```

### Advanced Training with Custom Settings

The training script supports many options. You can modify `scripts/train.py` or use it directly:

```python
from scripts.train import train_model
from data.loader import AudioTextDataset

dataset = AudioTextDataset("/path/to/dataset")
model, train_losses, val_losses = train_model(
    dataset=dataset,
    epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    validation_split=0.2,
    checkpoint_dir="models",
    save_every=5,
    device="cuda"  # or "cpu"
)
```

### Resuming Training

To resume from a checkpoint:

```python
from scripts.train import train_model
from data.loader import AudioTextDataset

dataset = AudioTextDataset("/path/to/dataset")
model, train_losses, val_losses = train_model(
    dataset=dataset,
    epochs=50,
    resume_from="models/checkpoint.pth"
)
```

## Training Features

The enhanced training script includes:

- **Validation Split**: Automatically splits dataset into train/validation (default 80/20)
- **Progress Bars**: Visual progress with tqdm
- **Checkpointing**: Saves checkpoints every N epochs and best model
- **Learning Rate Scheduling**: Reduces LR when validation loss plateaus
- **Gradient Clipping**: Prevents exploding gradients
- **Training History**: Saves loss curves to JSON
- **GPU Support**: Automatic CUDA detection and usage
- **Resume Capability**: Resume training from checkpoints

## Checkpoints

Training saves:
- `models/checkpoint.pth`: Latest checkpoint
- `models/best_model.pth`: Best model (lowest validation loss)
- `models/training_history.json`: Training loss curves

## Monitoring Training

### View Training History

```python
import json
import matplotlib.pyplot as plt

with open('models/training_history.json') as f:
    history = json.load(f)

plt.plot(history['train_losses'], label='Train')
plt.plot(history['val_losses'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Check Logs

Training logs are printed to console with:
- Epoch progress
- Train/validation loss
- Learning rate
- Best model saves

## Tips for Better Training

1. **Dataset Size**: More data = better results. Aim for at least 100+ samples.

2. **Audio Quality**: Use high-quality audio files (44.1kHz sample rate recommended).

3. **Descriptions**: Write detailed, consistent descriptions. Include:
   - Genre/style
   - Mood/emotion
   - Instruments
   - Tempo/energy level

4. **Neural Input**: Use realistic heart rate values:
   - Calm music: 50-70 BPM
   - Moderate: 70-90 BPM
   - Energetic: 90-120+ BPM

5. **Batch Size**: Adjust based on GPU memory:
   - GPU with 8GB: batch_size=4-8
   - GPU with 16GB+: batch_size=16-32

6. **Learning Rate**: Start with 1e-4. Reduce if loss is unstable.

7. **Early Stopping**: Monitor validation loss. Stop if it stops improving.

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `max_audio_length` in dataset
- Use gradient accumulation

### Loss Not Decreasing
- Check data quality
- Reduce learning rate
- Increase model capacity
- Add more training data

### Slow Training
- Use GPU (CUDA)
- Increase `num_workers` in DataLoader
- Use mixed precision training

## Next Steps

After training:
1. Test generation: `python main.py generate --prompt "your prompt" --neural-input '{"heart_rate": 80}'`
2. Start API server: `python main.py api`
3. Fine-tune hyperparameters based on results
