# Quick Start: Training Cosmic Composer

## Step 1: Create Example Dataset (for testing)

```bash
python scripts/create_example_dataset.py --output-dir example_dataset --num-samples 20
```

This creates a test dataset with 20 synthetic audio samples.

## Step 2: Train the Model

```bash
python main.py train --dataset example_dataset --epochs 10 --batch-size 4
```

## Step 3: Check Results

Training will create:
- `models/checkpoint.pth` - Latest checkpoint
- `models/best_model.pth` - Best model
- `models/training_history.json` - Loss curves

## Using Your Own Dataset

### Prepare Dataset

1. **Organize your files:**
```
my_dataset/
├── audio/
│   ├── song1.wav
│   ├── song2.wav
│   └── ...
└── metadata.jsonl
```

2. **Create metadata.jsonl:**
```json
{"audio_file": "song1.wav", "text": "upbeat electronic music", "neural": {"heart_rate": 90}}
{"audio_file": "song2.wav", "text": "calm ambient soundscape", "neural": {"heart_rate": 65}}
```

3. **Or use the preparation script:**
```bash
python scripts/prepare_dataset.py \
    --mode directory \
    --audio-dir /path/to/audio/files \
    --output-dir my_dataset
```

### Train

```bash
python main.py train --dataset my_dataset --epochs 50 --batch-size 16
```

## Advanced Usage

See `TRAINING_GUIDE.md` for:
- Custom training parameters
- Resuming from checkpoints
- Monitoring training progress
- Troubleshooting tips
