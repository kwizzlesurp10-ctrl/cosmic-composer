# Cosmic Composer

AI-powered music generation system that combines text prompts with neural input (e.g., heart rate) to create personalized music using transformer models and Gemini API integration.

## Features

- **Text-to-Music Generation**: Generate music from natural language prompts
- **Neural Input Integration**: Incorporate biometric data (heart rate, etc.) to modulate music generation
- **Gemini-Powered Prompt Refinement**: Uses Google's Gemini API to enhance music generation prompts
- **Transformer Architecture**: Custom CosmicTransformer model for high-quality audio synthesis
- **REST API**: FastAPI-based web service for music generation
- **Training Pipeline**: Complete training infrastructure for custom datasets

## Installation

```bash
pip install -r requirements.txt
```

## Setup

1. Set environment variables:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export MODEL_CHECKPOINT="models/checkpoint.pth"  # Optional
export LOG_LEVEL="INFO"  # Optional
```

## Usage

### Command Line Interface

#### Generate Music
```bash
python main.py generate \
    --prompt "upbeat electronic music with synthesizers" \
    --neural-input '{"heart_rate": 90}' \
    --output-file output.wav
```

#### Train Model
```bash
python main.py train \
    --dataset /path/to/dataset \
    --epochs 50 \
    --batch-size 16
```

#### Start API Server
```bash
python main.py api --host 0.0.0.0 --port 8000
```

#### Run Tests
```bash
python main.py test
```

### API Usage

Start the server:
```bash
python main.py api
```

Generate music via API:
```bash
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "calm ambient soundscape",
        "neural_input": {"heart_rate": 65}
    }'
```

## Project Structure

```
cosmic-composer/
├── main.py                 # Main CLI entry point
├── api/
│   └── app.py             # FastAPI application
├── models/
│   ├── transformer.py     # CosmicTransformer model
│   └── diffusion.py       # AudioDiffusion model (optional)
├── data/
│   └── loader.py          # Dataset loader
├── scripts/
│   └── train.py           # Training script
└── tests/
    └── test_main.py       # Unit tests
```

## Dataset Format

The dataset should be organized as follows:

```
dataset/
├── audio/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── metadata.jsonl
```

Each line in `metadata.jsonl` should be:
```json
{"audio_file": "file1.wav", "text": "description", "neural": {"heart_rate": 90}}
```

## Neural Input

The system supports various neural inputs:
- `heart_rate`: Heart rate in BPM (affects tempo)
- `feature_0` through `feature_9`: Additional neural features

## Model Architecture

- **CosmicTransformer**: Transformer-based encoder-decoder architecture
- **Neural Integration**: Projects neural features into model space
- **Audio Synthesis**: Multi-layer decoder for waveform generation

## License

MIT License
