import argparse
import os
import sys
import json
import logging
from typing import Dict

import torch
import google.generativeai as genai
from fastapi import FastAPI
import uvicorn

# Import project modules (assuming paths are set)
from models.transformer import CosmicTransformer
from models.diffusion import AudioDiffusion  # For hybrid use if needed
from scripts.train import train_model
from api.app import create_app  # FastAPI app factory
from data.loader import AudioTextDataset  # For training

# Setup private logging (in-memory, no external exposure)
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'), stream=sys.stdout)
logger = logging.getLogger(__name__)

# Gemini bootstrap function
def refine_prompt_with_gemini(raw_prompt: str, neural_input: Dict) -> str:
    """Use Gemini-Pro to refine prompt with neural data integration."""
    try:
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        model = genai.GenerativeModel('gemini-pro')
        
        # Calculated neural modulation equation
        if 'heart_rate' in neural_input:
            hr = neural_input['heart_rate']
            tempo_adjust = 60 + (hr / 150) * 120  # Scale to 60-180 BPM
            neural_desc = f" with tempo around {int(tempo_adjust)} BPM reflecting elevated energy"
        else:
            neural_desc = ""
        
        gemini_prompt = f"Enhance this music prompt for AI generation: '{raw_prompt}'. Incorporate emotional cues{neural_desc}. Output detailed description."
        response = model.generate_content(gemini_prompt)
        return response.text
    except KeyError:
        logger.error("GEMINI_API_KEY not set.")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"Gemini refinement failed: {e}. Using raw prompt.")
        return raw_prompt

# Inference function
def generate_music(prompt: str, neural_input: Dict, output_file: str) -> str:
    """Generate audio using refined prompt and model."""
    refined_prompt = refine_prompt_with_gemini(prompt, neural_input)
    logger.info(f"Refined prompt: {refined_prompt}")
    
    model = CosmicTransformer.load(os.environ.get('MODEL_CHECKPOINT', 'models/checkpoint.pth'))
    audio = model.generate(refined_prompt, neural_input)  # Assume generate returns torch tensor
    
    # Save audio (using torchaudio or ffmpeg)
    import torchaudio
    torchaudio.save(output_file, audio, sample_rate=44100)
    return output_file

# Main CLI parser
def main():
    parser = argparse.ArgumentParser(description="Cosmic Composer CLI: Bootstrap AI music framework with Gemini-Pro integration.")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', required=True, help='Path to dataset')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=16)

    # Generate subcommand
    gen_parser = subparsers.add_parser('generate', help='Generate music from prompt')
    gen_parser.add_argument('--prompt', required=True, help='Text prompt for music')
    gen_parser.add_argument('--neural-input', type=str, default='{}', help='JSON neural input, e.g. {"heart_rate": 90}')
    gen_parser.add_argument('--output-file', default='output.wav', help='Output audio file')

    # API subcommand
    api_parser = subparsers.add_parser('api', help='Start FastAPI server')
    api_parser.add_argument('--host', default='0.0.0.0')
    api_parser.add_argument('--port', type=int, default=8000)

    # Test subcommand
    test_parser = subparsers.add_parser('test', help='Run tests')

    args = parser.parse_args()

    if args.mode == 'train':
        dataset = AudioTextDataset(args.dataset)
        train_model(
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.mode == 'generate':
        neural_input = json.loads(args.neural_input)
        output = generate_music(args.prompt, neural_input, args.output_file)
        logger.info(f"Generated audio saved to {output}")
    elif args.mode == 'api':
        app: FastAPI = create_app()  # Factory to include models/routes
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.mode == 'test':
        import pytest
        sys.exit(pytest.main(['tests/']))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
