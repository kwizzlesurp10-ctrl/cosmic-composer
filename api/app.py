from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os

class MusicRequest(BaseModel):
    prompt: str
    neural_input: dict

# Preload model on app startup for better performance
_app_model = None

def create_app():
    app = FastAPI()
    
    @app.on_event("startup")
    async def startup_event():
        """Preload model on startup to avoid cold start delays."""
        global _app_model
        from models.cache import get_cached_model
        checkpoint_path = os.environ.get('MODEL_CHECKPOINT', 'models/checkpoint.pth')
        _app_model = get_cached_model(checkpoint_path)

    @app.post("/generate")
    async def gen_endpoint(request: MusicRequest):
        from main import generate_music
        output_file = generate_music(request.prompt, request.neural_input, 'temp_output.wav')
        return {"audio_url": output_file}  # Serve via static files or S3

    return app
