from fastapi import FastAPI
from pydantic import BaseModel
from main import generate_music  # Reuse from main.py

class MusicRequest(BaseModel):
    prompt: str
    neural_input: dict

def create_app():
    app = FastAPI()

    @app.post("/generate")
    async def gen_endpoint(request: MusicRequest):
        output_file = generate_music(request.prompt, request.neural_input, 'temp_output.wav')
        return {"audio_url": output_file}  # Serve via static files or S3

    return app
