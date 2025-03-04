from datetime import datetime
from pathlib import Path
import logging
import os
import torch
import io
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella"
    
class AudioSpeechRequest(BaseModel):
    model: str = "kokoro-v1_0"
    input: str
    voice: str = "af_bella"

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model and voice definitions
MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_FILE = "kokoro-v1_0.pth"
VOICES = {f"voices/{voice}.pt" for voice in [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"
]}
REQUIRED_FILES = [MODEL_FILE] + list(VOICES)

VOICE_DESCRIPTIONS = {v.split("/")[-1].replace(".pt", ""): v for v in VOICES}

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global variables for preloaded model and pipeline
model = None
pipeline = None

def load_files():
    """Downloads missing model files from Hugging Face."""
    if all(Path(f).exists() for f in REQUIRED_FILES):
        logging.info("All required files are present.")
        return
    
    logging.info("Downloading missing model files...")
    os.makedirs("voices", exist_ok=True)
    for file in REQUIRED_FILES:
        if not Path(file).exists():
            hf_hub_download(repo_id=MODEL_REPO, filename=file, local_dir=".")
    logging.info("Model files downloaded successfully.")

def load_model():
    """Loads the model once at startup for reuse."""
    global model, pipeline
    if model is not None and pipeline is not None:
        return  # Already loaded
    
    from kokoro import KModel, KPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = KModel().to(device)
    state_dict = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict({
        f"{comp}.{key[7:]}": value for comp, subdict in state_dict.items() for key, value in subdict.items() if key.startswith("module.")
    })
    model.eval()

    pipeline = KPipeline("a", model)
    logging.info("Model loaded successfully.")
    
@app.get("/voices")
async def voices():
    """Returns a list of all available voices."""
    return {"voices": list(VOICE_DESCRIPTIONS.keys())}


@app.post("/tts")
async def tts(request: TTSRequest) -> StreamingResponse:
    """Processes TTS requests as quickly as possible."""
    
    if model is None or pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if request.voice not in VOICE_DESCRIPTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid voice '{request.voice}'. Available voices: {list(VOICE_DESCRIPTIONS.keys())}")

    logging.info(f"\nGenerating speech for '{request.text}' with voice '{request.voice}'")
    start_time = datetime.now()

    try:
        audio_data = []
        for i, (_, _, audio) in enumerate(pipeline(request.text, voice=request.voice, speed=1)):
            audio_data.extend(audio.detach().cpu().numpy())

        # Stream audio directly from memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format="WAV")
        buffer.seek(0)

        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"Speech generation completed in {duration:.2f} seconds.")

        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

@app.post("/v1/audio/speech")
async def openaitts(request: AudioSpeechRequest) -> StreamingResponse:
    """OpenAI-compatible endpoint for TTS"""
    
    if model is None or pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if request.voice not in VOICE_DESCRIPTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid voice '{request.voice}'. Available voices: {list(VOICE_DESCRIPTIONS.keys())}")

    logging.info(f"\nGenerating speech for '{request.text}' with voice '{request.voice}'")
    start_time = datetime.now()

    try:
        audio_data = []
        for i, (_, _, audio) in enumerate(pipeline(request.text, voice=request.voice, speed=1)):
            audio_data.extend(audio.detach().cpu().numpy())

        # Stream audio directly from memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format="WAV")
        buffer.seek(0)

        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"Speech generation completed in {duration:.2f} seconds.")

        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

if __name__ == "__main__":
    load_files()  # Ensure model files are available
    load_model()  # Load model once
    uvicorn.run(app, host="0.0.0.0", port=8080)