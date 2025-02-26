# Kokoro TTS Server

A server for [Kokoro TTS](https://github.com/hexgrad/kokoro), an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers quality comparable to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed in production environments or personal projects.  

This server is built with FastAPI and provides an API for generating speech from text using pre-trained voices.

## Features
- **Multiple voices** from the Kokoro-82M model  
- **Streaming audio output** for real-time applications  
- **Automatic model downloading** from Hugging Face Hub  
- **Supports both CPU and GPU inference**  
- **Cross-Origin Resource Sharing (CORS) enabled**  

## Installation

### Clone the Repository
```bash
git clone https://github.com/your-repo/kokoro-tts.git
cd kokoro-tts
```

### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Server

### Start the FastAPI Server
```bash
python server.py
```
The server will start at `http://0.0.0.0:8080`.

## API Endpoints

### Get Available Voices
**Endpoint:** `GET /voices`  

**Response:**
```json
{
  "voices": ["af_heart", "af_alloy", "af_aoede", "am_adam", "am_echo", "am_fenrir"]
}
```

### Generate Speech
**Endpoint:** `POST /tts`  

**Parameters:**
- `text` (string, required): The text to synthesize.  
- `voice` (string, optional, default: `af_bella`): The voice model to use.  

**Example Request:**
```bash
curl -X POST "http://localhost:8080/tts" -d "text=Hello, how are you?" -d "voice=af_bella" --output output.wav
```

## Configuration

### Running on GPU
If a CUDA-compatible GPU is available, the model will automatically use it for inference. Otherwise, it defaults to CPU.

### Updating Model Files
The required model files are automatically downloaded from Hugging Face if they are missing. They are stored in the root directory under `voices/`.

## Logging
Logging is enabled by default and provides real-time updates on requests, model loading, and speech generation performance.
