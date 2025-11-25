"""
FastAPI Backend for Custom ASR and Summarization Models
Provides endpoints for transcription and summarization with Supabase integration
"""
import os
import sys
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import soundfile as sf
import librosa
import numpy as np
from typing import List
import json
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix imports - use absolute imports instead of relative
from supabase_connector import SupabaseConnector
from asr_handler import ASRHandler
from summarizer_handler import SummarizerHandler

app = FastAPI(title="AuthentiX Voice-to-Notes API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
supabase_connector = None
asr_handler = None
summarizer_handler = None

# Store active WebSocket connections for streaming
active_connections: List[WebSocket] = []

class TextInput(BaseModel):
    text: str

class TranscriptionResponse(BaseModel):
    transcript: str

class SummaryResponse(BaseModel):
    summary: str

class VoiceToNotesResponse(BaseModel):
    transcript: str
    summary: str

class StreamingTranscriptionResponse(BaseModel):
    partial_transcript: str
    is_final: bool

@app.on_event("startup")
async def startup_event():
    """Initialize models and Supabase connector at startup"""
    global supabase_connector, asr_handler, summarizer_handler
    
    try:
        print("Initializing Supabase connector...")
        supabase_connector = SupabaseConnector()
    except Exception as e:
        print(f"Warning: Could not initialize Supabase connector: {e}")
        print("Continuing with local model loading...")
        supabase_connector = None
    
    try:
        print("Loading ASR model...")
        asr_handler = ASRHandler(supabase_connector)
    except Exception as e:
        print(f"Error loading ASR model: {e}")
        asr_handler = None
        raise
    
    try:
        print("Loading Summarizer model...")
        summarizer_handler = SummarizerHandler(supabase_connector)
    except Exception as e:
        print(f"Error loading Summarizer model: {e}")
        summarizer_handler = None
        # Don't raise here, as we might still want to run with just ASR
    
    print("Startup completed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "asr_loaded": asr_handler is not None and asr_handler.model is not None,
        "summarizer_loaded": summarizer_handler is not None and summarizer_handler.model is not None
    }

def convert_webm_to_wav(webm_path, wav_path):
    """
    Convert webm audio file to wav format
    """
    try:
        # Try to read with soundfile first
        try:
            data, samplerate = sf.read(webm_path)
            print(f"Soundfile read: shape={data.shape}, samplerate={samplerate}")
            # Ensure mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            # Resample to 16kHz if needed
            if samplerate != 16000:
                data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                samplerate = 16000
            sf.write(wav_path, data, samplerate)
            print(f"Converted to WAV: {wav_path}")
            return True
        except Exception as e:
            print(f"Soundfile conversion failed: {e}")
            # Fallback to librosa
            try:
                data, samplerate = librosa.load(webm_path, sr=16000)  # Force 16kHz
                print(f"Librosa read: shape={data.shape}, samplerate={samplerate}")
                # Librosa already returns mono and resampled to 16kHz
                sf.write(wav_path, data, samplerate)
                print(f"Converted to WAV: {wav_path}")
                return True
            except Exception as e2:
                print(f"Librosa conversion failed: {e2}")
                return False
    except Exception as e:
        print(f"Error converting webm to wav: {e}")
        return False

@app.post("/voice-to-notes", response_model=VoiceToNotesResponse)
async def voice_to_notes(file: UploadFile = File(...)):
    """Process audio file and generate transcript + summary"""
    global asr_handler, summarizer_handler
    
    if asr_handler is None:
        raise HTTPException(status_code=500, detail="ASR model not loaded")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        # Write file content
        content = await file.read()
        tmp_file.write(content)
    
    try:
        # Convert WebM to WAV
        wav_path = tmp_path + ".wav"
        if not convert_webm_to_wav(tmp_path, wav_path):
            raise HTTPException(status_code=500, detail="Failed to convert audio file")
        
        # Transcribe audio
        transcript = asr_handler.transcribe(wav_path)
        
        # Generate summary if summarizer is available
        summary = "Summary generation not available"
        if summarizer_handler is not None:
            try:
                summary = summarizer_handler.summarize(transcript)
            except Exception as e:
                print(f"Error generating summary: {e}")
                summary = "Error generating summary"
        
        return VoiceToNotesResponse(transcript=transcript, summary=summary)
    
    except Exception as e:
        print(f"Error processing voice note: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for streaming transcription"""
    global asr_handler
    
    try:
        await websocket.accept()
        active_connections.append(websocket)
        print("WebSocket connection accepted")
    except Exception as e:
        print(f"Failed to accept WebSocket connection: {e}")
        return
    
    # Store accumulated audio data
    accumulated_audio = bytearray()
    
    try:
        while True:
            try:
                # Receive message with timeout to prevent blocking
                data = await websocket.receive()
                
                # Check if it's bytes data
                if "bytes" in data:
                    audio_bytes = data["bytes"]
                    accumulated_audio.extend(audio_bytes)
                    print(f"Received {len(audio_bytes)} bytes, total: {len(accumulated_audio)} bytes")
                    
                    # Process every 16KB of audio data (approx 0.5-1 seconds) for more responsive live transcription
                    if len(accumulated_audio) >= 16384:
                        print(f"Processing audio chunk of size: {len(accumulated_audio)} bytes")
                        try:
                            # Save accumulated bytes to temporary file as WebM
                            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
                                tmp_file.write(accumulated_audio)
                                tmp_path = tmp_file.name
                            
                            print(f"Saved {len(accumulated_audio)} bytes to {tmp_path}")
                            
                            # Validate file size
                            if os.path.getsize(tmp_path) < 1000:  # Less than 1KB, likely empty
                                print("Audio file too small, skipping transcription")
                                accumulated_audio = bytearray()
                                continue
                            
                            # Convert WebM to WAV
                            wav_path = tmp_path + ".wav"
                            if convert_webm_to_wav(tmp_path, wav_path):
                                print(f"Converted to WAV: {wav_path}")
                                
                                # Verify WAV file integrity
                                try:
                                    info = sf.info(wav_path)
                                    print(f"WAV info: {info.samplerate}Hz, {info.channels} channels, {info.duration}s")
                                    
                                    # Check if audio has content (not silence)
                                    data, sr = sf.read(wav_path)
                                    audio_energy = np.mean(np.abs(data))
                                    print(f"Audio energy: {audio_energy}")
                                    
                                    # Skip transcription if audio is too quiet (likely silence)
                                    if audio_energy < 0.001:
                                        print("Audio too quiet, likely silence. Skipping transcription.")
                                        response = {
                                            "partial_transcript": "...",  # Show processing indicator
                                            "is_final": False
                                        }
                                        await websocket.send_text(json.dumps(response, ensure_ascii=False))
                                    else:
                                        # Transcribe audio
                                        transcript = asr_handler.transcribe(wav_path)
                                        print(f"Generated transcript: '{transcript}'")
                                        
                                        # Only send non-empty transcripts
                                        if transcript and len(transcript.strip()) > 0:
                                            response = {
                                                "partial_transcript": transcript,
                                                "is_final": False
                                            }
                                            await websocket.send_text(json.dumps(response, ensure_ascii=False))
                                        else:
                                            # Send placeholder for empty transcripts
                                            response = {
                                                "partial_transcript": "...",
                                                "is_final": False
                                            }
                                            await websocket.send_text(json.dumps(response, ensure_ascii=False))
                                
                                except Exception as e:
                                    print(f"Error processing WAV file: {e}")
                                    error_response = {
                                        "partial_transcript": "Audio processing error",
                                        "is_final": False,
                                        "error": str(e)
                                    }
                                    await websocket.send_text(json.dumps(error_response))
                            
                            # Clean up temporary files
                            os.remove(tmp_path)
                            if os.path.exists(wav_path):
                                os.remove(wav_path)
                                
                            # Reset accumulated audio
                            accumulated_audio = bytearray()
                            
                        except Exception as e:
                            print(f"Error during streaming transcription: {e}")
                            error_response = {
                                "partial_transcript": "Streaming error",
                                "is_final": False,
                                "error": str(e)
                            }
                            await websocket.send_text(json.dumps(error_response))
                elif "text" in data:
                    # Handle text messages
                    try:
                        message = json.loads(data["text"])
                        if message.get("type") == "reset":
                            # Reset accumulation
                            accumulated_audio = bytearray()
                            print("Reset accumulated audio")
                    except Exception as e:
                        print(f"Error handling text message: {e}")
                        
            except Exception as e:
                # Handle receive errors gracefully
                print(f"WebSocket receive error: {e}")
                break
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        # Don't explicitly close the websocket as FastAPI handles this

if __name__ == "__main__":
    import uvicorn
    import os
    # Use RENDER_PORT or PORT environment variable if available, otherwise default to 8000
    port = int(os.environ.get("RENDER_PORT", os.environ.get("PORT", 8000)))
    uvicorn.run(app, host="0.0.0.0", port=port)
