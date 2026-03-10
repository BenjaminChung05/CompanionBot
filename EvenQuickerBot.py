import os
import sys
import time
import wave
import subprocess
import numpy as np
import sounddevice as sd
import scipy.signal
import ollama
import whisper
import threading
import re

# =========================================================================
# 1. CONFIGURATION
# =========================================================================

# AI Models
LLM_MODEL = "Gemma3:1b" 
STT_MODEL_SIZE = "tiny" 

# Paths
PIPER_BINARY = "/home/jwzy/.local/bin/piper"
PIPER_MODEL = "/home/jwzy/piper_voice/en_GB-alba-medium.onnx"

# Audio Settings
SAMPLE_RATE = 16000
INPUT_DEVICE_NAME = None 
USB_AUDIO_DEVICE = "plughw:3,0" # Your USB Speaker ID

# =========================================================================
# 2. INITIALIZATION
# =========================================================================

print(f"--- LOADING MODELS ---")
stt_model = whisper.load_model(STT_MODEL_SIZE)

print(f"[INIT] Warming up Ollama ({LLM_MODEL})...")
try:
    ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': 'hi'}])
    print("[INIT] Ready.")
except Exception as e:
    print(f"[ERROR] Ollama connection failed: {e}")
    sys.exit(1)

# =========================================================================
# 3. CORE FUNCTIONS
# =========================================================================

def record_voice_adaptive(filename="input.wav"):
    print("\n[LISTENING] ...", flush=True)
    
    try:
        device_info = sd.query_devices(kind='input')
        native_samplerate = int(device_info['default_samplerate'])
    except: native_samplerate = 48000 

    silence_threshold = 0.03 
    silence_duration = 1.0   
    max_record_time = 30.0
    chunk_duration = 0.05    
    chunk_size = int(native_samplerate * chunk_duration)
    
    buffer = []
    silent_chunks = 0
    num_silent_chunks = int(silence_duration / chunk_duration)
    max_chunks = int(max_record_time / chunk_duration)
    
    recording_started = False
    
    with sd.InputStream(samplerate=native_samplerate, channels=1, 
                        device=INPUT_DEVICE_NAME, blocksize=chunk_size) as stream:
        
        for _ in range(max_chunks):
            indata, overflowed = stream.read(chunk_size)
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            
            if volume_norm > silence_threshold:
                recording_started = True
                silent_chunks = 0
            elif recording_started:
                silent_chunks += 1
            
            if recording_started:
                buffer.append(indata.copy())
            
            if recording_started and silent_chunks >= num_silent_chunks:
                break
                
            if not recording_started and _ > (5.0 / chunk_duration):
                 return None

    if not buffer: return None

    audio_data = np.concatenate(buffer, axis=0).flatten()
    max_val = np.max(np.abs(audio_data))
    if max_val > 0: audio_data = audio_data / max_val
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(native_samplerate)
        wf.writeframes(audio_data.tobytes())
        
    return filename

def transcribe_audio(filename):
    try:
        result = stt_model.transcribe(filename, fp16=False)
        text = result["text"].strip()
        print(f"[USER] {text}")
        return text
    except: return ""

def speak(text):
    """
    Speaks the text immediately.
    """
    if not text.strip(): return

    # We run this command and WAIT for it to finish
    command = (
        f'echo "{text}" | '
        f'{PIPER_BINARY} --model {PIPER_MODEL} --output-raw | '
        f'aplay -D {USB_AUDIO_DEVICE} -r 22050 -f S16_LE -t raw -q'
    )
    subprocess.run(command, shell=True)

def generate_and_speak(text):
    """
    Generates text from LLM and speaks it.
    """
    print("[BOT] ", end="", flush=True)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers concise (1-3 sentences)."},
        {"role": "user", "content": text}
    ]
    
    # We use stream=False here to get the whole response at once
    # This avoids the complex buffering logic.
    response = ollama.chat(model=LLM_MODEL, messages=messages, stream=False)
    
    full_response = response['message']['content']
    print(full_response)
    
    # Speak the full response
    speak(full_response)

# =========================================================================
# 4. MAIN LOOP
# =========================================================================

def main():
    print("--- SYSTEM READY ---")
    speak("Ready.")

    while True:
        try:
            # 1. Listen
            audio_file = record_voice_adaptive()
            if not audio_file: continue 
                
            # 2. Transcribe
            user_text = transcribe_audio(audio_file)
            if not user_text: continue

            # 3. Think AND Speak (Simple)
            speak("Thinking...") # Optional: Un-comment if you want a processing acknowledgment
            generate_and_speak(user_text)

        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye.")
            break
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()