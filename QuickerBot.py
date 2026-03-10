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
LLM_MODEL = "llama3.2:3b" 
#LLM_MODEL = "Gemma3:4b" 
STT_MODEL_SIZE = "tiny" 

# Paths
PIPER_BINARY = "/home/jwzy/.local/bin/piper"
PIPER_MODEL = "/home/jwzy/piper_voice/en_GB-alba-medium.onnx"

# Audio Settings
SAMPLE_RATE = 16000
INPUT_DEVICE_NAME = None 
USB_AUDIO_DEVICE = "plughw:2,0" # Your USB Speaker ID

# =========================================================================
# 2. INITIALIZATION
# =========================================================================

print(f"--- LOADING MODELS ---")
stt_model = whisper.load_model(STT_MODEL_SIZE)

print(f"[INIT] Warming up Ollama...")
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

    # --- SPEED TWEAK: Reduced Silence Duration ---
    silence_threshold = 0.03 
    silence_duration = 0.8   
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
    # print("[THINKING] Transcribing...", flush=True) # Commented out to reduce clutter
    try:
        result = stt_model.transcribe(filename, fp16=False)
        text = result["text"].strip()
        print(f"[USER] {text}")
        return text
    except: return ""

def speak_chunk(text):
    """
    Speaks a single sentence immediately.
    """
    if not text.strip(): return
    # print(f"[SPEAKING] {text}", flush=True) 

    # We run this command and WAIT for it to finish so sentences don't overlap
    command = (
        f'echo "{text}" | '
        f'{PIPER_BINARY} --model {PIPER_MODEL} --output-raw | '
        f'aplay -D {USB_AUDIO_DEVICE} -r 22050 -f S16_LE -t raw -q'
    )
    subprocess.run(command, shell=True)

def chat_stream_and_speak(text):
    """
    Generates text from LLM and speaks it sentence-by-sentence
    to reduce perceived latency.
    """
    print("[BOT] ", end="", flush=True)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers concise (1-3 sentences)."},
        {"role": "user", "content": text}
    ]
    
    # Stream the response from Ollama
    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
    
    buffer = ""
    # Regex to detect sentence endings (. ? ! followed by space or newline)
    sentence_endings = re.compile(r'(?<=[.?!])\s+')

    for chunk in stream:
        content = chunk['message']['content']
        print(content, end="", flush=True)
        buffer += content
        
        # Check if we have a full sentence in the buffer
        parts = sentence_endings.split(buffer)
        
        if len(parts) > 1:
            # We have at least one complete sentence
            sentence_to_speak = parts[0]
            
            # Speak it immediately!
            speak_chunk(sentence_to_speak)
            
            # Keep the rest in the buffer
            buffer = parts[1]

    # Speak any remaining text in the buffer
    if buffer.strip():
        speak_chunk(buffer)
    
    print("\n")

# =========================================================================
# 4. MAIN LOOP
# =========================================================================

def main():
    print("--- SYSTEM READY ---")
    speak_chunk("Ready.")

    while True:
        try:
            # 1. Listen
            audio_file = record_voice_adaptive()
            if not audio_file: continue 
                
            # 2. Transcribe
            user_text = transcribe_audio(audio_file)
            if not user_text: continue

            # 3. Think AND Speak (Streaming)
            speak_chunk("Processing...")
            chat_stream_and_speak(user_text)

        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye.")
            break
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()