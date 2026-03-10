import os
import sys
import time
import wave
import subprocess
import numpy as np
import sounddevice as sd
import scipy.signal
import ollama
import whisper  # <--- CHANGED: Import standard whisper

# =========================================================================
# 1. CONFIGURATION
# =========================================================================

# AI Models
LLM_MODEL = "Gemma3:4b"
STT_MODEL_SIZE = "tiny" # Options: "tiny", "base", "small" (Small might lag on Pi CPU)

# Paths (Adjust these to match your setup)
PIPER_BINARY = "/home/jwzy/.local/bin/piper"
PIPER_MODEL = "/home/jwzy/piper_voice/en_GB-alba-medium.onnx"

# Audio Settings
SAMPLE_RATE = 16000
INPUT_DEVICE_NAME = None 

# =========================================================================
# 2. INITIALIZATION
# =========================================================================

print(f"--- LOADING MODELS ---")

# Initialize Standard Whisper
print(f"[INIT] Loading OpenAI Whisper ({STT_MODEL_SIZE})...")
# We load the model once into memory. It automatically selects CPU on Pi.
stt_model = whisper.load_model(STT_MODEL_SIZE)

# Check Ollama connection
print(f"[INIT] Connecting to Ollama ({LLM_MODEL})...")
try:
    ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': 'hi'}])
    print("[INIT] Ollama Connected.")
except Exception as e:
    print(f"[ERROR] Could not connect to Ollama: {e}")
    sys.exit(1)

# =========================================================================
# 3. CORE FUNCTIONS
# =========================================================================

def record_voice_adaptive(filename="input.wav"):
    """
    Records audio until silence is detected. 
    """
    print("\n[LISTENING] Speak now...", flush=True)
    
    try:
        device_info = sd.query_devices(kind='input')
        native_samplerate = int(device_info['default_samplerate'])
    except: 
        native_samplerate = 48000 

    # VAD Constants
    silence_threshold = 0.03 # Slightly higher threshold for noise
    silence_duration = 0.8   # Wait a bit longer before cutting off
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
            if overflowed:
                print("Warning: Audio buffer overflow")
            
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            
            if volume_norm > silence_threshold:
                recording_started = True
                silent_chunks = 0
            elif recording_started:
                silent_chunks += 1
            
            if recording_started:
                buffer.append(indata.copy())
            
            if recording_started and silent_chunks >= num_silent_chunks:
                print("[LISTENING] Silence detected, processing...")
                break
                
            if not recording_started and _ > (5.0 / chunk_duration):
                 return None

    if not buffer: return None

    # Save to file
    audio_data = np.concatenate(buffer, axis=0).flatten()
    # Normalize
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(native_samplerate)
        wf.writeframes(audio_data.tobytes())
        
    return filename

def transcribe_audio(filename):
    """
    Uses Standard Whisper to transcribe the audio file.
    """
    print("[THINKING] Transcribing...", flush=True)
    try:
        # Standard Whisper Transcribe Call
        # fp16=False is usually required on Pi CPU to avoid warnings/errors
        result = stt_model.transcribe(filename, fp16=False)
        
        text = result["text"].strip()
        print(f"[USER] {text}")
        return text
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""

def generate_llm_response(text):
    """
    Sends text to Gemma 3 via Ollama.
    """
    print("[THINKING] Gemma is thinking...", flush=True)
    
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep answers short and conversational."},
        {"role": "user", "content": text}
    ]
    
    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
    
    print("[BOT] ", end="", flush=True)
    
    full_text = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end="", flush=True)
        full_text += content
    
    print("\n")
    return full_text




def speak(text):
    """
    Pipes text to Piper and plays via aplay (Fixes USB Speaker issues).
    """
    if not text.strip(): return

    # --- CONFIGURATION FOR USB SPEAKER ---
    # Card 2, Device 0 (as confirmed by your 'pink noise' test)
    AUDIO_DEVICE = "plughw:2,0" 
    # -------------------------------------

    print(f"[SPEAKING] {text}", flush=True)

    # We construct a command line string:
    # echo "text" | piper ... | aplay -D plughw:2,0 ...
    command = (
        f'echo "{text}" | '
        f'{PIPER_BINARY} --model {PIPER_MODEL} --output-raw | '
        f'aplay -D {AUDIO_DEVICE} -r 22050 -f S16_LE -t raw -'
    )

    try:
        # shell=True is required to make the pipes (|) work
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Audio failed: {e}")
    except Exception as e:
        print(f"[ERROR] General failure: {e}")

# =========================================================================
# 4. MAIN LOOP
# =========================================================================

def main():
    print("--- SYSTEM READY ---")
    speak("I am ready.")

    while True:
        try:
            # 1. Listen
            audio_file = record_voice_adaptive()
            
            if not audio_file:
                continue 
                
            # 2. Transcribe
            user_text = transcribe_audio(audio_file)
            
            if not user_text:
                continue

            # 3. Think
            response_text = generate_llm_response(user_text)

            # 4. Speak
            speak(response_text)

        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye.")
            break
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()