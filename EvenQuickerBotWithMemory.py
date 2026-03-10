import os
import sys
import time
import wave
import subprocess
import numpy as np
import sounddevice as sd
import ollama
import whisper

# =========================================================================
# 1. CONFIGURATION
# =========================================================================

LLM_MODEL = "Gemma3:1b"
STT_MODEL_SIZE = "tiny" 

# Audio / TTS Settings
PIPER_BINARY = "/home/jwzy/.local/bin/piper"
PIPER_MODEL = "/home/jwzy/piper_voice/en_GB-alba-medium.onnx"
USB_AUDIO_DEVICE = "plughw:3,0"
SAMPLE_RATE = 16000
INPUT_DEVICE_NAME = None 

# Memory Settings
MAX_HISTORY = 6  # Only remember the last 3 exchanges (System + 3 User + 3 Bot)

# =========================================================================
# 2. INITIALIZATION
# =========================================================================

print(f"--- LOADING MODELS ---")
stt_model = whisper.load_model(STT_MODEL_SIZE)

# Initialize History with System Prompt
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant. Keep answers concise (1-2 sentence)."}
]

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
        
        # --- FILTER: Ignore common Whisper hallucinations ---
        if not text or len(text) < 2: return ""
        if text.lower() in ["thank you.", "thanks for watching.", "subtitles by", "you"]:
            return ""
            
        print(f"[USER] {text}")
        return text
    except: return ""

def speak(text):
    if not text.strip(): return
    # Clean text slightly for TTS (remove asterisks)
    clean_text = text.replace("*", "").replace("#", "")
    command = (
        f'echo "{clean_text}" | '
        f'{PIPER_BINARY} --model {PIPER_MODEL} --output-raw | '
        f'aplay -D {USB_AUDIO_DEVICE} -r 22050 -f S16_LE -t raw -q'
    )
    subprocess.run(command, shell=True)

def generate_and_speak(text):
    global conversation_history

    # --- RESET COMMAND ---
    if "reset" in text.lower() or "clear memory" in text.lower():
        conversation_history = [conversation_history[0]] # Keep only system prompt
        speak("Memory cleared.")
        print("[SYSTEM] Memory Cleared")
        return

    # 1. Add User Input
    conversation_history.append({"role": "user", "content": text})

    # 2. SLIDING WINDOW (Keep System Prompt + Last N Messages)
    # This ensures the 1B model doesn't get confused by old, irrelevant context
    if len(conversation_history) > MAX_HISTORY:
        # Keep index 0 (System), discard oldest, keep newest
        # Example: [System, User1, Bot1, User2, Bot2] -> [System, User2, Bot2]
        conversation_history = [conversation_history[0]] + conversation_history[-(MAX_HISTORY-1):]

    print("[BOT] ", end="", flush=True)
    
    response = ollama.chat(model=LLM_MODEL, messages=conversation_history, stream=False)
    full_response = response['message']['content']
    print(full_response)
    
    conversation_history.append({"role": "assistant", "content": full_response})
    speak(full_response)

# =========================================================================
# 4. MAIN LOOP
# =========================================================================

def main():
    print("--- SYSTEM READY ---")
    speak("Ready.")

    while True:
        try:
            audio_file = record_voice_adaptive()
            if not audio_file: continue 
                
            user_text = transcribe_audio(audio_file)
            if not user_text: continue # Skip if empty or hallucinated
            
            speak("Thinking...")
            generate_and_speak(user_text)

        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye.")
            break
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()