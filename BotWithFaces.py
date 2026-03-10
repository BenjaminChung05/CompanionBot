import os
import sys
import time
import wave
import subprocess
import threading
import random
import re
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Audio / AI Libraries
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
MAX_HISTORY = 6 

# GUI Settings
BG_WIDTH, BG_HEIGHT = 800, 480  # Adjust for your screen

# =========================================================================
# 2. GUI & STATE MANAGEMENT
# =========================================================================

class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WARMUP = "warmup"

class AssistantGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Assistant")
        self.master.geometry(f"{BG_WIDTH}x{BG_HEIGHT}")
        self.master.configure(bg="black")
        
        # Make fullscreen (optional, press Esc to exit)
        self.master.attributes('-fullscreen', True)
        self.master.bind('<Escape>', self.exit_fullscreen)

        # State Variables
        self.current_state = BotStates.WARMUP
        self.animations = {}
        self.current_frame_index = 0
        
        # Audio & Logic Variables
        self.stt_model = None
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful assistant. Keep answers concise (1-2 sentence)."}
        ]
        
        # Setup Label for Face
        self.face_label = tk.Label(master, bg="black")
        self.face_label.pack(expand=True, fill="both")

        # Load Images & Start Animation Loop
        self.load_animations()
        self.update_animation()

        # Start the "Brain" in a separate thread so GUI doesn't freeze
        self.worker_thread = threading.Thread(target=self.run_system_logic, daemon=True)
        self.worker_thread.start()

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)

    def set_state(self, state):
        if self.current_state != state:
            self.current_state = state
            self.current_frame_index = 0
            # If switching to IDLE, ensure we don't get stuck in a weird frame
            print(f"[STATE] {state.upper()}")

    def load_animations(self):
        base_path = "/home/jwzy/faces"
        states = [BotStates.IDLE, BotStates.LISTENING, BotStates.THINKING, BotStates.SPEAKING, BotStates.WARMUP]
        
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    try:
                        img = Image.open(os.path.join(folder, f))
                        img = img.resize((BG_WIDTH, BG_HEIGHT), Image.Resampling.LANCZOS)
                        self.animations[state].append(ImageTk.PhotoImage(img))
                    except Exception as e:
                        print(f"Error loading {f}: {e}")

            # Fallback if folder empty
            if not self.animations[state]:
                # If IDLE is missing, create a black square
                if state == BotStates.IDLE and not self.animations.get(BotStates.IDLE):
                    img = Image.new('RGB', (BG_WIDTH, BG_HEIGHT), color='black')
                    self.animations[state].append(ImageTk.PhotoImage(img))
                # Fallback to IDLE for others
                elif self.animations.get(BotStates.IDLE):
                    self.animations[state] = self.animations[BotStates.IDLE]

    def update_animation(self):
        frames = self.animations.get(self.current_state, [])
        if not frames:
            frames = self.animations.get(BotStates.IDLE, [])

        if frames:
            # Random frame for speaking to simulate lip sync, sequential for others
            if self.current_state == BotStates.SPEAKING and len(frames) > 1:
                self.current_frame_index = random.randint(0, len(frames) - 1)
            else:
                self.current_frame_index = (self.current_frame_index + 1) % len(frames)
            
            self.face_label.config(image=frames[self.current_frame_index])

        # Animation Speed
        delay = 150 if self.current_state == BotStates.SPEAKING else 800 
        self.master.after(delay, self.update_animation)

    # =========================================================================
    # 3. CORE LOGIC (Threaded)
    # =========================================================================

    def run_system_logic(self):
        print(f"--- LOADING MODELS ---")
        self.stt_model = whisper.load_model(STT_MODEL_SIZE)

        print(f"[INIT] Warming up Ollama ({LLM_MODEL})...")
        try:
            ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': 'hi'}])
            print("[INIT] Ready.")
        except Exception as e:
            print(f"[ERROR] Ollama connection failed: {e}")
            return

        self.set_state(BotStates.WARMUP)
        self.speak("Ready.")

        while True:
            try:
                # 1. Listen
                # We stay in IDLE until recording actually triggers, or you can force LISTENING here
                # Using LISTENING state for the whole wait period might look better:
                self.set_state(BotStates.LISTENING)
                
                audio_file = self.record_voice_adaptive()
                if not audio_file: 
                    self.set_state(BotStates.IDLE)
                    continue 
                
                # 2. Transcribe
                self.set_state(BotStates.THINKING)
                user_text = self.transcribe_audio(audio_file)
                
                if not user_text: 
                    self.set_state(BotStates.IDLE)
                    continue

                # 3. Think & Speak
                self.speak("Thinking...")
                self.generate_and_speak(user_text)
                

            except Exception as e:
                print(f"[CRITICAL ERROR] {e}")
                self.set_state(BotStates.IDLE)
                time.sleep(1)

    def record_voice_adaptive(self, filename="/home/jwzy/input.wav"):
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

    def transcribe_audio(self, filename):
        try:
            result = self.stt_model.transcribe(filename, fp16=False)
            text = result["text"].strip()
            
            # --- FILTER ---
            if not text or len(text) < 2: return ""
            if text.lower() in ["thank you.", "thanks for watching.", "subtitles by", "you"]:
                return ""
                
            print(f"[USER] {text}")
            return text
        except: return ""

    def speak(self, text):
        if not text.strip(): return
        
        # Change state to speaking
        prev_state = self.current_state
        self.set_state(BotStates.SPEAKING)
        
        clean_text = text.replace("*", "").replace("#", "")
        
        # Using the exact command from your original code
        command = (
            f'echo "{clean_text}" | '
            f'{PIPER_BINARY} --model {PIPER_MODEL} --output-raw | '
            f'aplay -D {USB_AUDIO_DEVICE} -r 22050 -f S16_LE -t raw -q'
        )
        subprocess.run(command, shell=True)
        
        # Return to previous state (usually THINKING or IDLE)
        # If we just spoke the final answer, the main loop will set IDLE next.
        # If we spoke "Thinking...", we want to go back to THINKING.
        if "thinking" in text.lower():
            self.set_state(BotStates.THINKING)
        else:
            self.set_state(BotStates.IDLE)

    def generate_and_speak(self, text):
        # --- RESET COMMAND ---
        if "reset" in text.lower() or "clear memory" in text.lower():
            self.conversation_history = [self.conversation_history[0]] 
            self.speak("Memory cleared.")
            print("[SYSTEM] Memory Cleared")
            return

        # 1. Add User Input
        self.conversation_history.append({"role": "user", "content": text})

        # 2. SLIDING WINDOW
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(MAX_HISTORY-1):]

        print("[BOT] ", end="", flush=True)
        
        # Ensure state is thinking while LLM generates
        self.set_state(BotStates.THINKING)
        
        response = ollama.chat(model=LLM_MODEL, messages=self.conversation_history, stream=False)
        full_response = response['message']['content']
        print(full_response)
        
        self.conversation_history.append({"role": "assistant", "content": full_response})
        self.speak(full_response)

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---")
    root = tk.Tk()
    app = AssistantGUI(root)
    root.mainloop()