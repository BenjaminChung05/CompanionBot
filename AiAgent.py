import os
import sys
import warnings

# --- 1. WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
os.environ["ONNXRUNTIME_EXTERNAL_WARNINGS"] = "0"
os.environ["ORT_LOGGING_LEVEL"] = "3" 

# --- 2. STANDARD IMPORTS ---
import time
import datetime
import wave
import subprocess
import threading
import random
import re
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# --- 3. AUDIO & AI LIBRARIES ---
import numpy as np
import sounddevice as sd
import ollama
import whisper
from ddgs import DDGS 
import inflect  

# =========================================================================
# 4. CONFIGURATION
# =========================================================================

LLM_MODEL = "gemma3:1b"
VISION_MODEL = "moondream"
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
BG_WIDTH, BG_HEIGHT = 800, 480  

# =========================================================================
# 5. GUI & STATE MANAGEMENT
# =========================================================================

class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WARMUP = "warmup"
    CAPTURING = "capturing"

class AssistantGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Assistant")
        self.master.geometry(f"{BG_WIDTH}x{BG_HEIGHT}")
        self.master.configure(bg="black")
        
        self.master.attributes('-fullscreen', True)
        self.master.bind('<Escape>', self.exit_fullscreen)

        self.current_state = BotStates.WARMUP
        self.animations = {}
        self.current_frame_index = 0
        
        self.stt_model = None
        self.conversation_history = []
        
        # --- NEW: Image toggle state to prevent Ollama caching
        self.capture_toggle = False 
        
        self.inflect_engine = inflect.engine()
        
        self.face_label = tk.Label(master, bg="black")
        self.face_label.pack(expand=True, fill="both")

        # Overlay for displaying captured images
        self.overlay_label = tk.Label(master, bg="black", bd=2, relief="solid")
        self.current_overlay_image = None

        self.load_animations()
        self.update_animation()

        self.worker_thread = threading.Thread(target=self.run_system_logic, daemon=True)
        self.worker_thread.start()

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)

    def set_state(self, state):
        if self.current_state != state:
            self.current_state = state
            self.current_frame_index = 0
            print(f"[STATE] {state.upper()}")

    def load_animations(self):
        base_path = "/home/jwzy/faces"
        states = [BotStates.IDLE, BotStates.LISTENING, BotStates.THINKING, BotStates.SPEAKING, BotStates.WARMUP, BotStates.CAPTURING]
        
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

            if not self.animations[state]:
                if state == BotStates.IDLE and not self.animations.get(BotStates.IDLE):
                    img = Image.new('RGB', (BG_WIDTH, BG_HEIGHT), color='black')
                    self.animations[state].append(ImageTk.PhotoImage(img))
                elif self.animations.get(BotStates.IDLE):
                    self.animations[state] = self.animations[BotStates.IDLE]

    def update_animation(self):
        frames = self.animations.get(self.current_state, [])
        if not frames:
            frames = self.animations.get(BotStates.IDLE, [])

        if frames:
            if self.current_state == BotStates.SPEAKING and len(frames) > 1:
                self.current_frame_index = random.randint(0, len(frames) - 1)
            else:
                self.current_frame_index = (self.current_frame_index + 1) % len(frames)
            
            self.face_label.config(image=frames[self.current_frame_index])

        delay = 150 if self.current_state == BotStates.SPEAKING else 800 
        self.master.after(delay, self.update_animation)

    # =========================================================================
    # 6. CAMERA & OVERLAY LOGIC
    # =========================================================================

    def show_camera_image(self, image_path):
        def _show():
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    img = img.resize((400, 300), Image.Resampling.LANCZOS)
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                except Exception as e:
                    print(f"[GUI ERROR] Could not load image: {e}")
        self.master.after(0, _show)

    def hide_camera_image(self):
        def _hide():
            self.overlay_label.place_forget()
        self.master.after(0, _hide)

    def take_picture(self):
        print("[CAMERA] Taking picture...", flush=True)
        self.set_state(BotStates.CAPTURING)
        time.sleep(0.5) 
        
        # --- FIX: Toggle between two file names to force Ollama to clear its cache
        self.capture_toggle = not self.capture_toggle
        img_name = "capture_A.jpg" if self.capture_toggle else "capture_B.jpg"
        capture_path = f"/home/jwzy/{img_name}"
        
        # Ensure old file is deleted so we know we have a fresh frame
        if os.path.exists(capture_path):
            os.remove(capture_path)

        try:
            subprocess.run([
                "rpicam-jpeg", 
                "-t", "200", 
                "--width", "1640", 
                "--height", "1232", 
                "-n", 
                "-o", capture_path
            ], check=True)
            return capture_path
        except Exception as e:
            print(f"[CAMERA ERROR] {e}")
            return None

    # =========================================================================
    # 7. CORE LOGIC (Threaded)
    # =========================================================================

    def run_system_logic(self):
        print(f"--- LOADING MODELS ---")
        self.stt_model = whisper.load_model(STT_MODEL_SIZE)

        print(f"[INIT] Warming up Ollama ({LLM_MODEL})...")
        try:
            ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': 'hi'}])
            print(f"[INIT] Warming up Vision ({VISION_MODEL})...")
            ollama.chat(model=VISION_MODEL, messages=[{'role': 'user', 'content': 'hi'}])
            print("[INIT] Ready.")
        except Exception as e:
            print(f"[ERROR] Ollama connection failed: {e}")
            return

        self.set_state(BotStates.WARMUP)
        self.speak("Ready.")

        while True:
            try:
                self.set_state(BotStates.LISTENING)
                
                audio_file = self.record_voice_adaptive()
                if not audio_file: 
                    self.set_state(BotStates.IDLE)
                    continue 
                
                self.set_state(BotStates.THINKING)
                user_text = self.transcribe_audio(audio_file)
                
                if not user_text: 
                    self.set_state(BotStates.IDLE)
                    continue

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
            
            if not text or len(text) < 2: return ""
            if text.lower() in ["thank you.", "thanks for watching.", "subtitles by", "you"]:
                return ""
                
            print(f"[USER] {text}")
            return text
        except: return ""

    def speak(self, text):
        if not text.strip(): return
        
        self.set_state(BotStates.SPEAKING)
        clean_text = text.replace("*", "").replace("#", "")
        
        def _replace_number(match):
            num_str = match.group(0)
            try:
                clean_num = num_str.replace(',', '')
                words = self.inflect_engine.number_to_words(clean_num, andword='')
                return words
            except:
                return num_str
                
        clean_text = re.sub(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', _replace_number, clean_text)
        
        command = (
            f'echo "{clean_text}" | '
            f'{PIPER_BINARY} --model {PIPER_MODEL} --output-raw | '
            f'aplay -D {USB_AUDIO_DEVICE} -r 22050 -f S16_LE -t raw -q'
        )
        subprocess.run(command, shell=True)
        
        if "thinking" in text.lower():
            self.set_state(BotStates.THINKING)
        else:
            self.set_state(BotStates.IDLE)

    # =========================================================================
    # 8. CHAT & SEARCH LOGIC
    # =========================================================================

    def perform_web_search(self, query):
        print(f"[SEARCH] Looking up: {query}...", flush=True)
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region='wt-wt', max_results=1))
                if results:
                    r = results[0]
                    return f"Title: {r.get('title', 'No title')}\nSnippet: {r.get('body', 'No snippet')}"
                return "No results found on the web."
        except Exception as e:
            print(f"[SEARCH ERROR] {e}")
            return "I could not connect to the internet to search."

    def generate_and_speak(self, text):
        if "reset" in text.lower() or "clear memory" in text.lower():
            self.conversation_history = [{"role": "system", "content": ""}] 
            self.speak("Memory cleared.")
            print("[SYSTEM] Memory Cleared")
            return

        current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        
        # --- FIX: Tighter instructions to force Gemma not to hallucinate vision from memory
        dynamic_system_prompt = f"""You are a helpful voice assistant.
The current date and time is: {current_time}.

Keep answers very concise (1-2 sentences).

CRITICAL RULES:
1. NORMAL CHAT: If the user says hello or asks a general question, reply normally. Do not use any prefixes.
2. WEB SEARCH: If you need real-time information (news, weather, prices), reply starting with exactly "SEARCH_WEB:" followed by 2-3 keywords.
3. CAMERA: If the user asks you to take a picture, look around, or see something, reply starting with exactly "CAPTURE_IMAGE:" followed by the question.

EXAMPLES:
User: Hello!
You: Hi there! How are you doing today?

User: What is the capital of Japan?
You: The capital of Japan is Tokyo.

User: Tell me the current date.
You: Today is {current_time}.

User: What is the price of gold?
You: SEARCH_WEB: current gold price USD

User: What is the weather like?
You: SEARCH_WEB: current weather

User: What do you see right now?
You: CAPTURE_IMAGE: Describe everything you see in this image.
"""
        if len(self.conversation_history) > 0 and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = dynamic_system_prompt
        else:
            self.conversation_history.insert(0, {"role": "system", "content": dynamic_system_prompt})

        self.conversation_history.append({"role": "user", "content": text})

        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(MAX_HISTORY-1):]

        self.set_state(BotStates.THINKING)
        
        response = ollama.chat(model=LLM_MODEL, messages=self.conversation_history, stream=False, options={"temperature": 0.2})
        full_response = response['message']['content'].strip()

        if "SEARCH_WEB:" in full_response:
            search_query = full_response.split("SEARCH_WEB:")[-1].strip()
            search_query = search_query.replace('"', '').replace('.', '') 
            
            search_result = self.perform_web_search(search_query)

            summary_prompt = self.conversation_history + [
                {"role": "system", "content": f"Summarize this search result for the user in 1-2 conversational sentences: {search_result}"}
            ]
            
            summary_response = ollama.chat(model=LLM_MODEL, messages=summary_prompt, stream=False, options={"temperature": 0.6})
            final_spoken_text = summary_response['message']['content']

            print(f"[BOT (From Search)] {final_spoken_text}")
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": f"SEARCH_WEB: {search_query}\n[Search Result: {final_spoken_text}]"
            })
            
            self.speak(final_spoken_text)
            return 

        if "CAPTURE_IMAGE:" in full_response:
            vision_prompt = full_response.split("CAPTURE_IMAGE:")[-1].strip()
            
            self.speak("Let me take a look.")
            
            # --- FIX: get the new dynamic path
            capture_path = self.take_picture()
            
            if capture_path:
                self.show_camera_image(capture_path)
                self.set_state(BotStates.THINKING)
                
                print(f"[VISION] Asking moondream: '{vision_prompt}' on file {capture_path}")
                try:
                    vision_msg = [{
                        "role": "user", 
                        "content": vision_prompt, 
                        "images": [capture_path] # Use the toggled path
                    }]
                    
                    vision_resp = ollama.chat(model=VISION_MODEL, messages=vision_msg, stream=False)
                    vision_text = vision_resp['message']['content'].strip()
                    
                    self.hide_camera_image()
                    
                    print(f"[BOT (Vision)] {vision_text}")
                    
                    # --- FIX: Structure the history so Gemma knows this is an isolated event that concluded
                    self.conversation_history.append({
                        "role": "assistant", 
                        "content": f"CAPTURE_IMAGE: {vision_prompt}\n[I looked through the camera and told the user: {vision_text}]"
                    })
                    
                    self.speak(vision_text)
                except Exception as e:
                    self.hide_camera_image()
                    self.speak("I took the picture, but my vision processing failed.")
                    print(f"[VISION ERROR] {e}")
            else:
                self.speak("I had trouble connecting to my camera.")
                
            return

        print(f"[BOT] {full_response}")
        self.conversation_history.append({"role": "assistant", "content": full_response})
        self.speak(full_response)

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---")
    root = tk.Tk()
    app = AssistantGUI(root)
    root.mainloop()