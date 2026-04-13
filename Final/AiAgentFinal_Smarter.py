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
import subprocess
import threading
import random
import re
from difflib import SequenceMatcher
import tempfile
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests

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
STT_MODEL_SIZE = "base.en"

# Audio / TTS Settings
PIPER_BINARY = "/home/jwzy/.local/bin/piper"
PIPER_MODEL = "/home/jwzy/piper_voice/en_GB-alba-medium.onnx"
USB_AUDIO_DEVICE = "plughw:3,0"
SAMPLE_RATE = 16000
INPUT_DEVICE_NAME = None 
STT_LANGUAGE = "en"
STT_SILENCE_THRESHOLD_BASE = 0.015
STT_SILENCE_THRESHOLD_MAX = 0.10
STT_SILENCE_DURATION = 0.9
STT_WAIT_FOR_SPEECH_SEC = 4.5
STT_MAX_RECORD_SEC = 12.0
STT_CHUNK_SEC = 0.04
STT_PREROLL_SEC = 0.25
STT_MIN_AUDIO_SEC = 0.25
QUIET_LISTENING_MODE = False
NOISE_GATE_STRENGTH = 1.25
SPEECH_BAND_MIN_HZ = 80
SPEECH_BAND_MAX_HZ = 7000
NOISE_FFT_SIZE = 512
NOISE_FFT_HOP = 128

LLM_OPTIONS = {"temperature": 0.15, "num_predict": 48, "repeat_penalty": 1.1}
SUMMARY_OPTIONS = {"temperature": 0.25, "num_predict": 44}
GROUNDED_OPTIONS = {"temperature": 0.1, "num_predict": 64, "repeat_penalty": 1.15}
WAKE_PROMPT = "Go ahead, I'm listening."
WAKE_WORDS = (
    "hey robot",
    "okay robot",
    "ok robot",
    "hello robot",
    "robot",
    "hey robert",
    "a robot",
)
WEB_SEARCH_PREFIXES = ("search up", "look up")

# Memory Settings
MAX_HISTORY = 6 

# GUI Settings
BG_WIDTH, BG_HEIGHT = 800, 480  

# Robot API Settings
ROBOT_URL = "http://127.0.0.1:5000"

# Music Assistant API Settings
MUSIC_ASSISTANT_API = os.environ.get("MUSIC_ASSISTANT_API", "http://127.0.0.1:8095/api")
MUSIC_ASSISTANT_TOKEN = os.environ.get("MUSIC_ASSISTANT_TOKEN", "")
MUSIC_ASSISTANT_PLAYER_ID = os.environ.get("MUSIC_ASSISTANT_PLAYER_ID", "")
MUSIC_ASSISTANT_TIMEOUT = 5.0

MUSIC_SEARCH_MEDIA_TYPES = ["track", "album", "playlist", "artist", "radio"]

MUSIC_PLAY_PREFIXES = (
    "play music ",
    "play song ",
    "play track ",
    "play album ",
    "play playlist ",
    "play artist ",
    "play ",
)

MUSIC_PAUSE_WORDS = ("pause music", "pause song", "pause")
MUSIC_RESUME_WORDS = ("resume music", "resume song", "resume", "continue music", "continue song")
MUSIC_STOP_WORDS = ("stop music", "stop song")
MUSIC_NEXT_WORDS = ("next song", "skip song", "next track", "skip track", "skip")

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
        self.noise_profile_mag = None
        
        self.capture_toggle = False 
        
        self.inflect_engine = inflect.engine()
        
        self.face_label = tk.Label(master, bg="black")
        self.face_label.pack(expand=True, fill="both")

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

    def set_robot_pause(self, pause_state):
        try:
            requests.post(f'{ROBOT_URL}/ai_override', json={"pause": pause_state}, timeout=1.0)
            state_str = "PAUSED" if pause_state else "RESUMED"
            print(f"[ROBOT COMMS] Motors {state_str}")
        except Exception as e:
            pass # Suppress constant error printing if robot is still booting

    def warmup_tts(self):
        warmup_file = os.path.join(tempfile.gettempdir(), "piper_warmup.wav")
        try:
            subprocess.run(
                [PIPER_BINARY, "--model", PIPER_MODEL, "--output_file", warmup_file],
                input="warm up\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
        except Exception:
            pass
        finally:
            if os.path.exists(warmup_file):
                try:
                    os.remove(warmup_file)
                except Exception:
                    pass

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
        print("[CAMERA] Asking Robot for a picture...", flush=True)
        self.set_state(BotStates.CAPTURING)
        time.sleep(0.5) 
        
        self.capture_toggle = not self.capture_toggle
        img_name = "capture_A.jpg" if self.capture_toggle else "capture_B.jpg"
        capture_path = f"/home/jwzy/{img_name}"
        
        if os.path.exists(capture_path):
            os.remove(capture_path)

        try:
            response = requests.get(f'{ROBOT_URL}/snapshot', timeout=3.0)
            
            if response.status_code == 200:
                with open(capture_path, 'wb') as f:
                    f.write(response.content)
                return capture_path
            else:
                print(f"[CAMERA ERROR] Robot returned status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[CAMERA ERROR] Could not fetch image from robot. Is robot.py running? {e}")
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

        self.warmup_tts()
        self.set_state(BotStates.WARMUP)
        self.speak("Ready.")

        while True:
            try:
                # 1. Wait for wake word. Do not pause motors unless quiet listening is explicitly enabled.
                self.set_robot_pause(QUIET_LISTENING_MODE)
                self.set_state(BotStates.IDLE)
                print("\n[WAKE] Waiting for wake word...", flush=True)
                wake_text = self.wait_for_wake_word()
                if not wake_text:
                    if QUIET_LISTENING_MODE:
                        self.set_robot_pause(False)
                    continue

                # 2. Wake detected, pause movement, then listen for the query.
                self.set_robot_pause(True)
                try:
                    self.speak(WAKE_PROMPT)
                    time.sleep(0.15)
                    print(f"[WAKE] Heard wake phrase: {wake_text}", flush=True)
                    print("[WAKE] Activating microphone for user query.", flush=True)
                    self.set_state(BotStates.LISTENING)
                    
                    audio_buffer = self.record_voice_adaptive(wait_for_speech_sec=7.0, max_record_sec=12.0)
                    if audio_buffer is None:
                        continue # If they didn't say anything, loop back and keep moving.
                    
                    # 3. Audio captured. Process.
                    self.set_state(BotStates.THINKING)
                    user_text = self.transcribe_audio(audio_buffer)
                    
                    if not user_text: 
                        continue

                    self.generate_and_speak(user_text)
                finally:
                    # Always release AI pause after handling one wake interaction.
                    self.set_robot_pause(False)

            except Exception as e:
                print(f"[CRITICAL ERROR] {e}")
                self.set_robot_pause(False)
                self.set_state(BotStates.IDLE)
                time.sleep(1)

    def _resample_audio(self, audio, src_rate, dst_rate):
        if src_rate == dst_rate or audio.size == 0:
            return audio.astype(np.float32)
        duration = len(audio) / float(src_rate)
        target_len = max(1, int(duration * dst_rate))
        x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def _calibrate_noise_floor(self, samplerate, seconds=0.35):
        frames = max(1, int(samplerate * seconds))
        try:
            ambient = sd.rec(
                frames,
                samplerate=samplerate,
                channels=1,
                device=INPUT_DEVICE_NAME,
                dtype="float32",
            )
            sd.wait()
            ambient = ambient.flatten()
            if ambient.size == 0:
                return STT_SILENCE_THRESHOLD_BASE
            self._update_noise_profile(ambient, samplerate)
            rms = float(np.sqrt(np.mean(np.square(ambient))))
            return float(
                np.clip(
                    rms * 1.8,
                    STT_SILENCE_THRESHOLD_BASE,
                    STT_SILENCE_THRESHOLD_MAX,
                )
            )
        except Exception:
            return STT_SILENCE_THRESHOLD_BASE

    def _update_noise_profile(self, ambient, samplerate):
        if ambient.size < NOISE_FFT_SIZE:
            return
        win = np.hanning(NOISE_FFT_SIZE).astype(np.float32)
        mags = []
        for i in range(0, len(ambient) - NOISE_FFT_SIZE, NOISE_FFT_HOP):
            frame = ambient[i : i + NOISE_FFT_SIZE] * win
            mags.append(np.abs(np.fft.rfft(frame)))
        if not mags:
            return
        mag = np.median(np.vstack(mags), axis=0)
        # Keep only speech-relevant band in the profile.
        freqs = np.fft.rfftfreq(NOISE_FFT_SIZE, d=1.0 / samplerate)
        band = (freqs >= SPEECH_BAND_MIN_HZ) & (freqs <= SPEECH_BAND_MAX_HZ)
        self.noise_profile_mag = np.where(band, mag, 0.0).astype(np.float32)

    def _denoise_speech_audio(self, audio, samplerate):
        if audio.size < NOISE_FFT_SIZE:
            return audio

        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - (0.97 * audio[:-1])

        n_fft = NOISE_FFT_SIZE
        hop = NOISE_FFT_HOP
        win = np.hanning(n_fft).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / samplerate)
        band = (freqs >= SPEECH_BAND_MIN_HZ) & (freqs <= SPEECH_BAND_MAX_HZ)

        pad = (n_fft - (len(emphasized) - n_fft) % hop) % hop
        padded = np.pad(emphasized, (0, pad + n_fft), mode="constant")
        out = np.zeros_like(padded, dtype=np.float32)
        norm = np.zeros_like(padded, dtype=np.float32)

        if self.noise_profile_mag is None or len(self.noise_profile_mag) != (n_fft // 2 + 1):
            self.noise_profile_mag = np.zeros(n_fft // 2 + 1, dtype=np.float32)

        for i in range(0, len(padded) - n_fft, hop):
            frame = padded[i : i + n_fft] * win
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)

            clean_mag = np.maximum(mag - (NOISE_GATE_STRENGTH * self.noise_profile_mag), 0.06 * mag)
            clean_mag = np.where(band, clean_mag, 0.0)
            clean_spec = clean_mag * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_spec, n=n_fft).astype(np.float32) * win

            out[i : i + n_fft] += clean_frame
            norm[i : i + n_fft] += win

        norm[norm < 1e-6] = 1.0
        cleaned = out / norm
        cleaned = cleaned[: len(emphasized)]

        # De-emphasis to restore natural speech contour.
        deemph = np.empty_like(cleaned)
        deemph[0] = cleaned[0]
        for i in range(1, len(cleaned)):
            deemph[i] = cleaned[i] + (0.97 * deemph[i - 1])

        peak = float(np.max(np.abs(deemph))) if deemph.size else 0.0
        if peak > 1e-6:
            deemph = np.clip(deemph / peak, -1.0, 1.0)
        return deemph.astype(np.float32)

    def record_voice_adaptive(self, wait_for_speech_sec=STT_WAIT_FOR_SPEECH_SEC, max_record_sec=STT_MAX_RECORD_SEC):
        try:
            device_info = sd.query_devices(kind='input')
            native_samplerate = int(device_info['default_samplerate'])
        except Exception:
            native_samplerate = 48000

        silence_threshold = self._calibrate_noise_floor(native_samplerate)
        chunk_size = max(1, int(native_samplerate * STT_CHUNK_SEC))

        buffer = []
        silent_chunks = 0
        pre_roll = []
        pre_roll_chunks = max(1, int(STT_PREROLL_SEC / STT_CHUNK_SEC))
        num_silent_chunks = max(1, int(STT_SILENCE_DURATION / STT_CHUNK_SEC))
        max_chunks = max(1, int(max_record_sec / STT_CHUNK_SEC))
        wait_chunks = max(1, int(wait_for_speech_sec / STT_CHUNK_SEC))

        recording_started = False

        with sd.InputStream(
            samplerate=native_samplerate,
            channels=1,
            device=INPUT_DEVICE_NAME,
            blocksize=chunk_size,
            dtype="float32",
            latency="low",
        ) as stream:
            for idx in range(max_chunks):
                indata, _ = stream.read(chunk_size)
                mono = indata[:, 0].copy()
                rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                pre_roll.append(mono)
                if len(pre_roll) > pre_roll_chunks:
                    pre_roll.pop(0)

                if rms > silence_threshold:
                    if not recording_started:
                        recording_started = True
                        buffer.extend(pre_roll)
                    silent_chunks = 0
                    buffer.append(mono)
                elif recording_started:
                    buffer.append(mono)
                    silent_chunks += 1

                if recording_started and silent_chunks >= num_silent_chunks:
                    break
                if not recording_started and idx >= wait_chunks:
                    return None

        if not buffer:
            return None

        audio_data = np.concatenate(buffer, axis=0).astype(np.float32)
        audio_data = self._denoise_speech_audio(audio_data, native_samplerate)
        peak = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0
        if peak > 1.0:
            audio_data = audio_data / peak

        audio_16k = self._resample_audio(audio_data, native_samplerate, SAMPLE_RATE)
        if audio_16k.size < int(SAMPLE_RATE * STT_MIN_AUDIO_SEC):
            return None
        return audio_16k

    def _normalize_text(self, text):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()

    def _contains_wake_word(self, text):
        normalized = self._normalize_text(text)
        if any(w in normalized for w in WAKE_WORDS):
            return True
        for phrase in WAKE_WORDS:
            if SequenceMatcher(None, normalized, phrase).ratio() >= 0.82:
                return True
            if SequenceMatcher(None, normalized[: len(phrase)], phrase).ratio() >= 0.86:
                return True
        return False

    def wait_for_wake_word(self):
        while True:
            audio_buffer = self.record_voice_adaptive(wait_for_speech_sec=8.0, max_record_sec=4.5)
            if audio_buffer is None:
                return None
            heard_text = self.transcribe_audio(audio_buffer)
            if not heard_text:
                continue
            if self._contains_wake_word(heard_text):
                return heard_text

    def transcribe_audio(self, audio_data):
        try:
            result = self.stt_model.transcribe(
                audio_data,
                fp16=False,
                language=STT_LANGUAGE,
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.45,
                logprob_threshold=-1.2,
                compression_ratio_threshold=2.2,
                beam_size=3,
                best_of=3,
            )
            text = re.sub(r"\s+", " ", result["text"]).strip()
            
            if not text or len(text) < 2:
                return ""
            normalized = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
            if normalized in {"thank you", "thanks for watching", "subtitles by", "you", "okay", "so"}:
                return ""
                
            print(f"[USER] {text}")
            return text
        except Exception:
            return ""

    def speak(self, text):
        if not text.strip(): return

        self.set_state(BotStates.THINKING)
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

        tts_file = os.path.join(tempfile.gettempdir(), f"assistant_tts_{time.time_ns()}.wav")

        try:
            # Generate audio first, then mark SPEAKING exactly at playback start.
            synth = subprocess.run(
                [PIPER_BINARY, "--model", PIPER_MODEL, "--output_file", tts_file],
                input=clean_text + "\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if synth.returncode == 0 and os.path.exists(tts_file) and os.path.getsize(tts_file) > 44:
                self.set_state(BotStates.SPEAKING)
                subprocess.run(
                    ["aplay", "-D", USB_AUDIO_DEVICE, "-q", tts_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            if os.path.exists(tts_file):
                try:
                    os.remove(tts_file)
                except Exception:
                    pass
        
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
                results = list(ddgs.text(query, region='wt-wt', max_results=3))
                if results:
                    lines = []
                    for i, r in enumerate(results, start=1):
                        title = r.get("title", "No title")
                        body = r.get("body", "No snippet")
                        href = r.get("href", "No source")
                        lines.append(f"[{i}] {title}\n{body}\nSource: {href}")
                    return "\n\n".join(lines)
                return "No results found on the web."
        except Exception as e:
            print(f"[SEARCH ERROR] {e}")
            return "I could not connect to the internet to search."

    def _answer_local_time_date(self, text):
        t = text.lower()
        now = datetime.datetime.now()
        if any(k in t for k in ["time", "what time", "current time"]):
            return now.strftime("It's %I:%M %p.")
        if any(k in t for k in ["date", "day today", "today date", "what day"]):
            return now.strftime("Today is %A, %B %d, %Y.")
        return None

    def _extract_explicit_search_query(self, text):
        lowered = text.strip().lower()
        for prefix in WEB_SEARCH_PREFIXES:
            if lowered == prefix:
                return ""
            token = prefix + " "
            if lowered.startswith(token):
                return text[len(token):].strip(" ?.!,")
        return None

    def _respond_from_web_grounded(self, user_text):
        search_query = user_text.strip()
        search_result = self.perform_web_search(search_query)
        grounded_prompt = [
            {
                "role": "system",
                "content": (
                    "Answer in 1-2 concise sentences using ONLY the snippets provided. "
                    "If snippets are insufficient, say you are not sure."
                ),
            },
            {"role": "user", "content": f"Question: {user_text}\n\nSnippets:\n{search_result}"},
        ]
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=grounded_prompt,
            stream=False,
            options=GROUNDED_OPTIONS,
        )
        return resp["message"]["content"].strip()

    def _music_headers(self):
        headers = {"Content-Type": "application/json"}
        if MUSIC_ASSISTANT_TOKEN:
            headers["Authorization"] = f"Bearer {MUSIC_ASSISTANT_TOKEN}"
        return headers

    def _music_api_command(self, command, args=None):
        payload = {"message_id": str(time.time_ns()), "command": command}
        if args is not None:
            payload["args"] = args
        try:
            resp = requests.post(
                MUSIC_ASSISTANT_API,
                json=payload,
                headers=self._music_headers(),
                timeout=MUSIC_ASSISTANT_TIMEOUT,
            )
            text = resp.text.strip()
            try:
                data = resp.json()
            except Exception:
                data = None
            return resp.status_code, text, data
        except Exception as e:
            return 0, str(e), None

    def _music_try_commands(self, attempts):
        last = (0, "No command attempted.", None, None)
        for command, args in attempts:
            status_code, text, data = self._music_api_command(command, args)
            lowered = text.lower()
            if "authentication required" in lowered:
                return False, "Music Assistant token is missing or invalid.", (status_code, text, data, command)
            if "invalid command" in lowered:
                last = (status_code, text, data, command)
                continue
            if status_code >= 400:
                last = (status_code, text, data, command)
                continue
            return True, "", (status_code, text, data, command)
        return False, "", last

    def _extract_music_query(self, user_text):
        lowered = user_text.strip().lower()
        for prefix in MUSIC_PLAY_PREFIXES:
            if lowered == prefix.strip():
                return ""
            if lowered.startswith(prefix):
                return user_text[len(prefix):].strip(" ?.!,")
        return None

    def _extract_first_media_uri(self, data):
        if not isinstance(data, dict):
            return None
        result = data.get("result")
        items = result if isinstance(result, list) else []
        if not items:
            return None
        first = items[0] if isinstance(items[0], dict) else None
        if not first:
            return None
        if first.get("uri"):
            return first["uri"]
        media_type = first.get("media_type")
        item_id = first.get("item_id")
        if media_type and item_id:
            return f"{media_type}://{item_id}"
        return None

    def _handle_music_command(self, user_text):
        normalized = user_text.strip().lower()

        play_query = self._extract_music_query(user_text)
        if (
            play_query is None
            and normalized not in MUSIC_PAUSE_WORDS
            and normalized not in MUSIC_RESUME_WORDS
            and normalized not in MUSIC_STOP_WORDS
            and normalized not in MUSIC_NEXT_WORDS
        ):
            return False

        if not MUSIC_ASSISTANT_TOKEN:
            self.speak("Music Assistant token is not set yet.")
            return True

        if not MUSIC_ASSISTANT_PLAYER_ID:
            self.speak("Music Assistant player id is not set yet.")
            return True

        if play_query is not None:
            if not play_query:
                self.speak("What should I play?")
                return True

            search_attempts = [
                ("music/search", {"search_query": play_query, "media_types": MUSIC_SEARCH_MEDIA_TYPES, "limit": 1}),
                ("music/search", {"query": play_query, "media_types": MUSIC_SEARCH_MEDIA_TYPES, "limit": 1}),
            ]
            ok, reason, detail = self._music_try_commands(search_attempts)
            if not ok:
                if reason:
                    self.speak(reason)
                else:
                    self.speak("Music search failed on Music Assistant.")
                return True

            _, _, search_data, _ = detail
            media_uri = self._extract_first_media_uri(search_data)
            if not media_uri:
                self.speak(f"I could not find {play_query}.")
                return True

            play_attempts = [
                ("player_queues/play_media", {"queue_id": MUSIC_ASSISTANT_PLAYER_ID, "media": [media_uri]}),
                ("player_queues/play_media", {"player_id": MUSIC_ASSISTANT_PLAYER_ID, "media": [media_uri]}),
                ("player_queues/play_media", {"queue_id": MUSIC_ASSISTANT_PLAYER_ID, "media": [{"uri": media_uri}]}),
                ("player_queues/play_media", {"player_id": MUSIC_ASSISTANT_PLAYER_ID, "media": [{"uri": media_uri}]}),
            ]
            ok, reason, _ = self._music_try_commands(play_attempts)
            if ok:
                self.speak(f"Playing {play_query}.")
            else:
                self.speak(reason or "I could not start playback on Music Assistant.")
            return True

        action_map = [
            (MUSIC_PAUSE_WORDS, "pause", "Paused."),
            (MUSIC_RESUME_WORDS, "resume", "Resumed."),
            (MUSIC_STOP_WORDS, "stop", "Stopped."),
            (MUSIC_NEXT_WORDS, "next", "Skipping."),
        ]
        for words, action, spoken in action_map:
            if normalized in words:
                action_attempts = [
                    (f"player_queues/{action}", {"queue_id": MUSIC_ASSISTANT_PLAYER_ID}),
                    (f"player_queues/{action}", {"player_id": MUSIC_ASSISTANT_PLAYER_ID}),
                    ("players/cmd", {"player_id": MUSIC_ASSISTANT_PLAYER_ID, "command": action}),
                ]
                ok, reason, _ = self._music_try_commands(action_attempts)
                if ok:
                    self.speak(spoken)
                else:
                    self.speak(reason or "I could not control Music Assistant.")
                return True

        return False

    def generate_and_speak(self, text):
        if self._handle_music_command(text):
            return

        if "reset" in text.lower() or "clear memory" in text.lower():
            self.conversation_history = [{"role": "system", "content": ""}] 
            self.speak("Memory cleared.")
            print("[SYSTEM] Memory Cleared")
            return

        explicit_search_query = self._extract_explicit_search_query(text)
        if explicit_search_query is not None:
            if not explicit_search_query:
                self.speak("What should I search up?")
                return
            grounded_reply = self._respond_from_web_grounded(explicit_search_query)
            print(f"[BOT (Grounded)] {grounded_reply}")
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": grounded_reply})
            self.speak(grounded_reply)
            return

        local_reply = self._answer_local_time_date(text)
        if local_reply:
            print(f"[BOT (Local)] {local_reply}")
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": local_reply})
            self.speak(local_reply)
            return

        current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        
        dynamic_system_prompt = f"""You are a helpful voice assistant.
The current date and time is: {current_time}.

Keep answers very concise (1-2 sentences).

CRITICAL RULES:
1. NORMAL CHAT: If the user says hello or asks a general question, reply normally. Do not use any prefixes.
2. TIME/DATE: Answer from local system time directly. Do not use web search for time/date.
3. WEB SEARCH: Only use web search if the user explicitly says "search up" or "look up".
4. CAMERA: If the user asks you to take a picture, look around, or see something, reply starting with exactly "CAPTURE_IMAGE:" followed by the question.

EXAMPLES:
User: Hello!
You: Hi there! How are you doing today?

User: What is the capital of Japan?
You: The capital of Japan is Tokyo.

User: Tell me the current date.
You: Today is {current_time}.

User: Search up current gold price
You: SEARCH_WEB: current gold price USD

User: Look up weather in Kuala Lumpur
You: SEARCH_WEB: Kuala Lumpur weather today

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
        
        response = ollama.chat(
            model=LLM_MODEL,
            messages=self.conversation_history,
            stream=False,
            options=LLM_OPTIONS,
        )
        full_response = response['message']['content'].strip()

        if "SEARCH_WEB:" in full_response:
            reminder = "Say search up, followed by your topic, if you want me to use web search."
            print(f"[BOT] {reminder}")
            self.conversation_history.append({"role": "assistant", "content": reminder})
            self.speak(reminder)
            return

        if "CAPTURE_IMAGE:" in full_response:
            vision_prompt = full_response.split("CAPTURE_IMAGE:")[-1].strip()
            
            self.speak("Let me take a look.")
            
            capture_path = self.take_picture()
            
            if capture_path:
                self.show_camera_image(capture_path)
                self.set_state(BotStates.THINKING)
                
                print(f"[VISION] Asking moondream: '{vision_prompt}' on file {capture_path}")
                try:
                    vision_msg = [{
                        "role": "user", 
                        "content": vision_prompt, 
                        "images": [capture_path] 
                    }]
                    
                    vision_resp = ollama.chat(model=VISION_MODEL, messages=vision_msg, stream=False)
                    vision_text = vision_resp['message']['content'].strip()
                    
                    self.hide_camera_image()
                    
                    print(f"[BOT (Vision)] {vision_text}")
                    
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
