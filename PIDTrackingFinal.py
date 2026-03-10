import cv2
import os
import threading
import time
import numpy as np
import RPi.GPIO as GPIO 
from flask import Flask, Response, request, jsonify, render_template_string
from ultralytics import YOLO
from picamera2 import Picamera2
from gpiozero import PWMOutputDevice, DigitalOutputDevice

app = Flask(__name__)

# --- Configuration ---
BASE_MODEL = "yolo11n.pt"
NCNN_MODEL_DIR = "yolo11n_ncnn_model"
CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 1400
INFERENCE_SIZE = 200 
COLOR_MATCH_THRESHOLD = 0.3
STOPPING_DISTANCE_CM = 35.0 
SIDE_CLEARANCE_CM = 15.0 

output_frame = None
condition = threading.Condition() 

# --- Global State ---
manual_override = False
ai_paused = False  # Tracks if the AI Agent is currently using the CPU
latest_distance = 999.0  # Stores the current distance for the AI to read

# --- Ultrasonic Setup ---
GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)

# Pin Definitions 
FRONT_TRIG = 17
FRONT_ECHO = 27
LEFT_TRIG = 2
LEFT_ECHO = 3
RIGHT_TRIG = 23
RIGHT_ECHO = 24

# Setup Outputs
for trig in [FRONT_TRIG, LEFT_TRIG, RIGHT_TRIG]:
    GPIO.setup(trig, GPIO.OUT)
# Setup Inputs
for echo in [FRONT_ECHO, LEFT_ECHO, RIGHT_ECHO]:
    GPIO.setup(echo, GPIO.IN)

# --- Motor Setup ---
ena = PWMOutputDevice(20, frequency=800) 
in1 = DigitalOutputDevice(26) 
in2 = DigitalOutputDevice(19) 

enb = PWMOutputDevice(5, frequency=800)  
in3 = DigitalOutputDevice(13) 
in4 = DigitalOutputDevice(6)

# --- PID Tuning Constants ---
KP_TURN = 0.002
KD_TURN = 0.0005  
KI_TURN = 0.009

KP_DIST = 0.00002 
KD_DIST = 0.00001 
TARGET_AREA = 150000 

DEAD_ZONE_X = 50     
DEAD_ZONE_AREA = 20000 

# --- Helper Functions ---
def get_distance(trigger_pin, echo_pin):
    GPIO.output(trigger_pin, True)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, False)

    StartTime = time.time()
    StopTime = time.time()
    
    timeout_limit = StartTime + 0.04 

    while GPIO.input(echo_pin) == 0:
        StartTime = time.time()
        if StartTime > timeout_limit: return 999.0 

    while GPIO.input(echo_pin) == 1:
        StopTime = time.time()
        if StopTime > timeout_limit + 0.04: return 999.0 

    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    return distance

def set_motors(left_speed, right_speed):
    left_speed = max(min(left_speed, 0.6), -0.8)
    right_speed = max(min(right_speed, 0.6), -0.8)

    if left_speed >= 0:
        in1.on()
        in2.off()
        ena.value = left_speed
    else:
        in1.off()
        in2.on()
        ena.value = -left_speed

    if right_speed >= 0:
        in3.on()
        in4.off()
        enb.value = right_speed
    else:
        in3.off()
        in4.on()
        enb.value = -right_speed

def stop_motors():
    ena.value = 0
    enb.value = 0
    in1.off()
    in2.off()
    in3.off()
    in4.off()

def setup_model():
    if not os.path.exists(NCNN_MODEL_DIR):
        print(f"Optimized model not found. Exporting {BASE_MODEL} to NCNN...")
        temp_model = YOLO(BASE_MODEL)
        temp_model.export(format="ncnn", imgsz=INFERENCE_SIZE)
        print("Export complete!")
    return YOLO(NCNN_MODEL_DIR)

def get_color_profile(frame, box):
    xmin, ymin, xmax, ymax = map(int, box)
    w, h = xmax - xmin, ymax - ymin
    
    cx_min, cx_max = int(xmin + w * 0.25), int(xmax - w * 0.25)
    cy_min, cy_max = int(ymin + h * 0.25), int(ymax - h * 0.25)
    
    cx_min, cy_min = max(0, cx_min), max(0, cy_min)
    cx_max, cy_max = min(frame.shape[1], cx_max), min(frame.shape[0], cy_max)
    
    roi = frame[cy_min:cy_max, cx_min:cx_max]
    
    if roi.size == 0:
        return None
        
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return hist

def get_closest_person(boxes, track_ids):
    if len(boxes) == 0:
        return None, None
        
    largest_area = 0
    best_id = None
    best_box = None
    
    for box, tid in zip(boxes, track_ids):
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > largest_area:
            largest_area = area
            best_id = tid
            best_box = box
            
    return best_id, best_box

def vision_loop():
    global output_frame, manual_override, ai_paused, latest_distance
    
    model = setup_model()
    print("Initializing Raspberry Pi Camera V2...")
    picam2 = Picamera2()
    
    config = picam2.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    DISPLAY_ID = 1 
    current_internal_id = None 
    target_color_profile = None 
    center_screen_x = CAMERA_WIDTH // 2

    prev_error_x = 0
    prev_error_area = 0
    integral_x = 0
    last_time = time.time()

    print("Vision loop running. YOLO active. Motors ready.")

    while True:
        frame = picam2.capture_array()
        
        results = model.track(frame, classes=[0], conf=0.4, tracker="botsort.yaml", persist=True, verbose=False)
        
        current_time = time.time()
        dt = current_time - last_time
        if dt == 0: dt = 0.001 
        last_time = current_time

        target_locked = False
        is_searching = False
        boxes = []
        track_ids = []

        if manual_override:
            cv2.putText(frame, "*** MANUAL OVERRIDE ACTIVE ***", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

        # 1. Initial Lock On
        if current_internal_id is None:
            if 1.0 in track_ids:
                current_internal_id = 1.0

        # 2. Fallback Re-ID via Color
        if current_internal_id is not None and current_internal_id not in track_ids and target_color_profile is not None:
            best_match_id = None
            best_match_score = float('inf')

            for box, track_id in zip(boxes, track_ids):
                candidate_profile = get_color_profile(frame, box)
                if candidate_profile is not None:
                    score = cv2.compareHist(target_color_profile, candidate_profile, cv2.HISTCMP_BHATTACHARYYA)
                    if score < best_match_score and score < COLOR_MATCH_THRESHOLD:
                        best_match_score = score
                        best_match_id = track_id
            
            if best_match_id is not None:
                current_internal_id = best_match_id

        # 3. Target Lost Protocol - Search & Reassign
        if current_internal_id is not None and current_internal_id not in track_ids:
            best_id, best_box = get_closest_person(boxes, track_ids)
            
            if best_id is not None:
                current_internal_id = best_id
                target_color_profile = get_color_profile(frame, best_box)
                DISPLAY_ID = 1 
                
                integral_x = 0 
                prev_error_x = 0
                prev_error_area = 0
                
            else:
                is_searching = True
                
                # Read all three sensors
                current_distance = get_distance(FRONT_TRIG, FRONT_ECHO)
                latest_distance = current_distance # Update telemetry for AI Agent
                dist_left = get_distance(LEFT_TRIG, LEFT_ECHO)
                dist_right = get_distance(RIGHT_TRIG, RIGHT_ECHO)
                
                if current_distance > STOPPING_DISTANCE_CM:
                    search_fwd_speed = 0.35
                    # Default search direction based on the last known error
                    search_turn_speed = 0.75 if prev_error_x > 0 else -0.75
                    
                    # --- SAFER SEARCH OVERRIDE ---
                    if search_turn_speed > 0 and dist_right < SIDE_CLEARANCE_CM:
                        search_turn_speed = -0.75
                        cv2.putText(frame, "WALL RIGHT! EVADING LEFT", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    elif search_turn_speed < 0 and dist_left < SIDE_CLEARANCE_CM:
                        search_turn_speed = 0.75
                        cv2.putText(frame, "WALL LEFT! EVADING RIGHT", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # ------------------------------

                    left_motor_speed = search_fwd_speed + search_turn_speed
                    right_motor_speed = search_fwd_speed - search_turn_speed
                    
                    # --- AI Pause Check for Searching ---
                    if not manual_override and not ai_paused:
                        set_motors(left_motor_speed, right_motor_speed)
                    elif ai_paused:
                        stop_motors()
                        cv2.putText(frame, "AI THINKING...", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
                        
                    cv2.putText(frame, "LOST! SEARCHING LAST POS...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(frame, f"F:{current_distance:.0f}cm L:{dist_left:.0f}cm R:{dist_right:.0f}cm", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    if not manual_override:
                        stop_motors()
                    cv2.putText(frame, "OBSTACLE: CANNOT SEARCH.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4. Standard Tracking and PID Execution
        if current_internal_id is not None and current_internal_id in track_ids:
            target_locked = True
            idx = list(track_ids).index(current_internal_id)
            target_box = boxes[idx]
            xmin, ymin, xmax, ymax = map(int, target_box)

            if target_color_profile is None:
                target_color_profile = get_color_profile(frame, target_box)

            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
            w, h = xmax - xmin, ymax - ymin
            current_area = w * h

            error_x = center_screen_x - cx
            error_area = TARGET_AREA - current_area 

            if abs(error_x) < DEAD_ZONE_X: error_x = 0
            if abs(error_area) < DEAD_ZONE_AREA: error_area = 0

            integral_x += error_x * dt
            derivative_x = (error_x - prev_error_x) / dt
            turn_speed = (KP_TURN * error_x) + (KI_TURN * integral_x) + (KD_TURN * derivative_x)
            prev_error_x = error_x

            derivative_area = (error_area - prev_error_area) / dt
            forward_speed = (KP_DIST * error_area) + (KD_DIST * derivative_area)
            prev_error_area = error_area

            left_motor_speed = forward_speed + turn_speed
            right_motor_speed = forward_speed - turn_speed

            # --- ULTRASONIC SAFETY CHECK ---
            current_distance = get_distance(FRONT_TRIG, FRONT_ECHO)
            latest_distance = current_distance # Update telemetry for AI Agent
            
            if current_distance < STOPPING_DISTANCE_CM:
                if not manual_override:
                    stop_motors()
                cv2.putText(frame, f"OBSTACLE: {current_distance:.1f}cm! STOPPED.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # --- AI Pause Check for Tracking ---
                if not manual_override and not ai_paused:
                    set_motors(left_motor_speed, right_motor_speed)
                elif ai_paused:
                    stop_motors()
                    cv2.putText(frame, "AI THINKING...", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
                    
                cv2.putText(frame, f"Dist: {current_distance:.1f}cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # -------------------------------

            # Draw Visuals
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            label = f"Locked ID: {DISPLAY_ID} | Turn: {turn_speed:.2f} | Fwd: {forward_speed:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, 30 - text_h - 5), (10 + text_w, 30 + 5), (0, 255, 0), -1)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        elif not is_searching:
            cv2.putText(frame, f"Searching for ID: {DISPLAY_ID}...", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if not target_locked and not is_searching:
            if not manual_override:
                stop_motors()
            prev_error_area = 0
            integral_x = 0

        # Encode Frame for Flask (Used by Web UI and AI Agent)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ret:
            with condition:
                output_frame = buffer.tobytes()
                condition.notify_all()

def generate_frames():
    global output_frame
    while True:
        with condition:
            condition.wait() 
            frame_data = output_frame
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

# --- FLASK ROUTES FOR WEB INTERFACE ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background: #222; color: #fff; margin: 0; padding: 20px;}
        .container { max-width: 1000px; margin: auto; }
        img { width: 100%; border-radius: 10px; border: 3px solid #555; }
        .controls-container { margin-top: 20px; }
        .btn { padding: 15px 30px; font-size: 20px; font-weight: bold; margin: 5px; cursor: pointer; border-radius: 8px; border: none; }
        .btn-mode { background: #007bff; color: white; width: 300px; padding: 20px; margin-bottom: 20px; }
        .btn-mode.manual { background: #dc3545; }
        .btn-dir { background: #444; color: white; width: 100px; height: 80px; transition: background 0.1s; }
        .btn-dir:hover { background: #666; }
        .btn-dir.active { background: #ff9800; color: #000; }
        
        #dpad { display: flex; flex-direction: column; align-items: center; opacity: 0.3; pointer-events: none; transition: 0.3s; }
        .row { display: flex; justify-content: center; }
        .instructions { margin-top: 20px; color: #aaa; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Autonomous Robot View</h2>
        <img src="/video_feed" alt="Camera Feed">
        
        <div class="controls-container">
            <button id="modeBtn" class="btn btn-mode" onclick="toggleMode()">Mode: AUTONOMOUS</button>
            
            <div id="dpad">
                <div class="row">
                    <button id="btn-w" class="btn btn-dir" onmousedown="sendCommand('forward')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('forward')" ontouchend="sendCommand('stop')">W</button>
                </div>
                <div class="row">
                    <button id="btn-a" class="btn btn-dir" onmousedown="sendCommand('left')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('left')" ontouchend="sendCommand('stop')">A</button>
                    <button id="btn-s" class="btn btn-dir" onmousedown="sendCommand('backward')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('backward')" ontouchend="sendCommand('stop')">S</button>
                    <button id="btn-d" class="btn btn-dir" onmousedown="sendCommand('right')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('right')" ontouchend="sendCommand('stop')">D</button>
                </div>
                <div class="instructions">Use W, A, S, D keys to drive when in Manual Mode.</div>
            </div>
        </div>
    </div>

    <script>
        let isManual = false;
        const keysPressed = {};

        function toggleMode() {
            fetch('/toggle_mode', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                isManual = data.manual;
                const modeBtn = document.getElementById('modeBtn');
                const dpad = document.getElementById('dpad');
                
                if (isManual) {
                    modeBtn.innerText = "Mode: MANUAL";
                    modeBtn.classList.add("manual");
                    dpad.style.opacity = "1";
                    dpad.style.pointerEvents = "auto";
                } else {
                    modeBtn.innerText = "Mode: AUTONOMOUS";
                    modeBtn.classList.remove("manual");
                    dpad.style.opacity = "0.3";
                    dpad.style.pointerEvents = "none";
                    sendCommand('stop'); 
                }
            });
        }

        function sendCommand(action) {
            if (!isManual) return;
            fetch('/manual_cmd', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            });
        }

        window.addEventListener('keydown', (e) => {
            if (!isManual) return;
            const key = e.key.toLowerCase();
            
            if (keysPressed[key]) return; 
            keysPressed[key] = true;

            const btn = document.getElementById('btn-' + key);
            if (btn) btn.classList.add('active');

            if (key === 'w') sendCommand('forward');
            if (key === 's') sendCommand('backward');
            if (key === 'a') sendCommand('left');
            if (key === 'd') sendCommand('right');
        });

        window.addEventListener('keyup', (e) => {
            if (!isManual) return;
            const key = e.key.toLowerCase();
            keysPressed[key] = false;

            const btn = document.getElementById('btn-' + key);
            if (btn) btn.classList.remove('active');

            if (['w', 'a', 's', 'd'].includes(key)) {
                if (!keysPressed['w'] && !keysPressed['a'] && !keysPressed['s'] && !keysPressed['d']) {
                    sendCommand('stop');
                }
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Endpoints for AI Agent Communication ---
@app.route('/snapshot')
def snapshot():
    """Serves a single JPEG frame from memory for the AI Vision model."""
    global output_frame
    if output_frame is None:
        return "No frame ready", 503
    return Response(output_frame, mimetype='image/jpeg')

@app.route('/ai_override', methods=['POST'])
def ai_override():
    """Allows the AI agent to pause and resume the robot's movement."""
    global ai_paused
    data = request.json
    ai_paused = data.get('pause', False)
    
    if ai_paused:
        stop_motors() # Immediately cut power to wheels
        
    return jsonify({"status": "ok", "paused": ai_paused})

@app.route('/status')
def status():
    """Tells the AI agent how far away the obstacle is."""
    global latest_distance
    return jsonify({"distance": latest_distance})
# --------------------------------------------

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    global manual_override
    manual_override = not manual_override
    if manual_override:
        stop_motors()
    return jsonify({"manual": manual_override})

@app.route('/manual_cmd', methods=['POST'])
def manual_cmd():
    global manual_override
    if not manual_override:
        return jsonify({"status": "ignored"})
    
    data = request.json
    action = data.get('action')
    
    speed = 0.55 
    turn = 0.60  
    
    # Inverted W and S logic applied here
    if action == 'forward':
        set_motors(-speed, -speed)
    elif action == 'backward':
        set_motors(speed, speed)
    elif action == 'left':
        set_motors(turn, -turn)
    elif action == 'right':
        set_motors(-turn, turn)
    elif action == 'stop':
        stop_motors()
        
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    stop_motors()
    
    t = threading.Thread(target=vision_loop, daemon=True)
    t.start()
    
    print("Starting Flask server on port 5000...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        stop_motors()
        GPIO.cleanup()