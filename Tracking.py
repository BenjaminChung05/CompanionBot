import cv2
import os
import threading
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
from picamera2 import Picamera2

app = Flask(__name__)

# --- Configuration ---
BASE_MODEL = "yolo11n.pt"
NCNN_MODEL_DIR = "yolo11n_ncnn_model"
CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 900
INFERENCE_SIZE = 190 
COLOR_MATCH_THRESHOLD = 0.3

output_frame = None
condition = threading.Condition() 

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

def vision_loop():
    global output_frame
    
    model = setup_model()
    print("Initializing Raspberry Pi Camera V2...")
    picam2 = Picamera2()
    
    config = picam2.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # [NEW] Strict separation of Display ID vs ByteTrack's Internal ID
    DISPLAY_ID = 1 
    current_internal_id = None 
    target_color_profile = None 
    
    center_screen_x = CAMERA_WIDTH // 2

    print("Vision loop running. YOLO active.")

    while True:
        frame = picam2.capture_array()
        results = model.track(frame, classes=[0], conf=0.4, tracker="botsort.yaml", persist=True, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # 1. Initial Lock On (Strictly waits for ByteTrack to output ID 1)
            if current_internal_id is None:
                if 1.0 in track_ids:
                    current_internal_id = 1.0

            # 2. Fallback: If internal ID is lost, find them via color and map them back
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
                
                # We found them! Map ByteTrack's new ID back to our tracker.
                if best_match_id is not None:
                    current_internal_id = best_match_id

            # 3. Process Target & Draw
            if current_internal_id in track_ids:
                idx = list(track_ids).index(current_internal_id)
                target_box = boxes[idx]
                xmin, ymin, xmax, ymax = map(int, target_box)

                # [NEW] Grab the color profile ONLY ONCE to prevent drift
                if target_color_profile is None:
                    target_color_profile = get_color_profile(frame, target_box)

                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                error_x = cx - center_screen_x

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Always display ID 1 to the user/system
                label = f"Locked ID: {DISPLAY_ID} | Error: {error_x}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                cv2.rectangle(frame, (10, 30 - text_h - 5), (10 + text_w, 30 + 5), (0, 255, 0), -1)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
            else:
                cv2.putText(frame, f"Searching for ID: {DISPLAY_ID}...", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "No humans detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=vision_loop, daemon=True)
    t.start()
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)