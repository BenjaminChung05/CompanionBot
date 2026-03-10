from flask import Flask, Response
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import io
import threading

app = Flask(__name__)

# Initialize Picamera2
picam2 = Picamera2()

# Configure the camera (640x480 is great for smooth, low-latency streaming)
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)

# Create a thread-safe custom output class to catch the JPEG frames
class StreamOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            # Notify the generator that a new frame is ready
            self.condition.notify_all()
        return len(buf)

stream_output = StreamOutput()

# Start continuous recording using the Pi's hardware JPEG encoder
picam2.start_recording(JpegEncoder(), FileOutput(stream_output))
picam2.start()

def generate_frames():
    while True:
        # Wait for the next hardware-encoded frame
        with stream_output.condition:
            stream_output.condition.wait()
            frame = stream_output.frame
            
        if frame:
            # Yield the raw JPEG stream to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # threaded=True allows multiple devices to view the stream without crashing
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)