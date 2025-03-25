from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import threading
import webbrowser
import socket
from logic.multi_face_test import process_face  # Import processing function

app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Get local IP for mobile access
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

# ✅ Serve the HTML page
@app.route('/')
def home():
    return render_template("index.html")

# ✅ Process frame from browser automatically
@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print("Received Frame from Browser!")  # Debugging log

        # ✅ Process the frame using multi_face_test.py
        result = process_face(frame)

        if not result:  # If result is empty or None, return an error message
            print("Error: No valid response from process_face()")
            return jsonify({"error": "No face detected or processing failed", "faces": []})

        print(f"Result: {result}")  # Debugging log

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")  # Debugging log
        return jsonify({"error": str(e), "faces": []}), 500  # Always return a valid JSON format

# ✅ Open browser automatically
def open_browser():
    local_ip = get_local_ip()
    webbrowser.open(f"http://{local_ip}:5000/")

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"Access Flask on: http://{local_ip}:5000/")
    threading.Timer(1, open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=5000)
