from flask import Flask, render_template, jsonify, Response
import threading
import time
import cv2

app = Flask(__name__)

# Global variable to store the count of detected people
count = 0

# Function to increment the count
def increment_count():
    global count
    while True:
        time.sleep(1)  # Sleep for 1 second to update the count
        count += 1

# Start a separate thread to increment the count
count_thread = threading.Thread(target=increment_count)
count_thread.daemon = True
count_thread.start()

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Load the pre-trained HOG detector for people
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(frame):
    # Detect people in the frame using HOG
    (rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    return rects

def generate_frames():
    global count
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect people in the frame
            rects = detect_people(frame)

            # Draw bounding boxes around detected people
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Update the count based on the number of detected people
            count = len(rects)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index-2.html', count=count)

@app.route('/get_count')
def get_count():
    global count
    return jsonify(count=count)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
