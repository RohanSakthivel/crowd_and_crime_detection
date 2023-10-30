import cv2
import numpy as np
import base64
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

custom_weights_path = "custom_yolov3.weights"
custom_cfg_path = "custom_yolov3.cfg"
classes_path = "classes.txt"

custom_net = cv2.dnn.readNet(custom_weights_path, custom_cfg_path)
output_layer_names = custom_net.getUnconnectedOutLayersNames()
classes = []

with open(classes_path, 'r') as file:
    classes = file.read().strip().split('\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_data = data['image'][22:]  # Remove 'data:image/jpeg;base64,' prefix
    image_bytes = base64.b64decode(image_data)

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = perform_object_detection(image)
    return jsonify({'message': result})

def perform_object_detection(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    custom_net.setInput(blob)
    outs = custom_net.forward(output_layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    result = ""
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            result += f"{label} (Confidence: {confidence:.2f})\n"

    return result

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000,debug=True)
