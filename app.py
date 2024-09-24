from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Directory to save processed images
processed_image_dir = 'static/processed'
if not os.path.exists(processed_image_dir):
    os.makedirs(processed_image_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])  # Updated route name
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']  # Corrected to match the FormData append name

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image in OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(image)

    # Draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]  # Get coordinates of the bounding box
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"  # Label with confidence
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed image
    processed_image_name = f"processed_{uuid.uuid4()}.jpg"
    processed_image_path = os.path.join(processed_image_dir, processed_image_name)
    cv2.imwrite(processed_image_path, image)

    # Respond with the processed image path and detections
    return jsonify({
        'processed_image': processed_image_name,
        'detections': [box.xyxy[0].tolist() + [model.names[int(box.cls[0])], box.conf[0].item()] for box in r.boxes]
    })

@app.route('/processed/<path:filename>', methods=['GET'])
def send_processed_image(filename):
    return send_from_directory(processed_image_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)



    # flask run 
