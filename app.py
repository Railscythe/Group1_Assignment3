from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_path = "models/fashion_mnist_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define class labels
fashion_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess_frame(frame):
    """Preprocess frame for model prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))
    return reshaped

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array and preprocess
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_frame = preprocess_frame(frame)

        # Make prediction
        predictions = model.predict(processed_frame)
        class_idx = np.argmax(predictions)
        label = fashion_classes[class_idx]
        probability = predictions[0][class_idx] * 100

        print(f"Predicted Label: {label}, Probability: {probability:.2f}%")
        return f"{label},{round(probability, 2)}"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error,Invalid frame", 500

if __name__ == "__main__":
    app.run(debug=False)
    
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
