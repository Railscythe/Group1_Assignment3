from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_path = "model/fashion_mnist_model.pkl"
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
    """Process a frame sent by the client."""
    try:
        # Get the image data from the client
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array and preprocess
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_frame = preprocess_frame(frame)

        # Make a prediction
        predictions = model.predict(processed_frame)
        class_idx = np.argmax(predictions)
        label = fashion_classes[class_idx]
        probability = predictions[0][class_idx] * 100

        return f"{label},{probability:.2f}"
    except Exception as e:
        print(f"Error: {e}")
        return "Error,Unable to process frame", 500

if __name__ == "__main__":
    app.run(debug=True)
