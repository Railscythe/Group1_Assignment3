from flask import Flask, Response, render_template, request
import pickle
import cv2
import numpy as np
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the model from the .pkl file
pkl_model_path = "model/fashion_mnist_model.pkl"
with open(pkl_model_path, "rb") as pkl_file:
    model = pickle.load(pkl_file)

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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process the frame sent from the client."""
    try:
        # Get image data from client
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Debug: Print the received image data length
        print(f"Received image data length: {len(image_bytes)}")

        # Convert to numpy array and preprocess
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_frame = preprocess_frame(frame)

        # Make prediction
        predictions = model.predict(processed_frame, verbose=0)
        class_idx = np.argmax(predictions)
        label = fashion_classes[class_idx]
        probability = predictions[0][class_idx] * 100

        # Debug: Print prediction details
        print(f"Predicted Label: {label}, Probability: {probability:.2f}%")

        return f"{label},{round(probability, 2)}"
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return f"Error,{str(e)}"


@app.route('/')
def index():
    """Main route to render the HTML page."""
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
