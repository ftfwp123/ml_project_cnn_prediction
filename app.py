import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = 'cifar10_cnn_model.h5'
IMG_SIZE = (32, 32)

def create_model():
    """Create a CNN model for CIFAR-10 classification"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    """Train CNN model on CIFAR-10 dataset and save it"""
    print("Training CNN model on CIFAR-10 dataset...")
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Create and train model
    model = create_model()
    model.fit(x_train, y_train, 
              epochs=10, 
              validation_data=(x_test, y_test),
              batch_size=64,
              verbose=1)
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def load_model():
    """Load the pre-trained model"""
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("No pre-trained model found. Training new model...")
        return train_and_save_model()

# Load model at startup
model = load_model()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image to match model input size
    image = image.resize(IMG_SIZE)
    # Convert to array and normalize
    image_array = np.array(image).astype('float32') / 255
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for image classification predictions"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Prepare response
        response = {
            'predicted_class': class_names[predicted_class],
            'confidence': confidence,
            'all_predictions': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the model"""
    try:
        global model
        model = train_and_save_model()
        return jsonify({'message': 'Model retrained successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Endpoints:")
    print("  POST /predict - Classify an image (send base64 encoded image in JSON)")
    print("  POST /retrain - Retrain the model")
    print("  GET  /health - Health check")
    
    app.run(host='0.0.0.0', port=5500, debug=False)
# comment something