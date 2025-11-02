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
    """Load the pre-trained model or create a new one"""
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from {MODEL_PATH}...")
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new untrained model...")
            return create_model()
    else:
        print("WARNING: No pre-trained model found!")
        print("Creating new untrained model. Use /retrain endpoint to train it.")
        return create_model()

# Load model at startup
print("Initializing model...")
model = load_model()
print("Model loaded successfully!")

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

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'CIFAR-10 Image Classification API',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /predict': 'Classify an image (send base64 encoded image in JSON)',
            'POST /retrain': 'Retrain the model (may take several minutes)'
        },
        'model_status': 'loaded' if model else 'not loaded'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for image classification predictions"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided. Send JSON with "image" field containing base64 encoded image'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
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
    """Endpoint to retrain the model (WARNING: This takes several minutes)"""
    try:
        global model
        print("Starting model retraining...")
        model = train_and_save_model()
        print("Model retraining completed!")
        return jsonify({
            'message': 'Model retrained successfully',
            'model_path': MODEL_PATH
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_file_exists': model_exists,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print("Starting Flask server...")
    print(f"Server running on port {port}")
    print("Endpoints:")
    print("  GET  / - API information")
    print("  POST /predict - Classify an image (send base64 encoded image in JSON)")
    print("  POST /retrain - Retrain the model")
    print("  GET  /health - Health check")
    
    app.run(host='0.0.0.0', port=port, debug=False)