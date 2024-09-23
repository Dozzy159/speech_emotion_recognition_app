# Import the necessary libraries

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('speech_emotion_recognition_project.keras')

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json  # Get the JSON data from the request
    
    input_data = np.array(data['input']).reshape(1, -1)  # Preprocess as needed
    
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)
    
    return jsonify({'predicted_class': int(predicted_class[0])})

if __name__ == '__main__':
    app.run(debug=True)