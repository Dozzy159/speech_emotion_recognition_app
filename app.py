from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
import librosa  # Make sure librosa is imported
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model(r'C:\Users\DELL\Emotional Speech Data\speech_emotion_recognition_project.keras')

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the audio file as needed
        input_data = preprocess_audio(file_path)

        # Make predictions
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_probs = predictions[0]  # Get the probabilities for all classes
        
        # Map the predicted class to emotion labels
        emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
        predicted_emotion = emotion_labels[int(predicted_class[0])]
        
        return jsonify({
            'predicted_emotion': predicted_emotion,
            'probabilities': predicted_probs.tolist()  # Convert numpy array to list
        })

def preprocess_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=22050, duration=3, offset=0.5)
    features = extract_features(y, sr)  # Extract features
    return features.reshape(1, -1, 1)  # Reshape for LSTM input

def extract_features(y, sr):
    """
    Extracts features from audio data.

    Parameters:
    - y: Audio time series
    - sr: Sample rate of the audio

    Returns:
    - Combined features (MFCC, Chroma, Spectral Contrast)
    """
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)  # Extract MFCC features
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)  # Extract Chroma features
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)  # Extract Spectral Contrast
    return np.hstack((mfcc, chroma, spectral_contrast))  # Combine features

if __name__ == '__main__':
    app.run(debug=True)