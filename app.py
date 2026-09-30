import os
import uuid
import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORT_FOLDER'] = 'reports'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# ----------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------
image_model = None
video_model = None
frame_feature_extractor = None
audio_model = None

IMG_SIZE = (299, 299)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (299, 299))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return np.array(frames)

def load_audio(file_path, target_sr=16000, duration=4):
    signal, sr = librosa.load(file_path, sr=target_sr, duration=duration)
    if len(signal) < target_sr * duration:
        signal = np.pad(signal, (0, target_sr*duration - len(signal)))
    else:
        signal = signal[:target_sr*duration]
    return signal

def audio_to_melspectrogram(signal, sr=16000, n_mels=128, fmax=8000):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, fmax=fmax)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - np.min(log_mel)) / (np.max(log_mel) - np.min(log_mel) + 1e-8)
    return log_mel

# ----------------------------------------------------------------------
# Load models (direct .h5 files)
# ----------------------------------------------------------------------
def load_models():
    global image_model, video_model, frame_feature_extractor, audio_model

    # Image model
    if os.path.exists('models/image_deepfake_model.h5'):
        image_model = keras.models.load_model('models/image_deepfake_model.h5')
        print("✅ Image model loaded.")
    else:
        print("❌ Image model not found at models/image_deepfake_model.h5")

    # Video model (LSTM) and feature extractor
    if os.path.exists('models/video_deepfake_model.h5'):
        video_model = keras.models.load_model('models/video_deepfake_model.h5')
        # Build feature extractor (Xception)
        base = applications.Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))
        base.trainable = False
        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        frame_feature_extractor = models.Model(inputs=base.input, outputs=x)
        print("✅ Video model loaded.")
    else:
        print("❌ Video model not found at models/video_deepfake_model.h5")

    # Audio model
    if os.path.exists('models/audio_deepfake_model.h5'):
        audio_model = keras.models.load_model('models/audio_deepfake_model.h5')
        print("✅ Audio model loaded.")
    else:
        print("❌ Audio model not found at models/audio_deepfake_model.h5")

load_models()

# ----------------------------------------------------------------------
# Prediction functions (same as before)
# ----------------------------------------------------------------------
def predict_image(file_path):
    if image_model is None:
        return "REAL", 50.0
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)
    pred = float(image_model.predict(img_batch, verbose=0)[0][0])
    if pred > 0.5:
        return "FAKE", pred * 100
    else:
        return "REAL", (1 - pred) * 100

def predict_video(file_path):
    if video_model is None or frame_feature_extractor is None:
        return "REAL", 50.0
    frames = extract_frames(file_path, num_frames=10)
    if len(frames) == 0:
        return "REAL", 50.0
    features = []
    for frame in frames:
        frame = tf.keras.applications.xception.preprocess_input(frame)
        f = frame_feature_extractor.predict(np.expand_dims(frame, axis=0), verbose=0)
        features.append(f.flatten())
    features = np.expand_dims(np.array(features), axis=0)
    pred = float(video_model.predict(features, verbose=0)[0][0])
    if pred > 0.5:
        return "FAKE", pred * 100
    else:
        return "REAL", (1 - pred) * 100

def predict_audio(file_path):
    if audio_model is None:
        return "REAL", 50.0
    signal = load_audio(file_path)
    mel = audio_to_melspectrogram(signal)
    mel = np.expand_dims(mel, axis=0)
    mel = np.expand_dims(mel, axis=-1)
    pred = float(audio_model.predict(mel, verbose=0)[0][0])
    if pred > 0.5:
        return "FAKE", pred * 100
    else:
        return "REAL", (1 - pred) * 100

# ----------------------------------------------------------------------
# Recent detections and routes (same as before)
# ----------------------------------------------------------------------
recent_detections = [
    {"file": "face_01.jpg", "type": "Image", "result": "FAKE", "confidence": 87.6, "report_id": "sample1"},
    {"file": "real_speech.wav", "type": "Audio", "result": "REAL", "confidence": 92.3, "report_id": "sample2"},
    {"file": "deepfake_vid.mp4", "type": "Video", "result": "FAKE", "confidence": 94.1, "report_id": "sample3"},
]

def generate_report(file_name, file_type, result, confidence):
    report_id = str(uuid.uuid4())[:8]
    report_path = os.path.join(app.config['REPORT_FOLDER'], f"{report_id}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Deepfake Detection Report\n")
        f.write(f"========================\n")
        f.write(f"File: {file_name}\n")
        f.write(f"Type: {file_type}\n")
        f.write(f"Result: {result}\n")
        f.write(f"Confidence: {confidence:.1f}%\n")
    return report_id

@app.route('/')
def index():
    return render_template('index.html', detections=recent_detections)

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext in {'png', 'jpg', 'jpeg', 'bmp'}:
        file_type = "Image"
        predict_func = predict_image
    elif ext in {'mp3', 'wav', 'm4a'}:
        file_type = "Audio"
        predict_func = predict_audio
    elif ext in {'mp4', 'avi', 'mov'}:
        file_type = "Video"
        predict_func = predict_video
    else:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result, confidence = predict_func(filepath)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    report_id = generate_report(filename, file_type, result, confidence)

    recent_detections.insert(0, {
        "file": filename,
        "type": file_type,
        "result": result,
        "confidence": round(confidence, 1),
        "report_id": report_id
    })
    if len(recent_detections) > 10:
        recent_detections.pop()

    return jsonify({
        "result": result,
        "confidence": confidence,
        "report_id": report_id,
        "file_type": file_type
    })

@app.route('/download_report/<report_id>')
def download_report(report_id):
    report_path = os.path.join(app.config['REPORT_FOLDER'], f"{report_id}.txt")
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name=f"deepfake_report_{report_id}.txt")
    return "Report not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)