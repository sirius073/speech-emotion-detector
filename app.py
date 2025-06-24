import streamlit as st
import torch
import numpy as np
import librosa
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import EnhancedCNNLSTM  # Make sure your model class is in model.py

# --------------------------
# Load Model and Setup
# --------------------------
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedCNNLSTM(num_classes=8)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# --------------------------
# Feature Extraction
# --------------------------
def extract_features(data, sample_rate):
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def process_audio(file):
    y, sr = librosa.load(file, duration=2.5, offset=0.6)
    features = extract_features(y, sr)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])[0]
    return features_scaled.reshape(1, 1, 9, 18)

# --------------------------
# Prediction
# --------------------------
def predict_emotion(file):
    model, device = load_model()
    tensor = torch.tensor(process_audio(file), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return emotion_labels[pred_idx]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🎧 Emotion Recognition", layout="centered")
st.title("🎤 Speech Emotion Recognizer")
st.write("Upload a `.wav` audio file and the model will classify the emotion.")

uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("Analyzing emotion..."):
        try:
            prediction = predict_emotion(uploaded_file)
            st.success(f"🎯 Predicted Emotion: **{prediction.upper()}**")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
