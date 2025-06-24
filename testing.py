import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from model import EnhancedCNNLSTM



# ====================
# Feature Extraction
# ====================
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

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(data, sample_rate)
    return res

# ====================
# Inference
# ====================
def predict_emotions(folder_path, scaler_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")

    # Load model
    model = EnhancedCNNLSTM(num_classes=8)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Emotion classes (order should match training)
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

    # Initialize scaler
    scaler = StandardScaler()

    features_list = []
    file_list = []

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            path = os.path.join(folder_path, file)
            features = get_features(path)
            features_list.append(features)
            file_list.append(file)

    features_array = np.array(features_list)
    features_scaled = scaler.fit_transform(features_array)  # Or load a pre-fitted scaler if available

    results = []

    for i in range(len(features_scaled)):
        sample = features_scaled[i].reshape(1, 1, 9, 18)  # Match CNN input
        tensor = torch.tensor(sample, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            emotion = emotion_labels[pred_idx]

        results.append((file_list[i], emotion))

    # Print and save predictions
    for fname, emo in results:
        print(f"{fname}: {emo}")

    df = pd.DataFrame(results, columns=["filename", "predicted_emotion"])
    df.to_csv("emotion_predictions.csv", index=False)
    print("\n✅ Saved predictions to emotion_predictions.csv")

# ====================
# Entry Point
# ====================
if __name__ == "__main__":
    folder = input("Enter path to folder containing .wav files: ").strip()
    if os.path.isdir(folder):
        predict_emotions(folder)
    else:
        print("❌ Invalid folder path!")
