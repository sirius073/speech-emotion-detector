# 🎧 Speech Emotion Recognition

This project is a deep learning-based system for recognizing **human emotions from audio** using a hybrid CNN-LSTM-Attention model. It supports:

- ✅ **Command-line interface (`testing.py`)**: Predict emotions for all `.wav` files in a folder.
- ✅ **Streamlit web app (`app.py`)**: Upload and predict emotion from a single audio file interactively.

---

## 📌 Project Description

The model classifies `.wav` audio recordings into 8 emotion categories:

- `neutral`
- `calm`
- `happy`
- `sad`
- `angry`
- `fear`
- `disgust`
- `surprise`

The project uses **Librosa** for audio feature extraction and **PyTorch** for deep learning.

---

## 🛠️ Preprocessing Methodology

Each audio file is processed using the following steps:

1. **Load audio** (duration = 2.5s, offset = 0.6s)
2. **Extract features**:
   - Zero Crossing Rate (ZCR)
   - Chroma STFT
   - MFCC
   - RMS Energy
   - Mel Spectrogram
3. **Aggregate** features using mean across time.
4. **Standardize** using `StandardScaler` (scikit-learn).
5. **Reshape** to `[1, 1, 9, 18]` for CNN input.

---

## 🧠 Model Architecture

The model (`EnhancedCNNLSTM`) combines:

- **Multi-scale CNNs** with residual connections:
  - 1×1, 3×3, and 5×5 filters
- **Bidirectional LSTM**:
  - 2 layers, 96 hidden units per direction
- **Multi-head Attention**:
  - 8 heads on LSTM outputs
- **Statistical Pooling**:
  - Mean + Standard Deviation across time
- **Fully Connected Layers**:
  - 256 → 128 → Output (with Dropout & BatchNorm)

---

## 📈 Accuracy Metrics

| Metric        | Value    |
|---------------|----------|
| Accuracy      | **92.6%** |
| F1 Score      | 91.8%    |
| Precision     | 92.4%    |
| Recall        | 92.1%    |

> Evaluation was done on a held-out validation split from the RAVDESS dataset.

---

## 🚀 How to Use

### 🔹 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 🔹 2. Run testing.py for Batch Inference
This script runs predictions on all .wav files in a folder.

```bash
python testing.py
```
✅ You'll be prompted to enter a folder path.

📁 Example folder structure:
audio_folder/
├── sample1.wav
├── sample2.wav
└── ...

📄 Output:

Predictions are printed in the terminal.
Also saved to emotion_predictions.csv.

### 🔹 3. Run app.py
You can run the Streamlit app to upload and predict a single audio file interactively.
```bash
streamlit run app.py
```
🖼️ The app supports:

File upload (.wav)
Real-time prediction
Emotion display and playback


### 💡 Future Improvements
🎙️ Add microphone-based live recording

📈 Add visualizations (e.g., spectrogram, attention maps)

🌐 Deploy app with Hugging Face Spaces or Streamlit Cloud


Made with 💙 for speech understanding & deep learning.
