# Speech AI Suite

A production-ready Flask web application for multi-task speech analysis (Emotion, Gender, Intent, Speaker) built on state-of-the-art self-supervised speech embeddings (WavLM/HuBERT/XLSR) and lightweight classifiers. 

## 🎯 Features

This application allows users to upload audio files (WAV, MP3, FLAC, OGG, M4A, WebM) or record directly from their microphone to perform inference across four specialized tasks:

1. **Emotion Recognition**
   - **Performance:** ~79.14% Accuracy on CREMA-D / IEMOCAP
   - **Model:** HuBERT-large embeddings + Support Vector Machine (SVM)
   - **Classes:** Neutral, Happy, Sad, Angry
2. **Gender Identification** 
   - **Model:** WavLM-base-plus embeddings -> PCA -> Logistic Regression
   - **Dataset:** LibriSpeech (dev-clean)
   - **Classes:** Male, Female
3. **Intent Identification**
   - **Model:** WavLM-base-plus embeddings -> PCA -> SVM Classifier
   - **Dataset:** SLURP (Spoken Language Understanding Researchers' Platform) 
   - **Classes:** 12 specific user commands (e.g., weather_query, music_query, alarm_set)
4. **Speaker Identification**
   - **Model:** XLSR-53 (Wav2Vec2 Large) embeddings -> PCA -> Logistic Regression
   - **Dataset:** Custom 40-speaker dataset
   - **Features:** Custom pooling (mean + std from layers 6-13)

## 📁 Project Architecture

The project follows a clean, MERN-like structure for the backend services:

```text
Speech-AI-Suite/
├── backend/                          
│   ├── app/                          # Flask web application
│   │   ├── app.py                    # Main Flask entry point
│   │   ├── templates/                # HTML templates for UI
│   │   ├── static/                   # CSS and JavaScript
│   │   └── uploads/                  # Temporary file processing
│   │
│   ├── services/                     # Inference services
│   │   ├── emotion.py                # Emotion classification service
│   │   ├── gender.py                 # Gender classification service
│   │   ├── intent.py                 # Intent classification service
│   │   ├── speaker.py                # Speaker identification service
│   │   └── utils/audio.py            # Audio processing utilities
│   │
│   └── config.py                     # Centralized configuration
│
├── ml_models/                        # Model training and feature extraction
│   ├── models/                       # Stored .pkl, .pt, and scaler artifacts
│   ├── data/                         # Datasets (IEMOCAP, CREMA-D, SLURP)
│   ├── src/                          # Feature extraction and training scripts
│   └── scripts/                      
│
├── docs/                             # Additional specialized documentation
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system (essential for processing user-uploaded audio like MP3s)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RoMyth682/Speech-AI-Suite.git
   cd Speech-AI-Suite
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: First-time inference will download large embedding models like WavLM-base-plus, HuBERT-large, and XLSR-53 from HuggingFace.*

### Running the Web Application

To start the Flask backend, simply run the `app.py` script:

```bash
python backend/app/app.py
```

The application will start on `http://localhost:5000`. Navigate there in your web browser to test the models using the interactive UI.

## 👥 Team

This project is developed by an AI/ML research team focused on advanced speech analysis:

- **Inthiyaz** - Emotion Recognition
- **Sahasra** - Intent Classification
- **Rohin** - Gender Identification
- **Romith** - Speaker Identification

## 📝 Citation

If you use this underlying embedding extraction code in your research, please refer to the inspiration paper:

```bibtex
@article{speechembeddings2024,
  title={From Raw Speech to Fixed Representations: A Comprehensive Evaluation of Speech Embedding Techniques},
  journal={IEEE/ACM Transactions},
  year={2024}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
