# Speech Processing: Emotion Recognition & Speech-to-Text

Two speech processing pipelines built on the **RAVDESS** emotional speech dataset: automatic speech transcription using a pretrained transformer, and speech emotion recognition from acoustic features.

## 1. Speech-to-Text (`speech-to-text/`)
Transcribes speech audio using **Wav2Vec2**, a pretrained transformer-based ASR (Automatic Speech Recognition) model from Hugging Face.
- Loads `facebook/wav2vec2-base-960h`
- Processes `.wav` files from the RAVDESS dataset
- Outputs transcriptions to `transcriptions.csv`

## 2. Speech Emotion Recognition (`speech-emotion-recognition/`)
Classifies the emotional tone of speech from extracted acoustic features.
- Extracts **MFCC**, **pitch**, and **energy** features from each audio sample using `librosa`
- Trains a **Support Vector Machine (SVM)** classifier on the extracted features
- Evaluates with a classification report (precision/recall/F1 per emotion class)

## Dataset
[RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/records/1188976) — 24 professional actors performing speech with 8 emotions, used here for both transcription and emotion classification.

## Tech Stack
- Python
- Hugging Face `transformers` (Wav2Vec2)
- `librosa` (audio feature extraction)
- Scikit-learn (SVM classifier)
- PyTorch

## How to Run
```bash
pip install transformers datasets torchaudio librosa scikit-learn
jupyter notebook speech-to-text/speech_to_text.ipynb
jupyter notebook speech-emotion-recognition/speech_project.ipynb
```
> Note: notebooks were originally run on Google Colab with the dataset stored on Google Drive — update the file paths to match your local/Colab environment.

## Roadmap
- [ ] Report Word Error Rate (WER) for the speech-to-text pipeline
- [ ] Compare SVM against a neural network classifier for emotion recognition
- [ ] Add a confusion matrix visualization for the emotion classifier
