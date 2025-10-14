!pip install -q torch librosa transformers soundfile

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import IPython.display as ipd

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    return y, sr

model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def transcribe_audio(file_path):
    y, sr = preprocess_audio(file_path)
    input_values = tokenizer(y, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    return transcription[0]

from google.colab import files
uploaded = files.upload()

import os
file_path = list(uploaded.keys())[0]

ipd.Audio(file_path)

transcription = transcribe_audio(file_path)
print("Transcription:", transcription)
