import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
import matplotlib.pyplot as plt
import seaborn as sns

sample_file = "/Users/kde/Documents/Music Genre Classification/audio_files/PinkPanther30.wav"

# Displaying the sample file in form of waveshow (by librosa)
x, sr = librosa.load(sample_file, sr=44100)
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(x, sr=sr)
# plt.show()

def plot_melspectrogram(y,sr):
    # Compute the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # Convert to decibels (log scale)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    # Visualize the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()

def plot_melspectrogram_chunks(y,sr):
    # units are in seconds
    chunk_duration = 4
    overlap_duration = 2

    # getting sample numbers
    chunk_samples = chunk_duration * sr
    overlap_samples = overlap_duration * sr

    # calculating  number of chunks
    chunk_nums = int (np.ceil((len(y)-chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range (chunk_nums):
        start = i*(chunk_samples - overlap_samples)
        end = start + chunk_samples

        chunk = y[start:end]

        melspectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr)
        spectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()

plot_melspectrogram_chunks(y=x,sr=sr)
plot_melspectrogram(y=x,sr=sr)
