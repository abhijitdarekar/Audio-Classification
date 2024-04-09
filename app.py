# import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten,Dense
from tensorflow.keras.models import Sequential
import tensorflow_io as tfio
import warnings 
warnings.filterwarnings("ignore")

import os

class CFG:
    FOREST_PATH  = os.path.join("data","Forest Recordings")
    MODEL_WEIGHTS = "data/model"

def get_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape = (1491,257,1)))
    model.add(Conv2D(16,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer = 'adam',loss = tf.keras.losses.BinaryCrossentropy(),
             metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    return model

model = get_model()

model.load_weights(CFG.MODEL_WEIGHTS)

def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

print(model.summary())