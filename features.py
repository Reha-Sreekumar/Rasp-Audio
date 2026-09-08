import tensorflow as tf
import pathlib
import numpy as np

data_dir = pathlib.Path('data/mini_speech_commands')
commands = np.array([c.name for c in sorted(data_dir.iterdir())
                      if c.is_dir() and c.name != '_background_noise_'])
print("Commands:", commands)

# 1. Build a list of all file paths + labels
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
filenames = [f for f in filenames if '_background_noise_' not in f]
filenames = tf.random.shuffle(filenames, seed=42)
print("Total files:", len(filenames))

# 2. Decode a wav file into a waveform tensor
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)
    return tf.squeeze(audio, axis=-1)  # shape: (16000,)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.sep)
    return parts[-2]  # folder name = label

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

# 3. Convert waveform -> log-mel spectrogram
def get_spectrogram(waveform):
    # pad to exactly 16000 samples (1 second at 16kHz)
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]  # add channel dim for CNN
    return spectrogram

# quick sanity test on one file
import os
test_file = filenames[0].numpy().decode('utf-8')
waveform, label = get_waveform_and_label(test_file)
spectrogram = get_spectrogram(waveform)
print("Label:", label.numpy())
print("Waveform shape:", waveform.shape)
print("Spectrogram shape:", spectrogram.shape)