import tensorflow as tf
import pathlib
import numpy as np
import os

# ---- Reload everything needed ----
data_dir = pathlib.Path('data/mini_speech_commands')
commands = np.array([c.name for c in sorted(data_dir.iterdir())
                      if c.is_dir() and c.name != '_background_noise_'])

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
filenames = [f for f in filenames if '_background_noise_' not in f]
filenames = tf.random.shuffle(filenames, seed=42)
num_samples = len(filenames)
train_files = filenames[:int(0.8 * num_samples)]

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds

# Representative dataset generator — FIXED: more samples, shuffled for better coverage
train_ds_raw = preprocess_dataset(train_files)

def representative_data_gen():
    for spec, _ in train_ds_raw.shuffle(1000, seed=42).batch(1).take(500):
        yield [spec]

# ---- Load trained model ----
model = tf.keras.models.load_model('kws_model.keras')

# 1. Float32 TFLite (no quantization, baseline)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()
with open('model_float32.tflite', 'wb') as f:
    f.write(tflite_float_model)
print("Saved model_float32.tflite — size (KB):", len(tflite_float_model) / 1024)

# 2. Dynamic-range INT8 (weights only — simplest quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic_model = converter.convert()
with open('model_dynamic_int8.tflite', 'wb') as f:
    f.write(tflite_dynamic_model)
print("Saved model_dynamic_int8.tflite — size (KB):", len(tflite_dynamic_model) / 1024)

# 3. Full-integer INT8 (weights + activations, needs representative dataset)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_full_int8_model = converter.convert()
with open('model_full_int8.tflite', 'wb') as f:
    f.write(tflite_full_int8_model)
print("Saved model_full_int8.tflite — size (KB):", len(tflite_full_int8_model) / 1024)