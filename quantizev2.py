import tensorflow as tf
import pathlib
import numpy as np
import os

# ---- Reload data pipeline ----
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

train_ds_raw = preprocess_dataset(train_files)

# ---- Load the full trained model ----
full_model = tf.keras.models.load_model('kws_model.keras')
full_model.summary()

# ---- Rebuild the graph with a fresh Input, splitting at 'normalization' ----
spec_shape = (124, 129, 1)  # our known spectrogram shape
input_full = tf.keras.Input(shape=spec_shape, name='full_input')

x = input_full
norm_output = None
post_norm_layers = []
started = False
for layer in full_model.layers:
    x = layer(x)
    if layer.name == 'normalization':
        norm_output = x
        started = True
    elif started:
        post_norm_layers.append(layer)

preprocess_model = tf.keras.Model(input_full, norm_output)
preprocess_model.save('preprocess_model.keras')
print("Saved preprocess_model.keras (Resizing + Normalization, float32)")
print("Preprocessed output shape:", preprocess_model.output_shape)

cnn_input = tf.keras.Input(shape=norm_output.shape[1:], name='cnn_input')
y = cnn_input
for layer in post_norm_layers:
    y = layer(y)
cnn_only_model = tf.keras.Model(cnn_input, y)
cnn_only_model.summary()

# ---- Representative dataset: preprocessed (resized+normalized) spectrograms ----
def representative_data_gen():
    for spec, _ in train_ds_raw.shuffle(1000, seed=42).batch(1).take(500):
        preprocessed = preprocess_model(spec)
        yield [preprocessed]

# ---- Quantize only the CNN portion to full int8 ----
converter = tf.lite.TFLiteConverter.from_keras_model(cnn_only_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_cnn_int8 = converter.convert()
with open('model_full_int8_v2.tflite', 'wb') as f:
    f.write(tflite_cnn_int8)
print("Saved model_full_int8_v2.tflite — size (KB):", len(tflite_cnn_int8) / 1024)