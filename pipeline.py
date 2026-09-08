import tensorflow as tf
import pathlib
import numpy as np
import os

data_dir = pathlib.Path('data/mini_speech_commands')
commands = np.array([c.name for c in sorted(data_dir.iterdir())
                      if c.is_dir() and c.name != '_background_noise_'])
print("Commands:", commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
filenames = [f for f in filenames if '_background_noise_' not in f]
filenames = tf.random.shuffle(filenames, seed=42)
num_samples = len(filenames)
print("Total files:", num_samples)

# 80/10/10 split
train_files = filenames[:int(0.8 * num_samples)]
val_files = filenames[int(0.8 * num_samples): int(0.9 * num_samples)]
test_files = filenames[int(0.9 * num_samples):]
print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")

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

train_ds = preprocess_dataset(train_files)
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# sanity check
for spec, label in train_ds.take(1):
    print("Batch spectrogram shape:", spec.shape)
    print("Batch label shape:", label.shape)
    print("Sample label ids:", label.numpy()[:10])
    # ---- MODEL (matches TF tutorial architecture) ----
for spec, _ in train_ds.take(1):
    input_shape = spec.shape[1:]
print("Input shape:", input_shape)
num_labels = len(commands)

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(data=train_ds.map(lambda spec, label: spec))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Resizing(32, 32),
    norm_layer,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

model.save('kws_model.keras')
print("Model saved as kws_model.keras")

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy (float32 model): {test_acc:.4f}")