import tensorflow as tf
import numpy as np
import pathlib
import os
import time

# ---- Reload data pipeline ----
data_dir = pathlib.Path('data/mini_speech_commands')
commands = np.array([c.name for c in sorted(data_dir.iterdir())
                      if c.is_dir() and c.name != '_background_noise_'])

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
filenames = [f for f in filenames if '_background_noise_' not in f]
filenames = tf.random.shuffle(filenames, seed=42)
num_samples = len(filenames)
test_files = filenames[int(0.9 * num_samples):]

noise_dir = pathlib.Path('data/ambient_noise')
noise_files = list(noise_dir.glob('*.wav'))

def load_noise_clips():
    clips = []
    for nf in noise_files:
        audio_binary = tf.io.read_file(str(nf))
        audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        clips.append(tf.squeeze(audio, axis=-1).numpy())
    return clips

noise_clips = load_noise_clips()

def add_noise(waveform, noise_clips, snr_db=10):
    noise = noise_clips[np.random.randint(len(noise_clips))]
    start = np.random.randint(0, len(noise) - 16000)
    noise_segment = noise[start:start + 16000]
    sig_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise_segment ** 2)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
    return waveform + noise_segment * scale

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    return file_path.split(os.sep)[-2]

def get_spectrogram(waveform):
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram[..., tf.newaxis]

def build_test_arrays(add_noise_flag=False, snr_db=10):
    X, y = [], []
    for f in test_files:
        f = f.numpy().decode('utf-8')
        label = get_label(f)
        label_id = np.argmax(commands == label)
        audio_binary = tf.io.read_file(f)
        waveform = decode_audio(audio_binary).numpy()
        if add_noise_flag:
            waveform = add_noise(waveform, noise_clips, snr_db=snr_db)
        spec = get_spectrogram(tf.constant(waveform, dtype=tf.float32)).numpy()
        X.append(spec)
        y.append(label_id)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

print("Building clean test set...")
X_clean, y_clean = build_test_arrays(add_noise_flag=False)
print("Building noisy test set (SNR=10dB)...")
X_noisy, y_noisy = build_test_arrays(add_noise_flag=True, snr_db=10)

# ---- Load the float32 preprocessing model ----
preprocess_model = tf.keras.models.load_model('preprocess_model.keras')

def evaluate_two_stage(tflite_path, X, y, n_latency_runs=100):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    latencies = []
    X_pre = preprocess_model.predict(X, verbose=0)  # batch preprocess once, float32

    for i in range(len(X)):
        x = X_pre[i:i+1].astype(input_details[0]['dtype'])

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000)

        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output[0])
        if pred == y[i]:
            correct += 1

    accuracy = correct / len(X)
    avg_latency = np.mean(latencies[-n_latency_runs:])
    return accuracy, avg_latency

print("\n=== FULL INT8 v2 (preprocessing kept float32) ===")
acc_clean, lat_clean = evaluate_two_stage('model_full_int8_v2.tflite', X_clean, y_clean)
acc_noisy, lat_noisy = evaluate_two_stage('model_full_int8_v2.tflite', X_noisy, y_noisy)
size_kb = os.path.getsize('model_full_int8_v2.tflite') / 1024
print(f"Full INT8 v2   | CNN size: {size_kb:.1f} KB | "
      f"clean acc: {acc_clean:.4f} lat: {lat_clean:.3f}ms | "
      f"noisy acc: {acc_noisy:.4f} lat: {lat_noisy:.3f}ms")
print("(Note: latency here is CNN-only inference time; add preprocessing time separately for full pipeline latency)")