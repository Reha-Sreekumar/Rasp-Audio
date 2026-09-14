# Voice Keyword-Spotting System (Rasp-Audio)

A voice keyword-spotting (KWS) system that recognizes 8 spoken commands, trained in TensorFlow/Keras, quantized with LiteRT (TensorFlow Lite), and deployed to a Raspberry Pi with live microphone inference.

## Overview

- **Task:** Recognize 8 spoken commands from short audio clips.
- **Commands:** `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes` (the Mini Speech Commands vocabulary).
- **Dataset:** [Mini Speech Commands](https://www.tensorflow.org/tutorials/audio/simple_audio) (8,000 one-second `.wav` clips, 1,000 per command).
- **Feature representation:** Log-magnitude STFT spectrogram (124 × 129 × 1).
- **Model:** Compact 2D CNN (~1.6M parameters), matching the TensorFlow "Simple Audio Recognition" tutorial architecture.
- **Deployment toolkit:** Google LiteRT (TensorFlow Lite), chosen for native TensorFlow interoperability and direct support on Raspberry Pi (aarch64).
- **Deployment target:** Raspberry Pi (Debian 13 "Trixie", aarch64), via [`ai-edge-litert`](https://pypi.org/project/ai-edge-litert/).

## Repository Structure

```
Rasp-Audio/
├── data/
│   ├── mini_speech_commands/     # 8 command folders of .wav clips
│   └── ambient_noise/            # Recorded real ambient noise (chatter, fan, keyboard)
├── data.py                       # Dataset download / folder verification
├── features.py                   # Waveform -> spectrogram feature extraction
├── pipeline.py                   # Full tf.data pipeline + train/val/test split
├── quantize.py                   # Converts trained model to float32 / dynamic-INT8 / full-INT8 .tflite
├── quantizev2.py                 # Full-INT8 variant with Resizing/Normalization isolated (diagnostic)
├── evaluate.py                   # Accuracy/latency evaluation, clean vs. noisy, all 3 quantization variants
├── evaluatev2.py                 # Evaluation for the isolated full-INT8 variant
├── kws_model.keras               # Trained Keras model (float32)
├── preprocess_model.keras        # Extracted Resizing+Normalization sub-model (used in quantizev2 diagnostics)
├── model_float32.tflite          # Baseline, unquantized
├── model_dynamic_int8.tflite     # ★ Deployed model — dynamic-range INT8 quantization
├── model_full_int8.tflite        # Full-integer INT8 (evaluated, not deployed — see Results)
├── model_full_int8_v2.tflite     # Full-integer INT8, preprocessing isolated (evaluated, not deployed)
├── pi_infer.py                   # On-device single-file inference (Raspberry Pi, NumPy-based feature extraction)
├── pi_batch_eval.py               # On-device batch accuracy/latency evaluation (Raspberry Pi)
├── pi_noisy_eval.py               # On-device clean vs. real-noise evaluation (Raspberry Pi)
└── test.py                       # Misc. debugging/sanity-check script
```

## Setup

### Development machine (training / quantization)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install tensorflow numpy matplotlib seaborn --break-system-packages
```

### Raspberry Pi (inference only)

TensorFlow is **not** installed on the Pi — only the lightweight LiteRT runtime, since full TensorFlow is unnecessarily heavy for pure inference:

```bash
python3 -m venv ~/kws_env
source ~/kws_env/bin/activate
pip install ai-edge-litert numpy
```

## Usage

### 1. Prepare data and train (development machine)

```bash
python data.py          # Download and verify dataset
python pipeline.py       # Build the tf.data pipeline (sanity check)
# training is run via the model section embedded in pipeline.py / quantize.py workflow
```

### 2. Quantize

```bash
python quantize.py
```
Produces `model_float32.tflite`, `model_dynamic_int8.tflite`, and `model_full_int8.tflite`.

### 3. Evaluate (clean + real ambient noise)

```bash
python evaluate.py
```

### 4. Deploy to Raspberry Pi

Model and code were transferred via a private GitHub repository (direct SSH/SCP was blocked by client isolation on the local network):

```bash
# On the Pi
git clone https://github.com/<your-username>/Rasp-Audio.git
cd Rasp-Audio
source ~/kws_env/bin/activate
```

### 5. Run on-device inference

**Single file:**
```bash
python pi_infer.py data/mini_speech_commands/yes/004ae714_nohash_0.wav
```

**Batch accuracy/latency:**
```bash
python pi_batch_eval.py
```

**Clean vs. real ambient noise:**
```bash
python pi_noisy_eval.py
```

**Live microphone:**
```bash
arecord -D plughw:<card>,0 -f S16_LE -r 16000 -c 1 -d 2 test_recording.wav
python pi_infer.py test_recording.wav
```
(Use `arecord -l` to find your USB microphone's card number.)

## Results

### Quantization comparison (development machine, 800-file test set)

| Model | Size | Clean Acc. | Clean Latency | Noisy Acc. (10dB) | Noisy Latency |
|---|---|---|---|---|---|
| Float32 | 6,354 KB | 92.87% | 0.318 ms | 77.88% | 0.316 ms |
| **Dynamic-range INT8** (deployed) | **1,596 KB** | **92.63%** | **0.124 ms** | **77.50%** | **0.102 ms** |
| Full-integer INT8 | 1,599 KB | 68.25% | 0.120 ms | 61.25% | 0.122 ms |

**Dynamic-range INT8 was selected for deployment.** Full-integer INT8 was investigated (increased calibration samples, isolated Resizing/Normalization from the quantized graph) but consistently underperformed, traced to the model's Dense(128) layer — which holds 98.6% of all parameters — being highly sensitive to 8-bit activation quantization. Full-integer quantization also gave no latency advantage over dynamic-range on this hardware.

### On-device results (Raspberry Pi, 120-file balanced test set)

| Condition | Accuracy | Avg. Latency |
|---|---|---|
| Clean | 90.83% | ~2.7 ms |
| Noisy (10dB, real recorded ambient noise) | 75.83% | ~2.7 ms |

On-device accuracy tracked development-machine results within ~2 percentage points, confirming the NumPy-based feature extraction reimplemented for the Pi (no TensorFlow available on-device) faithfully reproduces the training-time pipeline. Latency is ~20× higher than on the development machine, expected given the Pi's weaker CPU and a pure-Python FFT loop rather than TensorFlow's optimized implementation — still comfortably real-time (~2.7ms per inference).

## Notes / Known Deviations

- The Mini Speech Commands dataset does not ship with a background-noise folder (unlike the full 105,000-file Speech Commands dataset), so **real ambient noise was independently recorded** (background chatter, fan hum, keyboard typing) rather than using synthetic noise, to satisfy the "real ambient noise conditions" requirement literally.
- On-device feature extraction on the Pi uses a manual NumPy FFT implementation (`ai-edge-litert` has no TensorFlow signal-processing ops available), with FFT length explicitly set to 256 to match `tf.signal.stft`'s internal behavior of rounding the 255-sample frame length up to the next power of two.
- Live-microphone recordings use a loudest-1-second-window selection (rather than assuming the spoken word starts at t=0) to handle natural pauses before speaking.
