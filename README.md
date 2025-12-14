# Person Identification from EEG Signals using Hybrid CNN + LSTM

**Architecture:** Hybrid CRNN (Convolutional Recurrent Neural Network)

## 📌 Project Overview

The core objective of this project is to develop a highly accurate, deep learning-based **biometric identification system** that distinguishes between individuals using their unique **Electroencephalogram (EEG)** brainwave patterns.

This task is formulated as a multi-class classification problem involving **109 distinct subjects**. The system leverages a **Hybrid CNN-LSTM Model** to process time-frequency features (spectrograms) and capture the complex spatial, spectral, and temporal characteristics of the neural signals.

## 🧠 Dataset

We utilized the **PhysioNet EEG Motor Movement/Imagery Dataset**.

* **Focus:** Resting State recordings. We specifically used **Run 1 (Eyes Open)** and **Run 2 (Eyes Closed)** trials.
* **Scale:** The model was trained and tested on the **full dataset of 109 subjects**.
* **Channels:** Data includes 64 EEG channels.
* **Challenge:** The system must identify a "neural fingerprint" among a large population (109 classes), requiring a model capable of high discriminative power.

## ⚙️ Methodology

### 1. Preprocessing Pipeline

A memory-efficient and signal-processing rigorous pipeline was implemented to prepare the data for the deep learning model:

* **Filtering:** A bandpass filter (FIR design) was applied from **1 Hz to 40 Hz** to isolate meaningful brain activity bands (Delta, Theta, Alpha, Beta) and remove DC offset and high-frequency noise.
* **Segmentation:** Continuous signals were segmented into **1.0-second fixed-length, non-overlapping epochs** to generate individual training trials.
* **Feature Engineering: STFT (Short-Time Fourier Transform):** The key step involves converting the 1D time-series epochs into **Spectrograms** (time-frequency images). This allows the CNN to extract spectral features across time.
    * *STFT Parameters:* Sample rate (`fs`) of 160 Hz and a window size (`nperseg`) of 64.
* **Final Input Shape:** The 4D input tensor shape is `(Trials, 33 FreqBins, 6 TimeBins, 64 Channels)`.
* **Optimization:** Data was cast to `np.float32` to optimize memory usage for handling the large dataset.

### 2. Model Architecture: Hybrid CNN-LSTM

The model is structured to sequentially process the spatial/spectral data using CNNs, followed by temporal analysis using an LSTM layer.

| Layer (Type) | Output Shape | Parameters | Primary Function |
| :--- | :--- | :--- | :--- |
| `Input` | (None, 33, 6, 64) | 0 | Spectrogram input. |
| **Conv2D (1)** | (None, 33, 6, 32) | 18,464 | Extracts initial spectral and spatial features. |
| `BatchNormalization` | (None, 33, 6, 32) | 128 | Stabilizes feature distribution. |
| `MaxPooling2D` | (None, 33, 3, 32) | 0 | Downsamples time dimension by half. |
| **Conv2D (2)** | (None, 33, 3, 64) | 18,496 | Extracts higher-level features. |
| `BatchNormalization` | (None, 33, 3, 64) | 256 | |
| `MaxPooling2D` | (None, 33, 1, 64) | 0 | Compresses time dimension further. |
| **Reshape** | (None, 33, 64) | 0 | Transforms CNN output for LSTM: (`TimeSteps`, `Features`). |
| **LSTM** | (None, 64) | 33,024 | Recurrently processes features across frequency steps (TimeSteps=33). |
| **Dense (Output)** | (None, 109) | 7,101 | Final classification layer with Softmax activation. |

* **Total Trainable Parameters:** 140,497.
* **Training:** The model was compiled and trained for **50 epochs**.

## 📊 Results

The model achieved strong performance on the multi-class classification task, demonstrating the feasibility of using spectral-temporal features for biometric EEG identification.

| Metric | Score | Context |
| :--- | :--- | :--- |
| **Best Validation Accuracy** | **93.12%** | Achieved during training (Epoch 47). |
| **Weighted Average F1-score** | **0.92** | Indicating high balanced performance across all 109 classes. |
| **Test Accuracy (Inferred)** | **~92%** | Consistent with the high F1-score and validation accuracy. |

The high accuracy confirms that the hybrid CNN-LSTM architecture is highly effective at extracting the unique "neural fingerprint" from resting-state EEG spectrograms, providing a viable and scalable approach to biometric authentication.

## 🛠️ Dependencies

To run this project, you need the following Python libraries:

```bash
pip install mne tensorflow scikit-learn matplotlib seaborn numpy scipy
