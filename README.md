# Comprehensive Project Report: EEG Subject Identification using Hybrid CNN-LSTM

## 1. Introduction and Problem Statement

### 1.1 Project Goal
The primary objective of this project is to implement a robust and highly accurate biometric authentication system based on resting-state Electroencephalography (EEG) signals. This involves classifying the unique brain activity patterns of individuals—a multi-class classification task designed to identify which of the 109 subjects generated a specific EEG recording.

### 1.2 Approach
To effectively capture both the spectral characteristics and temporal dynamics inherent in EEG signals, we employ a sophisticated deep learning architecture: a **Hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) Model**. A key aspect of our methodology is the transformation of raw time-series data into a **time-frequency representation (Spectrogram)** using the Short-Time Fourier Transform (STFT), which serves as the input to the model.

## 2. Dataset and Data Acquisition

### 2.1 Dataset Description
The data utilized is sourced from the **PhysioNet EEG Motor Movement/Imagery Dataset** (accessed via `mne.datasets.eegbci`). This large public dataset provides 64-channel EEG recordings collected using the standard 10-20 system.

### 2.2 Subject and Trial Selection
* **Total Subjects:** The study includes data from **109 distinct subjects** (representing 109 classes).
* **Runs Used:** To capture resting-state brain dynamics, we focused on **Run 1 (Eyes Open)** and **Run 2 (Eyes Closed)** trials.
* **Sampling Rate:** The original sampling rate is 160 Hz.

## 3. Methodology: Data Preprocessing and Feature Engineering

The `load_and_preprocess_data` pipeline is critical for transforming the raw signals into a usable format for the deep learning model.

### 3.1 Signal Filtering and Epoching
1.  **Concatenation:** Raw EDF files for the selected runs were concatenated for each subject.
2.  **Band-Pass Filtering:** To focus on the main brain rhythms (Delta, Theta, Alpha, Beta, Gamma), the signal was filtered using a band-pass filter between **1 Hz and 40 Hz**.
3.  **Epoching:** The continuous filtered data was segmented into uniform, non-overlapping **1.0-second fixed-length windows** (epochs/trials).

### 3.2 Spectrogram Feature Extraction (STFT)

Instead of using raw time-series data, which struggles to capture frequency variations effectively, we utilize the Short-Time Fourier Transform (STFT) for feature generation.

* **Process:** STFT breaks the signal into short segments and calculates the Fourier Transform for each segment, resulting in a 2D time-frequency image (spectrogram).
* **Input Data Shape:** The data is transformed from `(Trials, 64 Channels, TimeSamples)` to the final 4D input required by the CNN: `(Trials, FreqBins, TimeBins, Channels)`.
* **Final Feature Shape:** `(Trials, 33 FreqBins, 6 TimeBins, 64 Channels)`.



### 3.3 Data Split
The complete dataset of trials was split into training, validation, and testing sets to ensure robust model evaluation.

## 4. Hybrid CNN-LSTM Model Architecture

The deep learning model is specifically designed to interpret the complex 4D spectrogram features.



### 4.1 Architecture Components

| Layer (Type) | Output Shape | Parameters | Function in Model |
| :--- | :--- | :--- | :--- |
| **Conv2D (1)** | (None, 33, 6, 32) | 1,056 | Extracts local spectral/temporal features. |
| `BatchNormalization` | (None, 33, 6, 32) | 128 | Stabilizes and speeds up training. |
| `MaxPooling2D (1)` | (None, 33, 3, 32) | 0 | Downsamples along the time dimension. |
| **Conv2D (2)** | (None, 33, 3, 64) | 18,496 | Extracts more complex, abstract features. |
| `BatchNormalization` | (None, 33, 3, 64) | 256 | |
| `MaxPooling2D (2)` | (None, 33, 1, 64) | 0 | Further reduces the time dimension to 1. |
| **Reshape** | (None, 33, 64) | 0 | Prepares 2D feature map for sequential processing (`Timesteps=33 FreqBins`, `Features=64`). |
| **LSTM** | (None, 64) | 33,024 | Captures long-range dependencies across the frequency domain. |
| **Dense (Output)** | (None, 109) | 7,101 | Classification layer with **Softmax** activation for 109 classes. |

**Total Trainable Parameters:** 140,497

### 4.2 Training
The model was trained for **50 epochs** using the Adam optimizer and Sparse Categorical Crossentropy loss function.

## 5. Results and Conclusion

### 5.1 Performance Metrics

The model demonstrated exceptional performance on the validation and independent test sets, validating the effectiveness of the spectrogram-based CNN-LSTM approach.

| Metric | Training Setting | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Best Validation Accuracy** | Epoch 47 (approx.) | 93.12% | High generalization capability during training. |
| **Test Accuracy** | Final Evaluation | 92% | The final measure of accuracy on unseen data. |
| **Weighted Average F1-score** | Classification Report | 0.92 | Excellent balance between precision and recall across all 109 classes. |

### 5.2 Discussion
The test accuracy of **92%** is a significant result for a multi-class problem with 109 distinct classes, confirming that the resting-state EEG signals contain highly discriminative features for individual identification.

* The **CNN layers** successfully learned spatial and frequency patterns within the spectrogram images.
* The **LSTM layer** was crucial for modeling the sequential dependencies, likely capturing the evolution of power across different frequency bands (e.g., from Delta to Gamma).

In conclusion, this hybrid deep learning system provides a robust framework for high-accuracy EEG-based subject identification, demonstrating strong potential for real-world biometric applications.
