# PhysioNet-CNN-RNN
Train a CNN + RNN hybrid model to classify which subject (1–109) a given EEG segment belongs to, based on brainwave patterns
# EEG Subject Identification using Hybrid CNN-LSTM

## Project Overview

This project implements a machine learning pipeline for **subject identification** (biometric authentication) using Electroencephalography (EEG) data. The model is a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture, designed to classify resting-state brain activity across a large cohort of individuals.

The key approach involves transforming the raw time-series EEG data into a **time-frequency representation (spectrogram)** before feeding it into the deep learning model.

## Features and Methodology

* **Task:** Multi-class classification (109 classes/subjects).
* **Data Source:** PhysioNet EEG Motor Movement/Imagery Dataset (accessed via `mne.datasets.eegbci`).
* **Data Used:** Resting-state activity from **109 subjects**, specifically Runs 1 (Eyes Open) and 2 (Eyes Closed).
* **Feature Engineering:** Short-Time Fourier Transform (STFT) is applied to convert the time-domain signal into a **spectrogram** feature matrix with shape `(Trials, FreqBins, TimeBins, Channels)`.
* **Model Architecture:** A powerful hybrid CNN-LSTM model is used, leveraging CNN layers for spatial/frequency feature extraction and LSTM layers for capturing temporal dependencies.

## Setup and Requirements

The project uses several specialized Python libraries for signal processing and deep learning.

### Prerequisites

You need a Python environment (3.x) and the following packages:

```bash
pip install numpy matplotlib seaborn mne tensorflow scikit-learn
