# Fitness Tracker: Exercise Recognition and Repetition Counting Using Wearable Sensors

This project implements a pipeline for analyzing wearable sensor data collected during strength training exercises. Accelerometer and gyroscope signals are processed to extract time- and frequency-domain features, train supervised machine learning models to classify exercises, and count repetitions using signal-processing techniques.

The project focuses on feature engineering, model comparison, and exercise-specific repetition counting.

## Supported Exercises

- Bench Press  
- Squat  
- Barbell Row  
- Overhead Press (OHP)  
- Deadlift  

## Data Processing Pipeline

### 1. Data Loading and Cleaning
- Load tri-axial accelerometer and gyroscope data
- Remove non-exercise segments (e.g. rest)
- Group recordings by exercise set and participant

### 2. Signal Transformation
- Compute vector magnitudes for accelerometer and gyroscope signals
- Reduce directional dependence while preserving motion intensity

### 3. Windowing
- Segment continuous sensor data into fixed-length windows
- Treat each window as one observation for feature extraction

### 4. Feature Engineering

For each window, the following features are extracted:

- Raw sensor features
- Magnitude features (`acc_r`, `gyr_r`)
- Time-domain statistical features
- Frequency-domain features (FFT-based)
- Power Spectral Entropy (PSE)
- Principal Component Analysis (PCA) features
- Clustering-based features

The result is a structured, ML-ready feature table.

## Exercise Classification

### Task
Multi-class supervised classification: engineered motion features → exercise label

### Models
- Random Forest
- Feedforward Neural Network
- k-Nearest Neighbors
- Decision Tree
- Naive Bayes

### Evaluation
- Stratified train/test splits
- Forward feature selection
- Hyperparameter tuning via grid search
- Confusion matrices
- Participant-based generalization testing

## Repetition Counting

Repetition counting is implemented using deterministic signal-processing methods:

1. Select an exercise-specific sensor signal
2. Apply a low-pass filter
3. Detect local maxima via peak detection
4. Count peaks as repetitions

Different exercises use different signals and filter parameters.

## Combined Inference Logic

Sensor data  
↓  
Feature extraction  
↓  
Exercise classification (ML)  
↓  
Exercise-specific repetition counting


## Evaluation Metrics

- Classification accuracy
- Confusion matrices
- Mean Absolute Error (MAE) for repetition counting

## Some of the results...
<img width="473" height="420" alt="Image" src="https://github.com/user-attachments/assets/19553fc4-969d-4f0e-93d1-4f2fa162a214" />

## Acknowledgements

This repository follows and reproduces a YouTube tutorial on wearable sensor data analysis for fitness applications. 
