# Fitness Tracker: Exercise Recognition and Repetition Counting Using Wearable Sensors

This project implements a pipeline for analyzing wearable sensor data collected during strength training exercises. Accelerometer and gyroscope signals are processed to extract time- and frequency-domain features, train supervised machine learning models to classify exercises, and count repetitions using signal-processing techniques.

The project focuses on feature engineering, model comparison, and exercise-specific repetition counting.

## Supported Exercises

- Bench Press  
- Squat  
- Barbell Row  
- Overhead Press (OHP)  
- Deadlift  

## Repository Structure

├── data/  
│   ├── raw/            Raw sensor recordings  
│   ├── interim/        Cleaned and segmented data  
│   └── processed/     Feature-engineered datasets  
├── src/  
│   ├── DataTransformation/  
│   │   └── LowPassFilter.py  
│   ├── FeatureEngineering/  
│   │   └── FeatureBuilder.py  
│   ├── LearningAlgorithms/  
│   │   └── ClassificationAlgorithms.py  
│   └── analysis scripts  
├── notebooks/          Exploratory analysis  
├── requirements.txt  
└── README.md  

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
