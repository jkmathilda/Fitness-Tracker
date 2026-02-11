# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML pipeline for exercise recognition and repetition counting using wearable sensor data (accelerometer + gyroscope) from MetaWear devices. Classifies 5 exercises (bench press, squat, barbell row, overhead press, deadlift) and counts reps using signal processing. This is a research/analysis project with script-based execution, not a packaged application.

## Environment Setup

```bash
conda env create -f environment.yml       # Create env (Python 3.8.15)
conda activate tracking-barbell-exercises
conda env update --file environment.yml --prune  # Update existing env
```

Key dependencies: numpy 1.23.5, pandas 1.5.2, scikit-learn, scipy, matplotlib 3.6.2.

## Running the Pipeline

Scripts must be executed sequentially. Each stage reads/writes pickle files in `data/interim/`.

1. **`src/data/make_dataset.py`** — Load raw CSVs from `data/raw/MetaMotion/`, extract metadata from filenames (participant, exercise, set), merge accelerometer + gyroscope data, resample to 200ms windows. Outputs `01_data_processed.pkl`.

2. **`src/features/remove_outliers.py`** — Detect and remove outliers using Chauvenet's criterion and LocalOutlierFactor. Outputs `02_outliers_removed_chauvenets.pkl`.

3. **`src/features/build_features.py`** — Feature engineering: Butterworth low-pass filtering, PCA (3 components), magnitude features (acc_r, gyr_r), temporal features (rolling mean/std), frequency features (FFT, power spectral entropy, weighted frequency), K-means clustering (5 clusters). Outputs `03_data_features.pkl` with 100+ features.

4. **`src/models/train_model.py`** — Train and evaluate 5 classifiers (Random Forest, Neural Network, kNN, Decision Tree, Naive Bayes) across 4 progressive feature sets. Includes forward feature selection, grid search, stratified splits, and participant-based generalization testing. Best model: Random Forest with Feature Set 4.

5. **`src/features/count_repetitions.py`** — Exercise-specific repetition counting via low-pass filtering and peak detection. Uses hardcoded per-exercise parameters (e.g., squat cutoff: 0.35, bench: 0.4, row uses gyr_x at 0.65).

Scripts use relative imports and paths — run them from their containing directory or adjust paths accordingly.

## Architecture

```
Raw CSVs → make_dataset.py → [01_data_processed.pkl]
  → remove_outliers.py → [02_outliers_removed_chauvenets.pkl]
  → build_features.py → [03_data_features.pkl]
  → train_model.py → model evaluation + figures
  → count_repetitions.py → rep counts + MAE evaluation
```

**Utility classes** in `src/features/`:
- `DataTransformation.py` — `LowPassFilter`, `PrincipalComponentAnalysis`
- `TemporalAbstraction.py` — `NumericalAbstraction` (rolling window stats)
- `FrequencyAbstraction.py` — `FourierTransformation` (FFT, PSE, weighted freq)

**ML wrapper** in `src/models/LearningAlgorithms.py` — `ClassificationAlgorithms` class providing a unified interface to all sklearn classifiers with forward feature selection.

## Sensor Data Format

Raw data: CSV files in `data/raw/MetaMotion/` with filenames encoding `participant-exercise-set-category.csv`. Six sensor columns: `acc_x`, `acc_y`, `acc_z` (accelerometer) and `gyr_x`, `gyr_y`, `gyr_z` (gyroscope). Participants labeled A–E, exercises include bench, squat, row, ohp, dead, with medium/heavy categories.

## Notes

- No test suite, linting, CI/CD, or build system exists.
- Visualization config in `src/visualization/plot_settings.py` (seaborn style, A4 figure size).
- `src/models/predict_model.py` is a placeholder (empty).
- Reports and figures go to `reports/figures/`.
