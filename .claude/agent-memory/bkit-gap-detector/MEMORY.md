# Gap Detector Memory - Fitness-Tracker

## Project Context
- ML pipeline for exercise classification (not a web app)
- Python 3.8.15, scikit-learn, pandas, scipy
- No formal design docs; analysis is code-review-findings vs implementation
- Pipeline: make_dataset -> remove_outliers -> build_features -> train_model -> predict_model

## Key Findings (2026-02-11)

### pipeline-refactor analysis: 95% match rate (11/11 implemented, 1 latent bug)
- All 11 items implemented across 3 priority tiers
- Latent bug: train_model.py:307 saves `learner` (ClassificationAlgorithms wrapper) not the fitted RF model; predict_model.py:66 calls `.predict()` which doesn't exist on that class
- build_features.py still fits PCA/KMeans on full dataset (train_model.py correctly drops and re-fits, but intermediate pickle has leaked features)

### File Structure
- Pipeline scripts: src/data/, src/features/, src/models/
- Utility classes: DataTransformation.py, TemporalAbstraction.py, FrequencyAbstraction.py
- ML wrapper: LearningAlgorithms.py (ClassificationAlgorithms class)
- Intermediate data: data/interim/*.pkl
- Models: models/*.pkl (6 artifacts)
