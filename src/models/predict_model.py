import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_model(model_dir='../../models/'):
    """Load saved model artifacts from train_model.py."""
    artifacts = {
        'model': joblib.load(model_dir + 'best_model_rf.pkl'),
        'scaler': joblib.load(model_dir + 'best_scaler.pkl'),
        'pca_scaler': joblib.load(model_dir + 'pca_scaler.pkl'),
        'pca': joblib.load(model_dir + 'pca_model.pkl'),
        'kmeans': joblib.load(model_dir + 'kmeans_model.pkl'),
        'feature_set': joblib.load(model_dir + 'feature_set.pkl'),
    }
    return artifacts


def predict(df, artifacts):
    """Run prediction on a feature DataFrame.

    Args:
        df: DataFrame with at minimum the 6 sensor columns plus any
            temporal/frequency features produced by build_features.py.
        artifacts: dict returned by load_model().

    Returns:
        np.ndarray of predicted exercise labels.
    """
    predictor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    cluster_columns = ['acc_x', 'acc_y', 'acc_z']

    df = df.copy()

    # Drop leaked columns if present (they will be recomputed)
    for col in [c for c in df.columns if c.startswith('pca_')] + ['cluster', 'duration']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Re-compute PCA using training-fitted transformers
    pca_input = artifacts['pca_scaler'].transform(df[predictor_columns])
    pca_result = artifacts['pca'].transform(pca_input)
    for i in range(pca_result.shape[1]):
        df[f'pca_{i+1}'] = pca_result[:, i]

    # Re-compute cluster using training-fitted KMeans
    df['cluster'] = artifacts['kmeans'].predict(df[cluster_columns])

    # Drop non-feature columns
    drop_cols = ['participant', 'category', 'set', 'label']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Select and scale features
    feature_set = artifacts['feature_set']
    X = df[feature_set]
    X_scaled = pd.DataFrame(
        artifacts['scaler'].transform(X),
        columns=feature_set,
        index=X.index,
    )

    # Predict
    predictions = artifacts['model'].predict(X_scaled)
    return predictions


if __name__ == '__main__':
    # Example: load model and predict on test data
    artifacts = load_model()

    df = pd.read_pickle('../../data/interim/03_data_features.pkl')
    predictions = predict(df, artifacts)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique labels predicted: {np.unique(predictions)}")
    print(f"Sample predictions: {predictions[:10]}")
