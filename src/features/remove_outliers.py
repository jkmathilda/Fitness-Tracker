import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor


# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score."""

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()
    plt.xlabel("samples")
    plt.ylabel("value")

    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

def mark_outliers_iqr(dataset, col):
    """Mark values as outliers using the IQR method."""

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers using Chauvenet's criterion (vectorized)."""

    dataset = dataset.copy()
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    deviation = abs(dataset[col] - mean) / std

    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)

    # Vectorized probability computation
    prob = 1.0 - 0.5 * (scipy.special.erf(high) - scipy.special.erf(low))
    dataset[col + "_outlier"] = prob < criterion
    return dataset


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF."""

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# --------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------

if __name__ == '__main__':
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (20, 5)
    plt.rcParams['figure.dpi'] = 100

    df = pd.read_pickle('../../data/interim/01_data_processed.pkl')
    outlier_columns = list(df.columns[:6])

    # ----------------------------------------------------------
    # Apply Chauvenet's criterion per label and deal with outliers
    # ----------------------------------------------------------

    outliers_removed_df = df.copy()
    for col in outlier_columns:
        for label in df['label'].unique():
            dataset = mark_outliers_chauvenet(df[df['label'] == label], col=col)

            # Replace values marked as outliers with NaN
            dataset.loc[dataset[col + '_outlier'] == True, col] = np.nan

            # Update the column in the original dataframe
            outliers_removed_df.loc[(outliers_removed_df['label'] == label), col] = dataset[col]

            n_outliers = len(dataset) - len(dataset[col].dropna())
            print(f'Removed {n_outliers} from {col} for {label}')

    # ----------------------------------------------------------
    # Export new dataframe
    # ----------------------------------------------------------

    outliers_removed_df.to_pickle('../../data/interim/02_outliers_removed_chauvenets.pkl')
