import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

# Per-exercise rep counting parameters
REP_PARAMS = {
    'bench': {'column': 'acc_r', 'cutoff': 0.4},
    'squat': {'column': 'acc_r', 'cutoff': 0.35},
    'row':   {'column': 'gyr_x', 'cutoff': 0.65},
    'ohp':   {'column': 'acc_r', 'cutoff': 0.35},
    'dead':  {'column': 'acc_r', 'cutoff': 0.4},
}

fs = 1000 / 200
LowPass = LowPassFilter()


def count_reps(dataset, cutoff=0.4, order=10, column='acc_r'):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(data[f'{column}_lowpass'])
    plt.plot(peaks[f'{column}_lowpass'], "o", color='red')
    ax.set_ylabel(f'{column}_lowpass')
    exercise = dataset['label'].iloc[0].title()
    category = dataset['category'].iloc[0].title()
    plt.title(f'{exercise} {category}: {len(peaks)} Reps')
    plt.show()

    return len(peaks)


if __name__ == '__main__':
    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["lines.linewidth"] = 2

    df = pd.read_pickle('../../data/interim/01_data_processed.pkl')
    df = df[df['label'] != 'rest']

    acc_r = df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2
    gyr_r = df['gyr_x'] ** 2 + df['gyr_y'] ** 2 + df['gyr_z'] ** 2
    df['acc_r'] = np.sqrt(acc_r)
    df['gyr_r'] = np.sqrt(gyr_r)

    # ----------------------------------------------------------
    # Create benchmark dataframe
    # ----------------------------------------------------------

    df['reps'] = df['category'].apply(lambda x: 5 if x == 'heavy' else 10)
    rep_df = df.groupby(['label', 'category', 'set'])['reps'].max().reset_index()
    rep_df['reps_pred'] = 0

    for s in df['set'].unique():
        subset = df[df['set'] == s]
        label = subset['label'].iloc[0]

        params = REP_PARAMS.get(label, {'column': 'acc_r', 'cutoff': 0.4})
        reps = count_reps(subset, cutoff=params['cutoff'], column=params['column'])
        rep_df.loc[rep_df['set'] == s, 'reps_pred'] = reps

    # ----------------------------------------------------------
    # Evaluate the results
    # ----------------------------------------------------------

    error = np.round(mean_absolute_error(rep_df['reps'], rep_df['reps_pred']), 2)
    print(f'Mean Absolute Error: {error}')
    rep_df.groupby(['label', 'category'])[['reps', 'reps_pred']].mean().plot.bar()
    plt.show()
