import pandas as pd
from glob import glob
import re


# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------

data_path = '../../data/raw/MetaMotion/'


# --------------------------------------------------------------
# Read all files into DataFrames
# --------------------------------------------------------------

def read_data_from_files(files):
    acc_dfs = []
    gyr_dfs = []

    acc_set = 1
    gyr_set = 1

    for f in sorted(files):  # sorted for reproducible set numbering
        participant = f.split('-')[0].replace(data_path, '')
        label = f.split('-')[1]
        category = re.split(r'\d+', f.split('-')[2])[0]

        df = pd.read_csv(f)

        df['participant'] = participant
        df['label'] = label
        df['category'] = category

        if 'Accelerometer' in f:
            df['set'] = acc_set
            acc_set += 1
            acc_dfs.append(df)

        if 'Gyroscope' in f:
            df['set'] = gyr_set
            gyr_set += 1
            gyr_dfs.append(df)

    acc_df = pd.concat(acc_dfs, ignore_index=True)
    gyr_df = pd.concat(gyr_dfs, ignore_index=True)

    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']
    del acc_df['elapsed (s)']

    del gyr_df['epoch (ms)']
    del gyr_df['time (01:00)']
    del gyr_df['elapsed (s)']

    return acc_df, gyr_df


if __name__ == '__main__':
    files = glob(data_path + '*.csv')
    acc_df, gyr_df = read_data_from_files(files)

    # --------------------------------------------------------------
    # Merging datasets
    # --------------------------------------------------------------

    data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

    data_merged.columns = [
        'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
        'participant', 'label', 'category', 'set',
    ]

    # --------------------------------------------------------------
    # Resample data (frequency conversion)
    # --------------------------------------------------------------

    # Accelerometer:    12.500HZ
    # Gyroscope:        25.000Hz

    sampling = {
        'acc_x' : 'mean',
        'acc_y' : 'mean',
        'acc_z' : 'mean',
        'gyr_x' : 'mean',
        'gyr_y' : 'mean',
        'gyr_z' : 'mean',
        'participant' : 'last',
        'label' : 'last',
        'category' : 'last',
        'set' : 'last'
    }

    # Split by day
    days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]

    data_resampled = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

    data_resampled['set'] = data_resampled['set'].astype(int)

    # --------------------------------------------------------------
    # Export dataset
    # --------------------------------------------------------------

    data_resampled.to_pickle('../../data/interim/01_data_processed.pkl')
