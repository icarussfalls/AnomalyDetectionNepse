from data_preprocessing_floorsheet import *
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

date_windows = pd.read_csv('combined_pos_neg.csv')
date_windows = date_windows.rename(columns={
    'stock': 'stock_name',
    'start': 'start_date',
    'end': 'end_date'
})

def extract_features(row, db_path):
    stock = row['stock_name']
    start = row['start_date']
    end = row['end_date']
    label = row['label']
    with sqlite3.connect(db_path) as conn:
        features = data_per_stock(conn, stock, start, end)  # shape: (93, 441)
    if features is not None:
        return features.values.flatten(), label
    return None

if __name__ == "__main__":
    db_path = "/kaggle/input/floorsheet-sqlite/floorsheet_database.db"  # db path on kaggle
    rows = [row for _, row in date_windows.iterrows()]
    extract_func = partial(extract_features, db_path=db_path)
    with Pool(processes=cpu_count()) as pool:
        results = []
        for r in tqdm(pool.imap(extract_func, rows), total=len(rows)):
            if r is not None:
                results.append(r)

    # Filter out None results
    results = [r for r in results if r is not None]
    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Save as numpy arrays
    np.save('X_features.npy', X)
    np.save('y_labels.npy', y)
    print("Saved X_features.npy and y_labels.npy")