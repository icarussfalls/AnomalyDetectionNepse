import pandas as pd
import os

window_size = 90
# load positive windows
positives = pd.read_csv('anomaly_date_windows.csv')

# store negative windows
neg_rows = []

for stock in positives['stock_name'].unique():
    ohlcv_path = f'/Users/icarus/Desktop/event_driven/datas/{stock}.csv'
    if not os.path.exists(ohlcv_path):
        continue
    df = pd.read_csv(ohlcv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    all_dates = df['date'].dt.date.values

    # get all positive window end dates for this stock
    pos_ends = set(positives[positives['stock_name'] == stock]['end_date'])

    # find all possible end indices
    for end_idx in range(window_size - 1, len(df)):
        end_date = df.loc[end_idx, 'date'].date()
        start_date = df.loc[end_idx - window_size + 1, 'date'].date()

        # if this window date is not a positive, add as negative
        if str(end_date) not in pos_ends:
            neg_rows.append({
                'stock': stock,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'label': 0
            })

# 
# print(neg_rows)

# lets randomly sample negatives to match the number of positives
neg_df = pd.DataFrame(neg_rows)
neg_sampled_rows = []

for stock in positives['stock_name'].unique():
    stock_neg_df = neg_df[neg_df['stock'] == stock]
    n_pos = len(positives[positives['stock_name'] == stock])
    if len(stock_neg_df) >= n_pos:
        stock_neg_sample = stock_neg_df.sample(n=n_pos, random_state=42)
        neg_sampled_rows.append(stock_neg_sample)
    # else: skip stocks with not enough negatives

neg_df_sampled = pd.concat(neg_sampled_rows, ignore_index=True)

# Add label 1 to positives
positives['label'] = 1

# Combine and shuffle
all_windows = pd.concat([
    positives[['stock_name', 'start_date', 'end_date', 'label']],
    neg_df_sampled.rename(columns={'stock': 'stock_name'})[['stock_name', 'start_date', 'end_date', 'label']]
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# save to csv
all_windows.to_csv('combined_pos_neg.csv', index=False)