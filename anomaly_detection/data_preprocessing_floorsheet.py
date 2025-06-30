from add_features import *
import json
from scipy.stats import skew, kurtosis, linregress

with open('broker_number.json', 'r') as f:
    data = json.load(f)

# Extract all broker numbers
broker_numbers = [entry["brk_num"] for entry in data]

# connect to the floorsheet database
import sqlite3
# conn = sqlite3.connect("/Users/icarus/Downloads/floorsheet_database.db")
# cursor = conn.cursor()

def data_per_stock(conn, stock_name, start_date, end_date):
    query = """
        SELECT * FROM floorsheet
        WHERE Symbol = ?
        AND Date >= ?
        AND Date <= ?
        """
    df = pd.read_sql_query(query, conn, params=(stock_name, start_date, end_date))
    df['Date'] = pd.to_datetime(df['Date'])
    unique_dates = sorted(df['Date'].dt.date.unique())

    daily_data_list = []
    for date in unique_dates:
        day_df = df[df['Date'].dt.date == date]
        broker_features = get_dimensionless_features(day_df, broker_numbers)
        broker_features['Broker'] = broker_features['Broker'].astype(int)
        broker_features = broker_features.sort_values(by='Broker')
        daily_data_list.append(broker_features)

    if not daily_data_list:
        return None  # No data for this window, skip

    # Get feature columns (excluding 'Broker' and 'Stock' if present)
    feature_cols = [col for col in daily_data_list[0].columns if col not in ['Broker', 'Stock']]

    # Stack daily data into a 3D array: (n_days, n_brokers, n_features)
    daily_array = np.stack([df[feature_cols].values for df in daily_data_list])  # shape: (n_days, n_brokers, n_features)

    # Since, we will be compressing the features, we will aggregate over time for each broker and feature
    agg_funcs = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'skew': lambda x: skew(x, 0, nan_policy='omit'),
        'kurtosis': lambda x: kurtosis(x, 0, nan_policy='omit'),
    }

    compressed_features = []

    for agg_name, agg_func in agg_funcs.items():
        if agg_name in ['skew', 'kurtosis']:
            agg_result = agg_func(daily_array)
        else:
            agg_result = agg_func(daily_array, axis=0)
        agg_df = pd.DataFrame(agg_result, columns=[f"{col}_{agg_name}" for col in feature_cols])
        compressed_features.append(agg_df)

    # Add first, last, and delta (last - first) features
    first_vals = daily_array[0]
    last_vals = daily_array[-1]
    delta_vals = last_vals - first_vals

    first_df = pd.DataFrame(first_vals, columns=[f"{col}_first" for col in feature_cols])
    last_df = pd.DataFrame(last_vals, columns=[f"{col}_last" for col in feature_cols])
    delta_df = pd.DataFrame(delta_vals, columns=[f"{col}_delta" for col in feature_cols])

    compressed_features.extend([first_df, last_df, delta_df])

    # Add trend (slope) features using linear regression for each broker-feature
    trend = np.zeros((daily_array.shape[1], daily_array.shape[2]))
    for i in range(daily_array.shape[1]):  # brokers
        for j in range(daily_array.shape[2]):  # features
            y = daily_array[:, i, j]
            x = np.arange(len(y))
            mask = ~np.isnan(y)
            if mask.sum() > 1:
                slope, _, _, _, _ = linregress(x[mask], y[mask])
            else:
                slope = np.nan
            trend[i, j] = slope
    trend_df = pd.DataFrame(trend, columns=[f"{col}_trend" for col in feature_cols])
    compressed_features.append(trend_df)

    # Concatenate all aggregations horizontally
    final_df = pd.concat(compressed_features, axis=1)
    final_df['Broker'] = daily_data_list[0]['Broker'].values

    print("Compressed feature shape:", final_df.shape)
    # print(final_df.head())
    return final_df

# stock_name = 'AHPC'
# start_date = '2024-04-09'
# end_date = '2024-07-10'
# data = data_per_stock(stock_name, start_date, end_date)

# F = data.drop(columns=['Broker']).fillna(0).to_numpy()
# F_enriched = attention_enrich_features(F)
# # Reattach Broker info
# final_df = pd.DataFrame(F_enriched, columns=[f'feat_{i}' for i in range(F_enriched.shape[1])])
# # final_df['Broker'] = data['Broker'].values
# print(final_df)