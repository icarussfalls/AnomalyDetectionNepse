import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

data_folder = '/Users/icarus/Desktop/event_driven/datas'
window_size = 30 # used in prior logic to train VAE, not used here
lookahead_days = 30
threshold = 1.5
min_years = 2  # Minimum years of data required
min_gap_days = 60  # Minimum days between doubling events to avoid consecutive detections

all_sequences = []      # final list of sequences
stock_labels = []       # to track stock for each sequence
doubled_stocks_info = []  # to store info about stocks that doubled

def filter_consecutive_points(double_points, data, min_gap=60):
    """
    Filter out consecutive doubling points that are too close together.
    Keep only the first point in each cluster of consecutive detections.
    """
    if len(double_points) == 0:
        return []
    
    filtered_points = [double_points[0]]  # Always keep the first point
    
    for point in double_points[1:]:
        # Check if this point is far enough from the last kept point
        last_kept_point = filtered_points[-1]
        days_between = (data.index[point] - data.index[last_kept_point]).days
        
        if days_between >= min_gap:
            filtered_points.append(point)
        # else:
            
            # print('')
            # print(f"    Skipping consecutive point at {data.index[point]} (only {days_between} days after previous)")
    
    return filtered_points

for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        stock_name = file.replace('.csv', '')
        path = os.path.join(data_folder, file)
        try:
            data = pd.read_csv(path, parse_dates=True)
            if 'close' not in data.columns or 'date' not in data.columns:
                continue
            
            data.set_index('date', inplace=True)
            data.index = pd.to_datetime(data.index).normalize()
            
            # Filter to only include last 5 years
            five_years_ago = data.index.max() - pd.DateOffset(years=5)
            data = data[data.index >= five_years_ago]

            data = data[['open', 'high', 'low', 'close', 'volume']].dropna()
            
            # === Filter out stocks with less than 2 years of data ===
            if len(data) == 0:
                continue
                
            date_range = data.index.max() - data.index.min()
            years_of_data = date_range.days / 365.25
            
            if years_of_data < min_years:
                # print(f"Skipping {stock_name}: Only {years_of_data:.1f} years of data (less than {min_years} years)")
                continue
                
            print(f"Processing {stock_name}: {years_of_data:.1f} years of data")
            
            # === Doubling point detection ===
            raw_double_points = []
            
            for i in range(min_gap_days, len(data) - lookahead_days):
                prices_now = data['close'].iloc[i]
                future_price = data['close'].iloc[i + lookahead_days]
                if future_price > threshold * prices_now:
                    raw_double_points.append(i)
            
            # Filter out consecutive points
            double_points = filter_consecutive_points(raw_double_points, data, min_gap_days)
            
            print(f"  Raw doubling points: {len(raw_double_points)}, After filtering consecutive: {len(double_points)}")
            
            # Only process stocks that actually doubled (after filtering)
            if len(double_points) > 0:
                # Store doubling information for filtered points only
                for i in double_points:
                    prices_now = data['close'].iloc[i]
                    future_price = data['close'].iloc[i + lookahead_days]
                    
                    doubled_stocks_info.append({
                        'stock_name': stock_name,
                        'doubling_date': data.index[i],
                        'price_at_detection': prices_now,
                        'price_after_lookahead': future_price,
                        'multiplier': future_price / prices_now,
                        'years_of_data': years_of_data,
                        'days_to_target': lookahead_days
                    })
                
                # Add doubling column (not scaled)
                data['doubling'] = 0
                data.iloc[double_points, data.columns.get_loc('doubling')] = 1
                
                
                # === Extract windows ending at doubling points (use raw OHLCV) ===
                features = data[['open', 'high', 'low', 'close', 'volume']].values

                for idx in double_points:
                    if idx - window_size >= 0:
                        window = features[idx - window_size:idx]  # shape: (30, 5)
                        all_sequences.append(window)
                        stock_labels.append(stock_name)
            else:
                print(f"  No valid doubling points found for {stock_name}")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue



# === Save doubled stocks information to CSV ===
def save_doubled_stocks_to_csv():
    if doubled_stocks_info:
        df_doubled = pd.DataFrame(doubled_stocks_info)
        
        # Sort by stock name and doubling date
        df_doubled = df_doubled.sort_values(['stock_name', 'doubling_date'])
        
        # Format the doubling_date for better readability
        df_doubled['doubling_date'] = df_doubled['doubling_date'].dt.strftime('%Y-%m-%d')
        
        # Round numerical values for better readability
        df_doubled['price_at_detection'] = df_doubled['price_at_detection'].round(2)
        df_doubled['price_after_lookahead'] = df_doubled['price_after_lookahead'].round(2)
        df_doubled['multiplier'] = df_doubled['multiplier'].round(3)
        df_doubled['years_of_data'] = df_doubled['years_of_data'].round(1)
        
        # Save to CSV
        output_file = 'doubled_stocks_info.csv'
        df_doubled.to_csv(output_file, index=False)
        # print(f"\nDoubled stocks information saved to '{output_file}'")
        # print(f"Total doubled stocks events (filtered): {len(df_doubled)}")
        # print(f"Unique stocks that doubled: {df_doubled['stock_name'].nunique()}")
        
        # Display summary
        # print("\nSummary:")
        stock_counts = df_doubled['stock_name'].value_counts()
        # print(f"Top 10 stocks with most doubling events (after filtering):")
        # print(stock_counts.head(10))
        
        # Show multiplier statistics
        # print(f"\nMultiplier statistics:")
        # print(f"Average multiplier: {df_doubled['multiplier'].mean():.2f}")
        # print(f"Max multiplier: {df_doubled['multiplier'].max():.2f}")
        # print(f"Min multiplier: {df_doubled['multiplier'].min():.2f}")
        
        return df_doubled
    else:
        print("No doubled stocks found to save.")
        return None

# Call the function to save the CSV
doubled_stocks_df = save_doubled_stocks_to_csv()

# Display first few rows of the results
if doubled_stocks_df is not None:
    print(f"\nFirst 10 rows of doubled stocks data:")
    print(doubled_stocks_df.head(10).to_string(index=False))