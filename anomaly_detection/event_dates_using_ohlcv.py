import pandas as pd
import os

doubles = pd.read_csv('/Users/icarus/Desktop/event_driven/doubled_stocks_info.csv')

ohlcv_dir = '/Users/icarus/Desktop/event_driven/datas'

def get_ohlcv_window(stock, event_date, lookback=90):
    """ Instead of going to the floorsheet data, we find the trading days from ohlcv data.
    This assumes we have the ohlcv stock data and event date.
    Change the path if appropriate.
    """
    ohlcv_path = os.path.join(ohlcv_dir, f"{stock}.csv")
    if not os.path.exists(ohlcv_path):
        print(f"Missing OHLCV data for {stock}")
        return None
    df = pd.read_csv(ohlcv_path, parse_dates=['date'])
    df = df.sort_values('date')

    # lets only keep the rows before the event date
    df['date'] = pd.to_datetime(df['date'])
    event_date = pd.to_datetime(event_date)

    # Compare only the date part
    df_until = df[df['date'].dt.date <= event_date.date()]

    # lets take the last 'lookback' rows before the event
    window = df_until.tail(lookback)

    if len(window) < lookback:
        print(f"Not enough data for {stock} on {event_date.date()}")
        return None
    
    return window

# lets try this
rows = {}
for idx, row in doubles.iterrows():
    stock = row['stock_name']
    event_date = row['doubling_date']
    window = get_ohlcv_window(stock, event_date, lookback=90)
    if window is not None:
        start_date = window['date'].iloc[0].date()
        end_date = window['date'].iloc[-1].date()
        if stock not in rows:
            rows[stock] = []
        rows[stock].append({
            'start_date': str(start_date),
            'end_date': str(end_date),
            'doubling_date': str(event_date)
        })
        # print(start_date, end_date)

# Flatten rows to a list of dicts
flat_rows = []
for stock, windows in rows.items():
    for w in windows:
        flat_rows.append({
            'stock_name': stock,
            'start_date': w['start_date'],
            'end_date': w['end_date']
        })

out_df = pd.DataFrame(flat_rows)
# print(out_df)

from nepse import Nepse
nepse = Nepse()
nepse.setTLSVerification(False)

# we need to remove any mutual funds datas if present
sector_scrips = nepse.getSectorScrips()
mutual_funds = set(sector_scrips['Mutual Fund'])

# lets filter out mutual funds scrips
out_df = out_df[~out_df['stock_name'].isin(mutual_funds)]

# print(out_df)
out_df.to_csv('anomaly_date_windows.csv', index=False)
