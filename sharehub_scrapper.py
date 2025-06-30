import requests
import pandas as pd
import json
import os
from concurrent.futures import ProcessPoolExecutor

def fetch_sharehubnepal_dividends(symbol, limit=50):
    url = "https://sharehubnepal.com/data/api/v1/dividend"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        "Referer": f"https://sharehubnepal.com/company/{symbol}/dividend-history",
        "Accept": "application/json",
    }
    all_rows = []
    params = {"symbol": symbol, "limit": limit, "page": 1}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    page_data = data.get("data", {})
    total_pages = page_data.get("totalPages", 1)
    content = page_data.get("content", [])
    all_rows.extend(content)
    print(f"Fetched page 1 with {len(content)} rows (total pages: {total_pages})")
    for page in range(2, total_pages + 1):
        params["page"] = page
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        page_data = data.get("data", {})
        content = page_data.get("content", [])
        all_rows.extend(content)
        print(f"Fetched page {page} with {len(content)} rows")
    if not all_rows:
        print("No data found")
        return None
    df = pd.DataFrame(all_rows)
    # Keep only the desired columns and reorder
    keep_cols = ["fiscalYear", "cash", "bonus", "total"]
    df = df[keep_cols]
    return df

def save_datas(symbol, out_dir):
    data = fetch_sharehubnepal_dividends(symbol)
    if data is None or data.empty:
        print(f"No data found for {symbol}, skipping save.")
        return
    filename = f"{symbol}.csv"
    path = os.path.join(out_dir, filename)
    data.to_csv(path, index=False)
    print(f"Data saved for {symbol} in {path}")

if __name__ == "__main__":
    with open('company_list.json', 'r') as f:
        company_list = json.load(f)
    stock_symbols = [c['d'] for c in company_list]
    data_dir = os.path.join(os.getcwd(), 'dividend_data')
    os.makedirs(data_dir, exist_ok=True)
    max_workers = 5
    from functools import partial
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(partial(save_datas, out_dir=data_dir), stock_symbols)