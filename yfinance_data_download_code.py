import yfinance as yf
import pandas as pd
import os

#CONFIG
tickers = ['AAPL', 'MSFT']
start_date = '1998-01-01'
end_date = '2023-12-31'
output_folder = 'stock_data'

#Create output folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()

    #Save CSV
    full_path = os.path.join(output_folder, f"{ticker}_full.csv")
    data.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

print("✅ Done!")

