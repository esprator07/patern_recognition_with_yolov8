import os
import random
import pandas as pd
import mplfinance as mpf
from binance.client import Client
from binance.enums import HistoricalKlinesType
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt

# Binance API Connection
client = Client()

# Definitions for random selection
symbols = []  # All USDT pairs will be stored here
timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]  # Timeframes to be used

def get_all_symbols():
    """ Fetches all USDT pairs from Binance. """
    global symbols
    exchange_info = client.get_exchange_info()
    symbols = sorted([symbol['symbol'] for symbol in exchange_info['symbols'] 
                      if symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')])

def generate_random_chart(save_path, num_charts=200):
    """ Generates and saves random charts. """
    if not symbols:
        get_all_symbols()  # If not fetched before, get the coin list
    
    os.makedirs(save_path, exist_ok=True)  # Create the folder
    
    for i in range(num_charts):
        print(i+1)
        plt.close('all')  # **Close all previous plots**
        try:
            # Randomly select coin and timeframe
            symbol = random.choice(symbols)
            timeframe = random.choice(timeframes)
            num_bars = random.randint(30, 70)  # Random number of bars (between 10-70)
            
            # Fetch historical data (randomly starting up to 30 days ago)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=random.randint(30, 70))
            
            klines = client.get_historical_klines(symbol, timeframe, start_time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  limit=num_bars, klines_type=HistoricalKlinesType.SPOT)
            if not klines:
                continue  # Skip if no data is returned
            
            # Convert data to DataFrame
            df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume", 
                                               "close_time", "quote_asset_volume", "trades", 
                                               "taker_base_vol", "taker_quote_vol", "ignore"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            
            # ✅ Fix: Convert volume data to float!
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            
            df.set_index("time", inplace=True)
            
            # **📌 Style settings**
            mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='gray')
            style = mpf.make_mpf_style(marketcolors=mc, gridcolor="none")  # **Removed grid**
            
            # **📌 Plot creation**
            fig, ax = mpf.plot(df, type='candle', style=style, returnfig=True)
            
            # **📌 Remove X and Y axes (Hide date and price labels)**
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
            ax[0].set_xlabel("")
            ax[0].set_ylabel("")
            
            # **📌 Set chart size to 512x512 px**
            fig.set_size_inches(5.12, 5.12)  # Inches setting for 512x512 px
            filename = f"{symbol}_{timeframe}_{num_bars}bars_{i+24120}.png"
            file_path = os.path.join(save_path, filename)
            fig.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0)  # Save as 512x512 px with dpi=100
            
            print(f"✅ Saved: {file_path}")

        except Exception as e:
            print(f"❌ Error ({symbol} - {timeframe}): {e}")

# Usage
save_directory = "random_charts"
generate_random_chart(save_directory, num_charts=2000)
