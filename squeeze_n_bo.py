import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
from requests.exceptions import ReadTimeout, ConnectionError

def find_clustered_high_low(df, lookback=50, bins=50, min_occurrences=3):
    """
    找出价格分布中的高频高点和低点
    """
    df = df.copy()
    df['prev_high'] = np.nan
    df['prev_low'] = np.nan
    
    for i in range(lookback, len(df)):
        window = df['price'].iloc[max(0, i - lookback):i]
        if len(window) < lookback:
            continue
        hist, bin_edges = np.histogram(window, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid_bins = bin_centers[hist >= min_occurrences]
        if len(valid_bins) == 0:
            continue
        df['prev_high'].iloc[i] = valid_bins.max()
        df['prev_low'].iloc[i] = valid_bins.min()
    
    return df['prev_high'], df['prev_low']

def detect_squeeze_breakout(df, ma_period=20, amplitude_period=10, squeeze_amplitude_threshold=0.015, lookback=50):
    """
    检测压缩和突破状态，使用EMA和振幅收缩趋势
    """
    df = df.copy()
    
    # 计算20天EMA
    df['ema'] = df['price'].ewm(span=ma_period, adjust=False).mean()
    
    # 计算高频高点和低点
    df['prev_high'], df['prev_low'] = find_clustered_high_low(df, lookback=lookback)
    
    # 计算振幅（相对于EMA）
    df['amplitude'] = (df['price'].rolling(window=amplitude_period).max() - 
                      df['price'].rolling(window=amplitude_period).min()) / df['ema']
    
    # 振幅收缩趋势（最近3天振幅递减）
    df['amplitude_shrinking'] = (
        (df['amplitude'] < df['amplitude'].shift(1)) & 
        (df['amplitude'].shift(1) < df['amplitude'].shift(2))
    )
    
    # 计算价格低点和高点
    df['low'] = df['price'].rolling(window=amplitude_period).min()
    df['high'] = df['price'].rolling(window=amplitude_period).max()
    
    # 计算价格低点和高点与EMA的距离
    df['low_to_ema'] = (df['ema'] - df['low']) / df['ema']
    df['high_to_ema'] = (df['high'] - df['ema']) / df['ema']
    
    # 价格低点递增（向上压缩）
    df['low_rising'] = (
        (df['low'] > df['low'].shift(1)) & 
        (df['low'].shift(1) > df['low'].shift(2))
    )
    
    # 价格高点递减（向下压缩）
    df['high_falling'] = (
        (df['high'] < df['high'].shift(1)) & 
        (df['high'].shift(1) < df['high'].shift(2))
    )
    
    # 初始化状态
    df['state'] = 'normal'
    
    # 压缩状态
    df['price_deviation'] = abs(df['price'] - df['ema']) / df['ema']
    df.loc[
        (df['amplitude'] <= squeeze_amplitude_threshold) & 
        (df['price_deviation'] <= 0.02) & 
        (df['amplitude_shrinking']) & 
        (df['low_rising']), 
        'state'
    ] = 'squeeze_up'
    df.loc[
        (df['amplitude'] <= squeeze_amplitude_threshold) & 
        (df['price_deviation'] <= 0.02) & 
        (df['amplitude_shrinking']) & 
        (df['high_falling']), 
        'state'
    ] = 'squeeze_down'
    
    # 突破状态
    df['prev_state'] = df['state'].shift(1)
    df.loc[
        (df['price'] > df['prev_high']) & 
        (df['prev_state'] == 'squeeze_up'), 
        'state'
    ] = 'breakout_up'
    df.loc[
        (df['price'] < df['prev_low']) & 
        (df['prev_state'] == 'squeeze_down'), 
        'state'
    ] = 'breakout_down'
    
    return df

def fetch_usdjpy_data(max_retries=3, retry_delay=5):
    """
    尝试获取USD/JPY数据，重试机制
    """
    end_date = datetime(2025, 7, 13)
    start_date = end_date - timedelta(days=2*365)
    ticker = 'USDJPY=X'
    
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
            if not df.empty and 'Close' in df.columns:
                return df
            else:
                print(f"Attempt {attempt + 1}: Empty data received for {ticker}")
        except (ReadTimeout, ConnectionError, Exception) as e:
            print(f"Attempt {attempt + 1}: Error fetching {ticker} - {str(e)}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    print(f"Failed to fetch {ticker} data after {max_retries} attempts")
    return None

# 主程序：加载USDJPY数据并生成JSON
if __name__ == "__main__":
    # 尝试获取数据
    df = fetch_usdjpy_data()
    
    if df is None:
        print("无法获取USDJPY数据，请尝试手动下载CSV或使用其他数据源")
        exit()
    
    df['price'] = df['Close']
    
    # 检测压缩和突破
    df = detect_squeeze_breakout(df)
    
    # 准备可视化数据
    viz_data = {
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'prices': df['price'].round(2).tolist(),
        'ema': df['ema'].round(2).tolist(),
        'squeeze_up': df[df['state'] == 'squeeze_up'].index.strftime('%Y-%m-%d').tolist(),
        'squeeze_down': df[df['state'] == 'squeeze_down'].index.strftime('%Y-%m-%d').tolist(),
        'breakout_up': df[df['state'] == 'breakout_up'][['price']].reset_index().to_dict('records'),
        'breakout_down': df[df['state'] == 'breakout_down'][['price']].reset_index().to_dict('records')
    }
    
    # 保存为JSON
    with open('usdjpy_data.json', 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print("数据已保存为 usdjpy_data.json")