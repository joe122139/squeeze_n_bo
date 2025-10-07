import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
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

def detect_squeeze_breakout(df, ma_period=20, amplitude_period=10, squeeze_amplitude_threshold=0.3, lookback=50):
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
    start_date = end_date - timedelta(days=10*365)
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

def get_squeeze_regions(df, state_col='state', state_value='squeeze_up'):
    """
    识别连续的squeeze区域，使用向量操作
    """
    regions = []
    df = df.copy()
    df['is_state'] = (df[state_col] == state_value).astype(int)
    df['group'] = (df['is_state'].diff() != 0).cumsum()
    
    state_groups = df[df['is_state'] == 1].groupby('group')
    
    for _, group in state_groups:
        if not group.empty:
            start_date = group.index[0]
            end_date = group.index[-1]
            regions.append((start_date, end_date))
    
    return regions

def plot_usdjpy(df):
    """
    使用Plotly可视化价格、EMA、squeeze区域、breakout点及prev_high/prev_low水平线
    """
    # 创建图表
    fig = go.Figure()
    
    # 添加价格曲线
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price'],
        mode='lines',
        name='USD/JPY Price',
        line=dict(color='#2196F3', width=2)
    ))
    
    # 添加20天EMA
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema'],
        mode='lines',
        name='20-day EMA',
        line=dict(color='#4CAF50', width=2)
    ))
    
    # 添加突破点
    fig.add_trace(go.Scatter(
        x=df[df['state'] == 'breakout_up'].index,
        y=df[df['state'] == 'breakout_up']['price'],
        mode='markers',
        name='Breakout Up (Buy)',
        marker=dict(color='#2196F3', size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=df[df['state'] == 'breakout_down'].index,
        y=df[df['state'] == 'breakout_down']['price'],
        mode='markers',
        name='Breakout Down (Sell)',
        marker=dict(color='#F44336', size=10, symbol='circle')
    ))
    
    # 获取squeeze区域
    squeeze_up_regions = get_squeeze_regions(df, 'state', 'squeeze_up')
    squeeze_down_regions = get_squeeze_regions(df, 'state', 'squeeze_down')
    
    # 添加squeeze区域和prev_high/prev_low线
    shapes = []
    
    # Squeeze Up区域（绿色背景）及水平线
    for start, end in squeeze_up_regions:
        # 背景区域
        shapes.append(dict(
            type="rect",
            x0=start,
            x1=end,
            y0=df['price'].min(),
            y1=df['price'].max(),
            fillcolor="rgba(76, 175, 80, 0.2)",
            line=dict(width=0),
            layer='below'
        ))
        # prev_high水平线
        high_value = df.loc[start:end, 'prev_high'].mean()
        if not pd.isna(high_value):
            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[high_value, high_value],
                mode='lines',
                name='Prev High (Squeeze Up)',
                line=dict(color='#9C27B0', width=1, dash='dash'),
                showlegend=False
            ))
        # prev_low水平线
        low_value = df.loc[start:end, 'prev_low'].mean()
        if not pd.isna(low_value):
            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[low_value, low_value],
                mode='lines',
                name='Prev Low (Squeeze Up)',
                line=dict(color='#FF9800', width=1, dash='dash'),
                showlegend=False
            ))
    
    # Squeeze Down区域（红色背景）及水平线
    for start, end in squeeze_down_regions:
        # 背景区域
        shapes.append(dict(
            type="rect",
            x0=start,
            x1=end,
            y0=df['price'].min(),
            y1=df['price'].max(),
            fillcolor="rgba(244, 67, 54, 0.2)",
            line=dict(width=0),
            layer='below'
        ))
        # prev_high水平线
        high_value = df.loc[start:end, 'prev_high'].mean()
        if not pd.isna(high_value):
            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[high_value, high_value],
                mode='lines',
                name='Prev High (Squeeze Down)',
                line=dict(color='#9C27B0', width=1, dash='dash'),
                showlegend=False
            ))
        # prev_low水平线
        low_value = df.loc[start:end, 'prev_low'].mean()
        if not pd.isna(low_value):
            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[low_value, low_value],
                mode='lines',
                name='Prev Low (Squeeze Down)',
                line=dict(color='#FF9800', width=1, dash='dash'),
                showlegend=False
            ))
    
    # 添加图例条目（仅一次）
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='Prev High',
        line=dict(color='#9C27B0', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='Prev Low',
        line=dict(color='#FF9800', width=1, dash='dash')
    ))
    
    # 更新图表布局
    fig.update_layout(
        title='USD/JPY Price Action with Squeeze and Breakout Levels (2023-2025)',
        xaxis_title='Date',
        yaxis_title='Price',
        shapes=shapes,
        showlegend=True,
        height=600,
        xaxis=dict(rangeslider=dict(visible=True), type='date')
    )
    
    # 保存为HTML
    fig.write_html('usdjpy_plot.html')
    print("图表已保存为 usdjpy_plot.html")
    fig.show()

# 主程序：加载数据并可视化
if __name__ == "__main__":
    # 尝试获取数据
    df = fetch_usdjpy_data()
    
    if df is None:
        print("无法获取USDJPY数据，请尝试手动下载CSV或使用其他数据源")
        print("CSV格式示例：")
        print("Date,Close\n2023-07-13,138.50\n...")
        print("修改代码以读取CSV：")
        print("df = pd.read_csv('usdjpy.csv', parse_dates=['Date'], index_col='Date')")
        print("df['price'] = df['Close']")
        exit()
    
    df['price'] = df['Close']
    
    # 检测压缩和突破
    df = detect_squeeze_breakout(df)
    
    # 可视化
    plot_usdjpy(df)