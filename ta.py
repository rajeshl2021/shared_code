# Full Pipeline: Pattern Detection, Signal, Backtest, and Visualization (Intraday Optimized)

import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import mplfinance as mpf
import logging
import os

# --- Setup logging ---
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(filename='logs/trading_log.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# --- Download intraday data (1-minute or 5-minute) ---
symbol = 'AAPL'
interval = '1m'  # Change to '5m' for 5-minute data
data = yf.download(symbol, period='8d', interval=interval)
data.reset_index(inplace=True)

# --- Calculate EMA, RSI, MACD, Bollinger Bands, Volume Filter ---
data['EMA20'] = talib.EMA(data['Close'], timeperiod=20)
data['EMA50'] = talib.EMA(data['Close'], timeperiod=50)
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['RSI_SMA5'] = data['RSI'].rolling(window=5).mean()
data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20)
data['VolumeMA'] = data['Volume'].rolling(20).mean()

# --- Single-bar classification ---
def classify_single_bar(row):
    body = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    if total_range == 0: return 'Neutral'
    upper_shadow = row['High'] - max(row['Open'], row['Close'])
    lower_shadow = min(row['Open'], row['Close']) - row['Low']
    body_ratio = body / total_range
    upper_ratio = upper_shadow / total_range
    lower_ratio = lower_shadow / total_range
    if body_ratio < 0.1 or (body_ratio < 0.3 and upper_ratio > 0.3 and lower_ratio > 0.3):
        return 'Neutral'
    return 'Bullish' if row['Close'] > row['Open'] else 'Bearish'

data['SingleBar'] = data.apply(classify_single_bar, axis=1)

# --- Multi-bar pattern detection using TA-Lib ---
patterns = {
    'CDLENGULFING': talib.CDLENGULFING,
    'CDLPIERCING': talib.CDLPIERCING,
    'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
    'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
    'CDL3OUTSIDE': talib.CDL3OUTSIDE,
    'CDL3INSIDE': talib.CDL3INSIDE,
    'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
    'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
    'CDLHARAMI': talib.CDLHARAMI,
    'CDLHARAMICROSS': talib.CDLHARAMICROSS,
    'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
    'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
    'CDLMATCHINGLOW': talib.CDLMATCHINGLOW
}

for name, func in patterns.items():
    data[name] = func(data['Open'], data['High'], data['Low'], data['Close'])

def get_multibar_signal_and_score(row):
    score = 0
    bullish = 0
    bearish = 0
    for name in patterns:
        val = row[name]
        if val > 0:
            bullish += 1
            score += 1
        elif val < 0:
            bearish += 1
            score += 1
    if bullish > bearish:
        return 'Bullish', score
    elif bearish > bullish:
        return 'Bearish', score
    else:
        return 'None', 0

multi_results = data.apply(get_multibar_signal_and_score, axis=1)
data['MultiBar'] = multi_results.apply(lambda x: x[0])
data['SignalScore'] = multi_results.apply(lambda x: x[1])

# --- Combine with next single-bar to form final signal ---
def final_trade_signal(i):
    if i + 1 >= len(data): return 'NoTrade'
    multi = data.loc[i, 'MultiBar']
    single_next = data.loc[i + 1, 'SingleBar']
    if multi == 'Bullish' and single_next == 'Bullish':
        return 'Buy'
    elif multi == 'Bearish' and single_next == 'Bearish':
        return 'Sell'
    return 'NoTrade'

data['FinalSignal'] = [final_trade_signal(i) for i in range(len(data))]

# --- Filter signals ---
def is_valid_signal(i):
    row = data.iloc[i]
    if row['FinalSignal'] not in ['Buy', 'Sell']:
        return False
    if row['SignalScore'] < 2:
        return False

    macd_cross = row['MACD'] > row['MACD_Signal'] if row['FinalSignal'] == 'Buy' else row['MACD'] < row['MACD_Signal']
    bb_confirm = row['Close'] < row['BB_lower'] if row['FinalSignal'] == 'Buy' else row['Close'] > row['BB_upper']

    if row['FinalSignal'] == 'Buy':
        if row['RSI'] < 30 or (30 <= row['RSI'] <= 70 and row['RSI'] > row['RSI_SMA5']
                                and row['Close'] > row['EMA20'] and row['Close'] > row['EMA50']
                                and macd_cross and bb_confirm):
            return True
    elif row['FinalSignal'] == 'Sell':
        if row['RSI'] > 70 or (30 <= row['RSI'] <= 70 and row['RSI'] < row['RSI_SMA5']
                                and row['Close'] < row['EMA20'] and row['Close'] < row['EMA50']
                                and macd_cross and bb_confirm):
            return True
    return False

data['ValidSignal'] = [is_valid_signal(i) for i in range(len(data))]

# --- Trade simulation ---
data['DateOnly'] = pd.to_datetime(data['Datetime' if 'Datetime' in data.columns else 'Date']).dt.date
trades = []
for i in range(len(data)):
    if not data.loc[i, 'ValidSignal']:
        continue
    row = data.iloc[i]
    direction = row['FinalSignal']
    entry_price = row['Close']
    entry_time = row['Datetime'] if 'Datetime' in row else row['Date']
    entry_date = row['DateOnly']
    target = entry_price * (1.001 if direction == 'Buy' else 0.999)
    stop = entry_price * (0.999 if direction == 'Buy' else 1.001)

    for j in range(i + 1, len(data)):
        future = data.iloc[j]
        if future['DateOnly'] != entry_date:
            break
        if direction == 'Buy' and future['High'] >= target:
            exit_price = target
            result = 'Win'
            break
        elif direction == 'Buy' and future['Low'] <= stop:
            exit_price = stop
            result = 'Loss'
            break
        elif direction == 'Sell' and future['Low'] <= target:
            exit_price = target
            result = 'Win'
            break
        elif direction == 'Sell' and future['High'] >= stop:
            exit_price = stop
            result = 'Loss'
            break
    else:
        exit_price = None
        result = 'Open'

    trades.append({
        'EntryTime': entry_time,
        'Direction': direction,
        'EntryPrice': entry_price,
        'ExitPrice': exit_price,
        'Result': result,
        'Return': ((exit_price - entry_price) / entry_price if direction == 'Buy'
                  else (entry_price - exit_price) / entry_price) if exit_price else 0
    })

results = pd.DataFrame(trades)
results = results[results['Result'].isin(['Win', 'Loss'])]

# --- Daily Summary ---
results['Date'] = pd.to_datetime(results['EntryTime']).dt.date
daily_summary = results.groupby('Date').agg(
    TotalTrades=('Result', 'count'),
    Wins=('Result', lambda x: (x == 'Win').sum()),
    Losses=('Result', lambda x: (x == 'Loss').sum()),
    WinRate=('Result', lambda x: (x == 'Win').mean() * 100),
    DailyReturn=('Return', 'sum')
)

daily_summary.to_csv('daily_summary.csv')

# --- Export trades ---
results.to_csv('intraday_trade_signals.csv', index=False)

logging.info(f"Total Trades: {len(results)}")
logging.info(f"Win Rate: {(results['Result'] == 'Win').mean() * 100:.2f}%")
logging.info(f"Average Return: {results['Return'].mean() * 100:.4f}%")
logging.info(f"Total Return: {results['Return'].sum() * 100:.4f}%")

# --- Plotting buy/sell points ---
data.set_index('Datetime' if 'Datetime' in data.columns else 'Date', inplace=True)
buy_signals = data[(data['FinalSignal'] == 'Buy') & data['ValidSignal']]
sell_signals = data[(data['FinalSignal'] == 'Sell') & data['ValidSignal']]

apds = [
    mpf.make_addplot(buy_signals['Close'], type='scatter', markersize=80, marker='^', color='g'),
    mpf.make_addplot(sell_signals['Close'], type='scatter', markersize=80, marker='v', color='r')
]

mpf.plot(data, type='candle', style='charles', addplot=apds,
         title=f'{symbol} Intraday Candlestick with Signals',
         volume=True, figratio=(12,8))
