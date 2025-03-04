import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 下載台積電數據
symbol = '2330.TW'
data = yf.download(symbol, period='5y', interval='1d', auto_adjust=False)

# 確保 Close 和 Volume 欄位存在
if 'Close' not in data.columns or 'Volume' not in data.columns:
    raise ValueError("Data does not contain required columns. Make sure auto_adjust=False.")

# 計算對數收益率
log_returns = np.log(data['Close'] / data['Close'].shift(1))

# 設定計算熵的窗口（例如 10 天）
window = 10
price_entropy = []
volume_entropy = []
renyi_entropy = []
tsallis_entropy = []

def calculate_entropy(data_series, bins=10):
    hist, _ = np.histogram(data_series, bins=bins, density=True)
    hist = hist[hist > 0]  # 避免 log(0) 問題
    return entropy(hist)

def calculate_renyi_entropy(data_series, alpha=2, bins=10):
    hist, _ = np.histogram(data_series, bins=bins, density=True)
    hist = hist[hist > 0]
    return (1 / (1 - alpha)) * np.log(np.sum(hist ** alpha))

def calculate_tsallis_entropy(data_series, q=2, bins=10):
    hist, _ = np.histogram(data_series, bins=bins, density=True)
    hist = hist[hist > 0]
    return (1 / (q - 1)) * (1 - np.sum(hist ** q))

for i in range(len(data) - window + 1):
    window_returns = log_returns.iloc[i:i + window].dropna()
    window_volumes = data['Volume'].iloc[i:i + window].dropna()
    
    price_entropy.append(calculate_entropy(window_returns) if len(window_returns) > 0 else np.nan)
    volume_entropy.append(calculate_entropy(window_volumes) if len(window_volumes) > 0 else np.nan)
    renyi_entropy.append(calculate_renyi_entropy(window_returns) if len(window_returns) > 0 else np.nan)
    tsallis_entropy.append(calculate_tsallis_entropy(window_returns) if len(window_returns) > 0 else np.nan)

# 將熵加入 DataFrame
entropy_series_price = pd.Series([np.nan] * (window - 1) + price_entropy, index=data.index)
entropy_series_volume = pd.Series([np.nan] * (window - 1) + volume_entropy, index=data.index)
entropy_series_renyi = pd.Series([np.nan] * (window - 1) + renyi_entropy, index=data.index)
entropy_series_tsallis = pd.Series([np.nan] * (window - 1) + tsallis_entropy, index=data.index)
data['Price_Entropy'] = entropy_series_price
data['Volume_Entropy'] = entropy_series_volume
data['Renyi_Entropy'] = entropy_series_renyi
data['Tsallis_Entropy'] = entropy_series_tsallis

data.dropna(inplace=True)

# 設置 Ising Model 參數
N = len(data)
J = 0.5  # 投資者之間的影響力（可調整）
h = data['Price_Entropy'].values * -1  # 市場趨勢作為外場影響
T = data['Volume_Entropy'].values  # 市場波動作為溫度

# 初始化自旋狀態（+1 = 買入, -1 = 賣出）
spins = np.random.choice([-1, 1], size=N)

# 進行 Metropolis-Hastings 模擬
num_iterations = 1000
for _ in range(num_iterations):
    i = np.random.randint(0, N)
    dE = 2 * spins[i] * (J * (spins[(i-1) % N] + spins[(i+1) % N]) + h[i])
    if dE < 0 or np.random.rand() < np.exp(-dE / (T[i] + 1e-6)):
        spins[i] *= -1

# 計算市場磁化量（趨勢方向）
M = np.cumsum(spins) / np.arange(1, N+1)

# 生成交易信號
data['Signal'] = np.sign(M)
data['Signal'] = data['Signal'].replace(0, np.nan).ffill()

# 構建機器學習數據集
features = data[['Price_Entropy', 'Volume_Entropy', 'Renyi_Entropy', 'Tsallis_Entropy']]
labels = np.where(log_returns.shift(-1).iloc[-len(features):] > 0, 1, 0)  # 下一天上漲為 1，否則為 0

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 訓練 XGBoost 模型
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train.ravel())

# 進行預測
y_pred = model.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 生成機器學習交易信號
data['ML_Signal'] = model.predict(features)

# === 回測模組 ===
# 計算每日收益率
data['Returns'] = log_returns

# 策略收益率：根據 ML_Signal 持有（1）或空倉（0），這裡假設不做空
data['Strategy_Returns'] = data['ML_Signal'].shift(1) * data['Returns']  # 前一天信號決定當天持倉
data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)  # 初始無信號時設為0

# 基準收益率：買入並持有
data['Benchmark_Returns'] = data['Returns']

# 計算累積收益率
data['Strategy_Cum_Returns'] = (1 + data['Strategy_Returns']).cumprod()
data['Benchmark_Cum_Returns'] = (1 + data['Benchmark_Returns']).cumprod()

# === 計算績效指標 ===
def calculate_performance_metrics(returns, benchmark_returns, annual_trading_days=252):
    # 年化收益率
    total_return = (1 + returns).prod() - 1
    years = len(returns) / annual_trading_days
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # 年化波動率
    annualized_volatility = returns.std() * np.sqrt(annual_trading_days)

    # 夏普比率（假設無風險利率為0）
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan

    # 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 勝率
    win_rate = (returns > 0).mean()

    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate
    }

# 計算策略和基準的績效
strategy_metrics = calculate_performance_metrics(data['Strategy_Returns'], data['Benchmark_Returns'])
benchmark_metrics = calculate_performance_metrics(data['Benchmark_Returns'], data['Benchmark_Returns'])

# 顯示績效指標
print("\n=== Strategy Performance Metrics ===")
for key, value in strategy_metrics.items():
    print(f"{key}: {value:.4f}")

print("\n=== Benchmark (Buy & Hold) Performance Metrics ===")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.4f}")

# === 繪製結果 ===
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Close'], label='TWII Price', alpha=0.7)
plt.scatter(data.index[data['ML_Signal'] == 1], data['Close'][data['ML_Signal'] == 1], 
            color='g', label='ML Buy Signal', marker='^')
plt.scatter(data.index[data['ML_Signal'] == 0], data['Close'][data['ML_Signal'] == 0], 
            color='r', label='ML Sell Signal', marker='v')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.title("Trading Strategy Based on Ising Model & XGBoost")

plt.subplot(2, 1, 2)
plt.plot(data.index, data['Strategy_Cum_Returns'], label='Strategy Cumulative Returns', color='b')
plt.plot(data.index, data['Benchmark_Cum_Returns'], label='Buy & Hold Cumulative Returns', color='gray', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.title("Strategy vs. Benchmark Performance")

plt.tight_layout()
plt.show()

# 顯示數據
print("\n=== Last 20 Days of Data ===")
print(data[['Close', 'Price_Entropy', 'Volume_Entropy', 'Renyi_Entropy', 'Tsallis_Entropy', 
            'Signal', 'ML_Signal', 'Strategy_Returns', 'Strategy_Cum_Returns']].tail(20))
