# maximam-entropy-Ising-XGBoost-Trading-Strategy
本專案運用 Ising Model 模擬市場交易行為，且透過maxi,am entropy產生交易訊號，並透過 XGBoost 機器學習模型 預測下一交易日的股價變動方向。策略包含市場熵分析、機器學習信號生成、回測模組，適用於個股交易與投資決策分析。
1️⃣ 熵（Entropy）計算
本策略使用 Shannon 熵、Rényi 熵、Tsallis 熵 來衡量市場的不確定性：

價格熵（Price Entropy）：衡量價格變動的隨機性
交易量熵（Volume Entropy）：衡量資金流動的集中度
Rényi 熵 & Tsallis 熵：衡量市場分布的長尾效應
2️⃣ Ising Model（市場行為模擬）
利用 Metropolis-Hastings 演算法 進行市場趨勢模擬：

投資者行為視為自旋（+1=買入，-1=賣出）
外場影響（市場趨勢）：價格熵控制市場方向
市場溫度（波動性）：交易量熵決定市場不確定性
最終輸出交易信號
3️⃣ 機器學習（XGBoost）
使用 XGBoost 訓練模型，輸入市場熵特徵，預測 下一天股價變動方向：

訓練集 & 測試集劃分（80% 訓練 / 20% 測試）
ML_Signal 交易信號生成
計算模型準確率（Accuracy）、Precision、Recall
4️⃣ 交易回測模組
策略收益率 vs. 買入持有（Benchmark）
計算績效指標
年化報酬（Annualized Return）
年化波動率（Annualized Volatility）
夏普比率（Sharpe Ratio）
最大回撤（Max Drawdown）
勝率（Win Rate）
可視化
交易信號圖（買入/賣出）
累積收益率對比

