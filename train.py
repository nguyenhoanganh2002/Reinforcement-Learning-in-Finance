from stock_env_trading import StockTradingEnv
import pandas as pd
from stable_baselines3 import PPO

def load_data():
    df = pd.read_csv("/content/drive/MyDrive/stock_data/VN30F1M_adjust.csv").drop(columns=["Unnamed: 0"])
    df['Date'] = pd.to_datetime(df.Date)
    df.set_index('Date',inplace=True)

    df_5min = df.resample('5min').agg({'Open':'first', 'High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(how="any")
    df_10min = df.resample('10min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(how="any")
    df_15min = df.resample('15min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(how="any")

    data = pd.read_csv("pca_features.csv")
    p_close = df_15min["Close"][-len(data):]/1000

    return p_close, data

if __name__ == "__main__":
    p_close, data = load_data()
    path="/content/drive/MyDrive/stock_data/pos.csv"
    myenv = StockTradingEnv(data=data[:-10000], p_close=p_close[:-10000], path=path)
    model = PPO(policy="MlpPolicy", env=myenv, verbose=0, device="cuda")
    trained = model.learn(total_timesteps=1000000)


