import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
from sklearn.decomposition import PCA

def laguerre(g = None, data = None):
    p = (data["High"] + data["Low"])/2
    m = p.size
    L0 = pd.Series(0, index = p.index)
    L1 = pd.Series(0, index = p.index)
    L2 = pd.Series(0, index = p.index)
    L3 = pd.Series(0, index = p.index)
    f = pd.Series(0, index = p.index)

    for i in range(1,m):
        p_L0 = L0.iloc[i - 1]
        p_L1 = L1.iloc[i - 1]
        p_L2 = L2.iloc[i - 1]
        p_L3 = L3.iloc[i - 1]

        L0.iloc[i] = (1-g)*p[i]+g*p_L0
        L1.iloc[i] = -g*L0[i]+p_L0+g*p_L1
        L2.iloc[i] = -g*L1.iloc[i]+p_L1+g*p_L2
        L3.iloc[i] = -g*L2.iloc[i]+p_L2+g*p_L3

    f = (L0 + 2*L1 + 2*L2 + L3)/6

    return f

def zlema(data, period):
    lag = int((period-1)/2)
    return data.ta.ema(close=2*data["Close"] - data["Close"].shift(lag), length=20)

def stdScaler(pdSerie):
    s = pdSerie.std(ddof=0)
    u = pdSerie.mean()
    return (pdSerie - u)/s, {"s": s, "u": u}

if __name__ == "__main__":
    df = pd.read_csv("/content/drive/MyDrive/stock_data/VN30F1M_adjust.csv").drop(columns=["Unnamed: 0"])
    df['Date'] = pd.to_datetime(df.Date)
    df.set_index('Date',inplace=True)

    df_5min = df.resample('5min').agg({'Open':'first', 'High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(how="any")
    df_10min = df.resample('10min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(how="any")
    df_15min = df.resample('15min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(how="any")

    test = df_15min.copy()

    test["AO"] = ta.ao(test["High"], test["Low"])
    test["APO"] = ta.apo(test["Close"])
    test["BIAS"] = ta.bias(test["Close"])
    test["BOP"] = ta.bop(test["Open"], test["High"], test["Low"], test["Close"])
    test = pd.concat([test, ta.brar(test["Open"], test["High"], test["Low"], test["Close"])], axis=1)
    test["CCI"] = ta.cci(test["High"], test["Low"], test["Close"])
    test["CFO"] = ta.cfo(test["Close"])
    test["CG"] = ta.cg(test["Close"])
    test["CMO"] = ta.cmo(test["Close"])
    test["COPPOCK"] = ta.coppock(test["Close"])
    test["CTI"] = ta.cti(test["Close"])
    test = pd.concat([test, ta.dm(test["High"], test["Low"])], axis=1)
    test = pd.concat([test, ta.eri(test["High"], test["Low"], test["Close"])], axis=1)
    test = pd.concat([test, ta.fisher(test["High"], test["Low"])], axis=1)
    test["INERTIA"] = ta.inertia(test["Close"], test["High"], test["Low"])
    test = pd.concat([test, ta.kdj(test["High"], test["Low"], test["Close"])], axis=1)
    test = pd.concat([test, ta.kst(test["Close"])], axis=1)
    test = pd.concat([test, ta.macd(test["Close"])], axis=1)
    test["MOM"] = ta.mom(test["Close"])
    test["PGO"] = ta.pgo(test["High"], test["Low"], test["Close"])
    test = pd.concat([test, ta.ppo(test["Close"])], axis=1)
    test["PSL"] = ta.psl(test["Close"])
    test = pd.concat([test, ta.pvo(test["Volume"])], axis=1)

    qqe = ta.qqe(test["Close"]).fillna(0)
    qqe["QQEl_s"] = qqe["QQEl_14_5_4.236"] - qqe["QQEs_14_5_4.236"]
    test = pd.concat([test, qqe[["QQE_14_5_4.236_RSIMA", "QQE_14_5_4.236", "QQEl_s"]]], axis=1)

    test["ROC"] = ta.roc(test["Close"])
    test["LF"] = laguerre(0.05, test)
    test["RSI"] = ta.rsi(test["LF"])
    test["RSX"] = ta.rsx(test["Close"])
    test = pd.concat([test, ta.rvgi(test["Open"], test["High"], test["Low"], test["Close"])], axis=1)
    test["SLOPE"] = ta.slope(test["Close"])
    test = pd.concat([test, ta.smi(test["Close"])], axis=1)
    test = pd.concat([test, ta.squeeze(test["High"], test["Low"], test["Close"])[["SQZ_20_2.0_20_1.5", "SQZ_OFF"]]], axis=1)
    test = pd.concat([test, ta.squeeze_pro(test["High"], test["Low"], test["Close"])], axis=1)
    test = pd.concat([test, ta.stc(test["Close"])], axis=1)
    test = pd.concat([test, ta.stoch(test["High"], test["Low"], test["Close"])], axis=1)
    test = pd.concat([test, ta.stochrsi(test["Close"])], axis=1)
    test = pd.concat([test, ta.trix(test["Close"])], axis=1)

    test["UO"] = ta.uo(test["High"], test["Low"], test["Close"])
    test["WILLR"] = ta.willr(test["High"], test["Low"], test["Close"])
    test["ALMA"] = ta.alma(test["Close"])
    test["DEMA"] = ta.dema(test["Close"])
    test["EMA_10"] = ta.ema(test["Close"], 10)
    test["EMA_20"] = ta.ema(test["Close"], 20)
    test["EMA_50"] = ta.ema(test["Close"], 50)
    test["EMA_20"] = ta.sma(test["Close"], 20)
    test["EMA_50"] = ta.sma(test["Close"], 50)
    test["EMA_100"] = ta.sma(test["Close"], 100)
    test["FWMA"] = ta.fwma(test["Close"])
    test.ta.vwma(append=True)
    test["ZLEMA_20"] = ta.zlma(test["Close"], 20)
    test["ZLEMA_20_20"] = ta.zlma(test["ZLEMA_20"], 20)
    test["ENTROPY"] = ta.entropy(test["Close"])
    test["KURTOSIS"] = ta.kurtosis(test["Close"])
    test["SKEW"] = ta.skew(test["Close"])
    test = pd.concat([test, ta.adx(test["High"], test["Low"], test["Close"])], axis=1)
    test = pd.concat([test, ta.amat(test["Close"])], axis=1)
    test = pd.concat([test, ta.aroon(test["High"], test["Low"])], axis=1)
    test.ta.chop(append=True)
    test.ta.cksp(append=True)
    test.ta.decay(append=True)
    # test["PSAR"] = ta.psar(test["High"], test["Low"])["PSARl_0.02_0.2"]
    test.ta.qstick(append=True)
    test.ta.ttm_trend(append=True)
    test.ta.vhf(append=True)
    test.ta.vortex(append=True)
    test.ta.aberration(append=True)
    test.ta.accbands(append=True)
    test.ta.atr(append=True)
    test.ta.bbands(append=True)
    test.ta.donchian(append=True)
    test.ta.hwc(append=True)
    test.ta.massi(append=True)
    test.ta.natr(append=True)
    test.ta.pdist(append=True)
    test.ta.rvi(append=True)
    test.ta.ui(append=True)

    testt = test.copy()

    testt.dropna(inplace=True)

    testt.drop(columns=["SQZPRO_NO"], inplace=True)

    anot = {}
    for column in testt.columns:
        try:
            scaled, s_u = stdScaler(testt[column])
            testt[column] = scaled
            anot[column] = s_u
        except:
            print(column)

    n_components=20
    pca = PCA(n_components=n_components)
    r_data = pca.fit_transform(testt)

    columns = [f"PCA{i}" for i in range(n_components)]
    pca_features = pd.DataFrame(data=r_data, columns=columns)
    pca_features.to_csv("pca_features.csv")