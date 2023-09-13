import sys
import subprocess
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from datetime import timedelta, time
import matplotlib.pyplot as plt

def portfolio_pnl_future(position_long, position_short, Close):
    ''' tính PNL của một chiến thuật
    position_long: series position long
    position_short: series position short'''
    intitial_capital_long = (position_long.iloc[0])*(Close.iloc[0])
    cash_long = (position_long.diff(1)*Close)
    cash_long[0] = intitial_capital_long
    cash_cs_long = cash_long.cumsum()
    portfolio_value_long = (position_long*Close)

    intitial_capital_short = (position_short.iloc[0])*(Close.iloc[0])
    cash_short = (position_short.diff(1)*Close)
    cash_short[0] = intitial_capital_short
    cash_cs_short = cash_short.cumsum()
    portfolio_value_short = (position_short*Close)

    backtest = (portfolio_value_long - cash_cs_long) + (cash_cs_short - portfolio_value_short)
    backtest.fillna(0, inplace=True)
    cash_max = (cash_long + cash_short).max()
    pnl =  backtest/cash_max

    ''' return PNL, lần vào lệnh lớn nhất, PNL tương đối theo % '''
    return backtest, cash_max, pnl

def Sharp(pnl):
    ''' Tính Sharp ratio '''
    r = pnl.diff(1)
    return r.mean()/r.std() * np.sqrt(252)

def maximum_drawdown_future(gain, cash_max):
    ''' Tính maximum drawdown theo điểm, theo % '''
    return (gain.cumsum().cummax() - gain.cumsum()).max(), (gain.cumsum().cummax() - gain.cumsum()).max()/cash_max

def Margin(test):
    ''' Tính Margin '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test['inLong'] = test.signal_long.diff()[test.signal_long.diff() > 0].astype(int)
    test['inShort'] = test.signal_short.diff()[test.signal_short.diff() > 0].astype(int)
    test['outLong'] = -test.signal_long.diff()[test.signal_long.diff() < 0].astype(int)
    test['outShort'] = -test.signal_short.diff()[test.signal_short.diff() < 0].astype(int)
    test.loc[test.index[0], 'inLong'] = test.signal_long.iloc[0]
    test.loc[test.index[0], 'inShort'] = test.signal_short.iloc[0]
    test.fillna(0, inplace=True)

    ''' return dataframe chưa thêm các cột inLong, inShort, outLong, outShort và Margin '''
    return test, test.total_gain.iloc[-1]/(test.inLong * test.Close + test.inShort * test.Close + test.outLong * test.Close + test.outShort * test.Close).sum()*10000

def HitRate(test):
    ''' Tính Hit Rate '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test = Margin(test)[0]
    test = test[((test.outLong == 1) | (test.outShort == 1) | (test.inLong == 1) | (test.inShort == 1))]
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test.fillna(0, inplace=True)
    test['gain'] = test.total_gain.diff()
    test.fillna(0, inplace=True)
    test['gain'] = np.where(np.abs(test.gain) < 0.00001, 0, test.gain)
    try:
        ''' return dataframe thu gọn và Hit Rate'''
        return test, len(test[test.gain > 0])/(len(test[test.inLong == 1]) + len(test[test.inShort == 1]))
    except:
        return 0

def test_live(duration, fromtimestamp=1651727820):
    ''' Lấy dữ liệu từ API '''
    ''' Input: duration: sample dữ liệu theo phút '''
    def vn30f():
        return requests.get(f"https://services.entrade.com.vn/chart-api/chart?from={fromtimestamp}&resolution=1&symbol=VN30F1M&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    s = pd.read_csv('../Data/DataMinute/VN30F1M.csv', index_col=0, parse_dates=True)[datetime.datetime.fromtimestamp(fromtimestamp).strftime('%Y-%m-%d'):].reset_index()
    s['Date'] = pd.to_datetime(s['Date']) + timedelta(hours =7)
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()#change s
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = pd.concat([process_data(vn30fm), process_data(s)]).sort_values('Date').drop_duplicates('Date').sort_values('Date')
    return vn30f_base

def test_live_realtime(duration, fromtimestamp=1651727820):
    ''' Lấy dữ liệu từ API '''
    ''' Input: duration: sample dữ liệu theo phút '''
    def vn30f():
        return requests.get(f"https://services.entrade.com.vn/chart-api/chart?from={fromtimestamp}&resolution=1&symbol=VN30F1M&to={int(datetime.datetime.now().timestamp())}").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',}
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()#change s
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = process_data(vn30fm).sort_values('Date').drop_duplicates('Date')
    return vn30f_base

def send_to_telegram(message, token='5683073192:AAHOAHjiRwk3pbNWI4dPFfURa4YaySvbfLY', id='-879820435'):
    ''' Gửi tin nhắn đến telegram '''
    ''' Input: message: tin nhắn muốn gửi
               token: token của bot
                id: id của chat group '''
    apiToken = token
    chatID = id
    try:
        apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage?chat_id={chatID}&text={message}"
        requests.get(apiURL).json()
    except Exception as e:
        print(e)

def position_input(position, path_Po='G:/alpha_live_pos/PHU/PS13_PHU.txt'):
    ''' ghi file position input (cho chiến thuật chạy live) '''
    ''' Input: position: vị thế của chiến thuật
               path_Po: đường dẫn file position input'''
    f = open(path_Po, "w")
    if position != 0:
        info = "pos={}\ntime=5".format(position)
        f.write(info)
    else:
        info = "pos=0\ntime=0"
        f.write(info)

def position_report(position, path_CP='G:/alpha_live_pos/PHU/PS13_PHU_CP.txt'):
    ''' ghi file position report (vị thế hiện tại) (cho chiến thuật chạy live)
        Input: position: vị thế của chiến thuật
               path_CP: đường dẫn file position report'''
    f = open(path_CP, "w")
    pos_rp = "pos={}".format(position)
    f.write(pos_rp)

def DumpCSV_and_MesToTele(name, path_csv_intraday, Position, Close, token, id, position_input=1, fee=0.8):
    ''' Ghi file csv và gửi tin nhắn đến telegram
        Input: name: tên của chiến thuật
               path_csv_intraday: đường dẫn file csv intraday
               Position: Series vị thế của chiến thuật
               Close: Series giá khớp lệnh
               token: token của bot telegram
               id: id của chat group telegram
               position_input: số hợp đồng vào mỗi lệnh'''
    ip_address = output.decode().strip()
    try:
        df = pd.read_csv(path_csv_intraday)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'Position': df.Position.tolist(),
            'Close': df.Close.tolist(),
            'total_gain': df.total_gain.tolist(),
            'gain': df.gain.tolist(),
        }
        try:
            dict_data['Datetime'] = pd.to_datetime(dict_data['Datetime']).to_list()
        except:
            for i in range(len(dict_data['Datetime'])):
                dict_data['Datetime'][i] = pd.to_datetime(dict_data['Datetime'][i])
            dict_data['Datetime'] = list(dict_data['Datetime'])
        df = pd.DataFrame(data=dict_data)
    except:
        dict_data = {
            'Datetime': [pd.to_datetime((datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))],
            'Position': [0],
            'Close': [0],
            'total_gain': [0],
            'gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_intraday, index=False)

    Close = Close.iloc[-1]
    new_Pos = int(Position.iloc[-1])
    time_now = datetime.datetime.now()
    profit = 0
    profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()
    mes = f'{ip_address}\n{name}:'

    if new_Pos != dict_data['Position'][-1] or time_now.time() >= datetime.time(14, 45):

        inputPos = int(new_Pos - dict_data['Position'][-1])
        dict_data['Datetime'].append(pd.to_datetime(time_now.strftime('%Y-%m-%d %H:%M:%S')))
        dict_data['Close'].append(Close)
        dict_data['Position'].append(new_Pos)
        dict_data['total_gain'].append(0)
        dict_data['gain'].append(0)

        df = pd.DataFrame(data=dict_data)
        df['signal_long'] = np.where(df.Position > 0, df.Position, 0)
        df['signal_short'] = np.where(df.Position < 0, np.abs(df.Position), 0)
        df['total_gain'] = portfolio_pnl_future(df['signal_long'], df['signal_short'], df.Close)[0]
        df['gain'] = df.total_gain.diff()
        df.fillna(0, inplace=True)
        df['gain'] = np.where(np.abs(df.gain.to_numpy()) < 0.00001, 0, df.gain.to_numpy())
        df.loc[df['Position'].diff().fillna(0) != 0, 'gain'] = df.loc[df['Position'].diff() != 0, 'gain'] - fee/2
        df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] = df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] - fee
        df['total_gain'] = df.gain.cumsum()
        profit = df.gain.iloc[-1]
        profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()

        if inputPos > 0:
            mes = f'{ip_address}\n{name}:\nLong {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        elif inputPos < 0:
            mes = f'{ip_address}\n{name}:\nShort {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        else:
            mes = f'{ip_address}\n{name}:\nClose at {Close}, Current Pos: {new_Pos*position_input}'

        if np.round(profit*10)/10 != 0:
            mes += f'\nProfit: {np.round(profit*10)/10}'
        mes += f'\nProfit today: {np.round(profit_today*10)/10}'

        df.drop(columns=['signal_long', 'signal_short'], inplace=True)
        send_to_telegram(mes, token, id)
        df.to_csv(path_csv_intraday, index=False)

    else:
        inputPos = 0

    profit_today = np.round(profit_today*10)/10
    print(name)
    print(time_now)
    print(Close)
    print('Input Position:', inputPos*position_input)
    print('Current Position:', new_Pos*position_input)
    if np.round(profit*10)/10 != 0:
        print(f'Profit: {np.round(profit*10)/10}')
    print(f'Profit today: {profit_today}')
    print('\n')

    ''' return dataframe intraday, input position, current position'''
    df['profit_today'] = profit_today
    return df, inputPos, new_Pos

def PNL_per_day(path_csv_daily, profit_today):
    ''' Ghi file csv PNL theo ngày
        Input: path_csv_daily: đường dẫn file csv PNL theo ngày
               profit_today: Series profit_today của chiến thuật
               (Lấy ra từ dataframe df của hàm DumpCSV_and_MesToTele)'''
    try:
        df = pd.read_csv(path_csv_daily)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'gain': df.gain.tolist(),
        }
    except:
        dict_data = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')],
            'gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_daily, index=False)

    gain = profit_today.iloc[-1]
    time_now = datetime.datetime.now()

    if time_now.strftime('%Y-%m-%d') != pd.to_datetime(dict_data['Datetime'][-1]).strftime('%Y-%m-%d'):
        if gain != dict_data['gain'][-1]:
            dict_data['Datetime'].append(time_now.strftime('%Y-%m-%d'))
            dict_data['gain'].append(gain)
            df = pd.DataFrame(data=dict_data)
    else:
        dict_data['gain'][-1] = gain
        df = pd.DataFrame(data=dict_data)

    df['total_gain'] = df['gain'].cumsum()
    df['total_gain'].apply(lambda x: np.round(x*10)/10)
    df.fillna(0, inplace=True)

    df.to_csv(path_csv_daily, index=False)
    ''' return dataframe PNL theo ngày '''
    return df

class BacktestInformation:
    ''' Thông tin backtest của chiến thuật
        Input: Datetime: Series Datetime
                Position: Series Position
                Close: Series Close '''
    ''' CHÚ Ý: Nên dùng class này để lấy được các thông tin của chiến thuật chứ không nên dùng các hàm riêng lẻ
               vì các hàm riêng lẻ phía trên có thể có định dạng position không đồng nhất với class này '''
    def __init__(self, Datetime, Position, Close, fee=0.8):
        signal_long = np.where(Position >= 0, Position, 0)
        signal_short = np.where(Position <= 0, np.abs(Position), 0)
        try:
            Datetime = pd.to_datetime(Datetime)
        except:
            Datetime = Datetime.to_list()
            for i in range(len(Datetime)):
                Datetime[i] = pd.to_datetime(Datetime[i])
        self.df = pd.DataFrame(data={'Datetime': Datetime, 'Position': Position, 'signal_long': signal_long, 'signal_short': signal_short, 'Close': Close})
        self.hold_overnight = not (self.df.loc[self.df['Datetime'].dt.time >= time(14, 45), 'Position'] == 0).all()
        self.df.set_index('Datetime', inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        self.df_brief = HitRate(self.df)[0]
        self.fee = fee

    def PNL(self):
        ''' Tính PNL của chiến thuật '''
        total_gain, cash_max, pnl = portfolio_pnl_future(self.df.signal_long, self.df.signal_short, self.df.Close)

        ''' return Series PNL, cash_max '''
        return total_gain, cash_max, pnl

    def Sharp(self):
        ''' Tính Sharp của chiến thuật '''
        return Sharp(self.PNL()[0].resample("1D").last().dropna())

    def Margin(self):
        ''' Tính Margin của chiến thuật '''
        return Margin(self.df_brief)[1]

    def MDD(self):
        ''' Tính MDD của chiến thuật '''
        return maximum_drawdown_future(self.df_brief.gain, self.PNL()[1])

    def Hitrate(self):
        ''' Tính Hitrate của chiến thuật '''
        return HitRate(self.df_brief)[1]

    def Number_of_trade(self):
        ''' Tính số lần giao dịch của chiến thuật '''
        return len(self.df_brief[self.df_brief.inLong == 1]) + len(self.df_brief[self.df_brief.inShort == 1])

    def Profit_per_trade(self):
        ''' Tính Profit trung bình của 1 giao dịch '''
        return self.PNL()[0].iloc[-1]/self.Number_of_trade() - self.fee

    def Profit_after_fee(self):
        ''' Tính Profit sau khi trừ phí '''
        return np.round(self.Profit_per_trade() * self.Number_of_trade()*10)/10

    def Profit_per_day(self):
        ''' Tính Profit trung bình theo ngày '''
        return self.Profit_after_fee()/len(self.PNL()[0].resample("1D").last().dropna())

    def Hitrate_per_day(self):
        ''' Tính Hitrate theo ngày '''
        if not self.hold_overnight:
            Profit = self.df_brief['gain'].cumsum().resample("1D").last().dropna().diff()
            Profit.loc[Profit.index[0]] = self.df_brief['gain'].cumsum().resample("1D").last().dropna().iloc[0]
            return Profit, len(Profit[Profit > 0])/len(Profit)
        else:
            if self.PNL()[0].resample("1D").last().dropna().iloc[0] != 0:
                Profit = self.PNL()[0].resample("1D").last().dropna().diff()
                Profit.loc[Profit.index[0]] = self.PNL()[0].resample("1D").last().dropna().iloc[0]
            else:
                Profit = self.PNL()[0].resample("1D").last().dropna().iloc[1:].diff()
                Profit.loc[Profit.index[0]] = self.PNL()[0].resample("1D").last().dropna().iloc[1:].iloc[0]
            return Profit, len(Profit[Profit > 0])/len(Profit)

    def Return(self):
        ''' Tính Return trung bình mỗi năm theo % của chiến thuật '''
        cash_max = self.PNL()[1]
        return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D").last()) / 365)/cash_max

    def Profit_per_year(self):
        ''' Tính Profit trung bình theo năm '''
        return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D").last()) / 365)

    def Plot_PNL(self, window_MA=None):
        ''' Print thông tin và Vẽ biểu đồ PNL của chiến thuật
            Input: after_fee: bool, True: plot có trừ phí, False: plot không trừ phí'''

        total_gain, cash_max, pnl = self.PNL()
        total_gain = pd.DataFrame(total_gain.to_numpy(), index=total_gain.index, columns=['total_gain'])
        total_gain.loc[self.df['Position'].diff().fillna(0) != 0, 'fee'] = self.fee/2
        total_gain.loc[np.abs(self.df['Position'].diff().fillna(0)) == 2, 'fee'] = self.fee
        total_gain['fee'] = total_gain['fee'].fillna(0).cumsum()
        total_gain['total_gain_after_fee'] = total_gain['total_gain'] - total_gain['fee']

        print('Margin:',Margin(self.df_brief)[1])
        print(f'MDD: {maximum_drawdown_future(self.df_brief.gain, cash_max)}\n')

        data = [('Total trading quantity', self.Number_of_trade()),
                ('Profit per trade',self.Profit_per_trade()),
                ('Total Profit', np.round(total_gain.total_gain.iloc[-1]*10)/10),
                ('Profit after fee', self.Profit_after_fee()),
                ('Trading quantity per day', self.Number_of_trade()/len(total_gain.total_gain.resample("1D").last().dropna())),
                ('Profit per day after fee', self.Profit_per_day()),
                ('Return', self.Return()),
                ('Profit per year', self.Profit_per_year()),
                ('HitRate', self.Hitrate()),
                ('HitRate per day', self.Hitrate_per_day()[1]),
                ]
        for row in data:
            print('{:>25}: {:>1}'.format(*row))

        if total_gain['total_gain'].resample('1D').last().dropna().iloc[0] != 0:
            total_gain.reset_index(inplace=True)
            previous_day = pd.DataFrame(total_gain.iloc[0].to_numpy(), index=total_gain.columns).T
            previous_day.loc[previous_day.index[0], 'Datetime'] = pd.to_datetime(previous_day['Datetime'].iloc[0]) - timedelta(days = 1)
            previous_day.loc[previous_day.index[0], 'total_gain'] = 0
            total_gain = pd.concat([previous_day, total_gain]).set_index('Datetime')

        # total_gain[f'MA{window_MA}'] = total_gain['total_gain'].rolling(window_MA).mean().fillna(0)
        (total_gain.total_gain.resample("1D").last().dropna()).plot(figsize=(15, 4), label=f'{Sharp(total_gain.total_gain.resample("1D").last().dropna())}')
        (total_gain.total_gain_after_fee.resample("1D").last().dropna()).plot(figsize=(15, 4), label=f'{Sharp(total_gain.total_gain_after_fee.resample("1D").last().dropna())}')

        if window_MA != None:
            (total_gain.total_gain.resample("1D").last().dropna().rolling(window_MA).mean()).plot(figsize=(15, 4), label=f'MA{window_MA}')
        plt.grid()
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('PNL')
        plt.show()

        plt.figure()
        (1 + pnl).plot(figsize=(15, 4), label=f'{self.Sharp()}')
        plt.legend()
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Return')
        plt.show()

        # self.df.set_index('Datetime', inplace=True)
        total_gain['Position'] = self.df['Position']
        total_gain['Close'] = self.df['Close']
        total_gain.reset_index(inplace=True)

        return total_gain[(total_gain['Position'].diff().fillna(0) != 0)|(total_gain['Datetime'].dt.time >= time(14, 45))].set_index('Datetime')
    def getMetrics(self, metric="sharpe"):
        total_gain, cash_max, pnl = self.PNL()
        total_gain = pd.DataFrame(total_gain.to_numpy(), index=total_gain.index, columns=['total_gain'])
        total_gain.loc[self.df['Position'].diff().fillna(0) != 0, 'fee'] = self.fee/2
        total_gain.loc[np.abs(self.df['Position'].diff().fillna(0)) == 2, 'fee'] = self.fee
        total_gain['fee'] = total_gain['fee'].fillna(0).cumsum()
        total_gain['total_gain_after_fee'] = total_gain['total_gain'] - total_gain['fee']

        data = [('Total trading quantity', self.Number_of_trade()),
                ('Profit per trade',self.Profit_per_trade()),
                ('Total Profit', np.round(total_gain.total_gain.iloc[-1]*10)/10),
                ('Profit after fee', self.Profit_after_fee()),
                ('Trading quantity per day', self.Number_of_trade()/len(total_gain.total_gain.resample("1D").last().dropna())),
                ('Profit per day after fee', self.Profit_per_day()),
                ('Return', self.Return()),
                ('Profit per year', self.Profit_per_year()),
                ('HitRate', self.Hitrate()),
                ('HitRate per day', self.Hitrate_per_day()[1]),
                ]

        if total_gain['total_gain'].resample('1D').last().dropna().iloc[0] != 0:
            total_gain.reset_index(inplace=True)
            previous_day = pd.DataFrame(total_gain.iloc[0].to_numpy(), index=total_gain.columns).T
            previous_day.loc[previous_day.index[0], 'Datetime'] = pd.to_datetime(previous_day['Datetime'].iloc[0]) - timedelta(days = 1)
            previous_day.loc[previous_day.index[0], 'total_gain'] = 0
            total_gain = pd.concat([previous_day, total_gain]).set_index('Datetime')

        return Sharp(total_gain.total_gain_after_fee.resample("1D").last().dropna())