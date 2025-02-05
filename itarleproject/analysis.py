import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime
from sklearn.impute import SimpleImputer

class Itarle_data:
    """This is a class for preprocessing data and the analysis in the sequel
        Attributes
        ========
        trading_time_avg, self.trading_time_med, trading_max_time: pd.Series
            pd.Series which stores the asked quantity for each stock related to trading time
        
        tick_time_avg, tick_time_med, tick_max_time: pd.Series
            pd.Series which stores the asked quantity for each stock related to tick changing time

        spread_bid_ask_avg, spread_bid_ask_med: pd.Series
            pd.Series which stores the asked quantity for each stock related to bid ask spread

        digit_prx_trade_freq, digit_vol_trade_freq, prx_test_res, vol_test_res: pd.Series
            pd.Series which stores the asked quantity for each stock related to the figure with the last digit 0

        prx_test_res, vol_test_res: pd.Series
            pd.Series which stores the asked quantity for each stock related to the hypothesis test on round number effect 

        stock_not_liquid: list
            store stocks not operating

        link: str
            link for data

        Methods:
        =======
        get_data:
            retrive the data and preprocess it.

        check_missing:
            check if there is any missing data. If there is, impute using SimpleImputer.

        data_grouper:
            group data according to columns

        get_sta_trade_time, get_sta_tick, get_sta_bid_ask, get_sta_end0:
            funtion to calculate the desired quantities

        check_higher_freq0:
            function to test the hypothesis on the null being frequency of 0 is 1/10, the alternative is larger than 1/10

        visualize:
            function to plot histograms given data
        
        """
    
    def __init__(self, link):
        self.trading_time_avg = pd.Series(dtype='float64')
        self.trading_time_med = pd.Series(dtype='float64')
        self.trading_max_time = pd.Series(dtype='float64')

        self.tick_time_avg = pd.Series(dtype='float64')
        self.tick_time_med = pd.Series(dtype='float64')
        self.tick_max_time = pd.Series(dtype='float64')

        self.spread_bid_ask_avg = pd.Series(dtype='float64')
        self.spread_bid_ask_med = pd.Series(dtype='float64')

        self.digit_prx_trade_freq = pd.Series(dtype='float64')
        self.digit_vol_trade_freq = pd.Series(dtype='float64')
        self.prx_test_res = pd.Series(dtype='float64')
        self.vol_test_res = pd.Series(dtype='float64')

        self.prx_test_res = pd.Series(dtype='float64')
        self.vol_test_res = pd.Series(dtype='float64')

        self.stock_not_liquid = []
        self.link = link
        self.get_data(self.link)
        self.data_grouper()

    def get_data(self, link):
        df = pd.read_csv(link, usecols=range(12), header=None)
        df = df.join(pd.read_csv(link, usecols=[14], header=None))
        df.drop([1, 9], axis=1, inplace=True)
        df.columns = ['Stock identifier', 'Bid Price', 'Ask Price', 'Trade Price', 'Bid Volume',
                    'Ask Volume', 'Trade Volume', 'Update type', 'Date', 'Time in seconds past midnight',
                    'Condition codes']
        df = df[df['Condition codes'].isnull()  | df['Condition codes'].isin(['@1', 'XT'])]

        dates = pd.to_datetime(df['Date'].apply(str))
        times = df['Time in seconds past midnight'].apply(lambda x: datetime.timedelta(seconds=x))

        df.index = dates + times
        df = df.drop(['Time in seconds past midnight', 'Condition codes'], axis=1)
        self.data = df

    def check_missing(self):
        if self.data.iloc[:, :-1].isna().sum().sum() > 0:
            imp = SimpleImputer()
            imputed = imp.fit_transform(self.data)
            self.data = pd.DataFrame(imputed, columns=self.data.columns)
            print("There is missing data in the dataframe. Impute the missing data by simpleimpiter")
        else:
            print("No missing data.")


    def data_grouper(self, attr='Stock identifier'):
        self.grouped = self.data.groupby(attr)

    def get_sta_trade_time(self):
        for stock, df in self.grouped:
            df = df[df['Update type'] == 1].sort_index()
            if len(df) == 0:
                self.stock_not_liquid.append(stock)
            else:
                diff_trade = df.reset_index().groupby('Date')['index'].apply(lambda x:x.diff()).dropna()
                self.trading_max_time[stock] = np.max(diff_trade).total_seconds()
                self.trading_time_avg[stock] = np.mean(diff_trade).total_seconds()
                self.trading_time_med[stock] = diff_trade.median().total_seconds()

    def get_sta_tick(self):
        for stock, df in self.grouped:
            df = df[df['Update type'] == 1].sort_index().reset_index()
            if len(df) > 0:
                tick_diff = df.groupby('Date')['Trade Price'].diff().fillna(0)
                idx = tick_diff != 0
                time_diff = df[['index', 'Date']][idx].groupby('Date')['index'].diff().dropna()
                self.tick_time_avg[stock] = np.mean(time_diff).total_seconds()
                self.tick_time_med[stock] = time_diff.median().total_seconds()
                self.tick_max_time[stock] = np.max(time_diff).total_seconds()

    def get_sta_bid_ask(self):
        for stock, df in self.grouped:
            spread = df['Ask Price'] - df['Bid Price']
            self.spread_bid_ask_avg[stock] = np.mean(spread)
            self.spread_bid_ask_med[stock] = np.median(spread)

    def get_sta_end0(self):
        for stock, df in self.grouped:
            if stock not in self.stock_not_liquid:
                vol_trade_num = df['Trade Volume'].map(lambda x:str(x)[-1])
                prx_trade_num = df['Trade Price'].map(lambda x:str(x)[-1])
                if (vol_trade_num == '0').any():
                    self.digit_vol_trade_freq[stock] = vol_trade_num.value_counts().loc['0'] / len(df)
                else:
                    self.digit_vol_trade_freq[stock] = 0
            

                if (prx_trade_num == '0').any():
                    self.digit_prx_trade_freq[stock] = prx_trade_num.value_counts().loc['0'] / len(df)
                else:
                    self.digit_prx_trade_freq[stock] = 0

    def check_higher_freq0(self):
        for stock, df in self.grouped:
            if stock not in self.stock_not_liquid:
                vol_trade_num = df['Trade Volume'].map(lambda x: str(x)[-1])
                prx_trade_num = df['Trade Price'].map(lambda x: str(x)[-1])
        
                success_vol = (vol_trade_num == '0').sum()
        
                success_prx = (prx_trade_num == '0').sum()

                _, p_score_vol = sm.stats.proportions_ztest(success_vol, len(df), 0.1, alternative='larger')
                _, p_score_prx = sm.stats.proportions_ztest(success_prx, len(df), 0.1, alternative='larger')

                self.prx_test_res[stock] = p_score_prx < 0.05  # True if we reject H0

                self.vol_test_res[stock] = p_score_vol < 0.05  # True if we reject H0

    def visualize(self, input, title):
        if input is None:
            print("Please give some data to plot")
        else:
            plt.hist(input)
            plt.set_title(title)
            plt.show()

if __name__=='__main__':
    sol = Itarle_data("C:/Users/kaihu/Desktop/ItarleTest/scandi.csv")
    sol.get_sta_end0()
    print(sol.digit_vol_trade_freq.head())