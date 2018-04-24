import os
import argparse
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,TimeDistributed,Flatten,Masking
import numpy as np
import pandas as pd
import util.util as util


class PortfolioOptimizer:
    def __init__(self, date, returns_wanted, risk_level, utility, mean_returns, stock_data, sp_returns, risk_free, cov_matrix):
        self.run_date = date
        self.returns_wanted = returns_wanted
        self.risk_level = risk_level
        self.utility = utility
        self.mean_returns = mean_returns
        self.sp_returns = sp_returns
        self.risk_free = risk_free
        self.stock_data = stock_data
        self.cov_matrix = cov_matrix
        self.num_portfolios = 1000

    def return_possible_portfolio(self):
        max_min_portfolio = util.find_max_sharpe_min_vol(self.mean_returns,
                                                         self.cov_matrix,
                                                         self.num_portfolios,
                                                         self.risk_free,
                                                         self.stock_data)

        returns_annual = self.mean_returns * 250

        data_utility = self.stock_data[self.stock_data.index == self.run_date].reset_index(drop=True)
        stock_list = [stock for stock in self.stock_data.columns.unique()]
        return_list = [returns_annual[stock] for stock in stock_list]
        utility_prices = [data_utility[stock].iloc[0] for stock in stock_list]


        max_utility = util.find_max_utility(stock_list,
                                            return_list,
                                            utility_prices,
                                            self.utility)

        cov_annual = self.cov_matrix * 250

        abs_min_var = util.min_var(stock_list,
                                   return_list,
                                   cov_annual,
                                   self.risk_free,
                                   self.returns_wanted)

        all_types = max_min_portfolio.join(max_utility)\
            .join(abs_min_var)

        return all_types


def main(date, returns_wanted, risk_level, utility):
    stock_data = pd.read_csv('stock_predictions.csv')
    stock_data[['open', 'high', 'low', 'close']] = stock_data[['open', 'high', 'low', 'close']].astype('float')
    sp_returns = pd.read_csv('sp_prediction.csv')
    sp_returns[['open', 'high', 'low', 'close']] = sp_returns[['open', 'high', 'low', 'close']].astype('float')
    risk_free = pd.read_csv('rf_prediction.csv')
    risk_free[['open', 'high', 'low', 'close']] = risk_free[['open', 'high', 'low', 'close']].astype('float')

    ##find forecasted risk free rate
    risk_free = util.find_avg_price(risk_free, ['open', 'low', 'close'])
    risk_free = risk_free['avg_price']
    risk_free_rate = np.average(risk_free[-180:-1])/100.0

##find forecasted market rates
    sp_returns = util.find_avg_price(sp_returns, ['open', 'low', 'high'])
    sp_returns = sp_returns[['avg_price']]
    sp_mean_return = util.find_percent_returns(sp_returns).mean()

    all_stock_list = stock_data.ticker.unique()

    stock_data = util.find_avg_price(stock_data, ['open', 'low', 'high'])

    stock_pivot = util.set_up_stock_data(stock_data)

    returns = util.find_percent_returns(stock_pivot)

    mean_returns = returns.mean()

    cov_matrix = returns.cov()

    optimize = PortfolioOptimizer(
        date,
        float(returns_wanted),
        risk_level,
        float(utility),
        mean_returns,
        stock_pivot,
        sp_returns,
        risk_free_rate,
        cov_matrix)

    final_output = optimize.return_possible_portfolio()
    print(final_output.head())
    return final_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='2017-04-12')
    parser.add_argument('--returns_wanted', default='0.08')
    parser.add_argument('--risk_level', default='medium')
    parser.add_argument('--utility', default='20000')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)










