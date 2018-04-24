import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import stockstats
from math import sqrt
import scipy.optimize as spo


valid_set_size_percentage = 10
test_set_size_percentage = 10


def load_data_simple(stock, seq_len):
    data_raw = stock  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data);
    valid_set_size = int(np.round(np.float(valid_set_size_percentage) / 100 * data.shape[0]));
    test_set_size = int(np.round(np.float(test_set_size_percentage) / 100 * data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


def load_data_output_multiple(diff_df, array, seq_len, y_cols=[1,2,3,4]):
    data_raw_x = array
    data_raw_y = array[:, y_cols]  # convert to numpy array

    min_max_scaler_x = StandardScaler()
    x_norm = min_max_scaler_x.fit_transform(data_raw_x)

    x_norm = add_date_features(diff_df.index[0], x_norm)

    min_max_scaler_y = MinMaxScaler()
    y_norm = min_max_scaler_y.fit_transform(data_raw_y)

    index_range_x = seq_len * 2

    data = []
    data_y = []

    # create all possible sequences of length seq_len
    for index in range(len(x_norm) - index_range_x):
        data.append(x_norm[index: index + seq_len])

    for index in range(seq_len, len(y_norm) - seq_len):
        data_y_array = y_norm[index: index + seq_len]
        data_y.append(data_y_array)

    data = np.array(data);
    data_y = np.array(data_y);
    valid_set_size = int(np.round(np.float(valid_set_size_percentage) / 100 * data.shape[0]));
    test_set_size = int(np.round(np.float(test_set_size_percentage) / 100 * data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data_y[:train_set_size, :-1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data_y[train_set_size:train_set_size + valid_set_size, :-1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data_y[train_set_size + valid_set_size:, :-1, :]

    return_dict = {'x_train': x_train,
                   'y_train': y_train,
                   'x_valid': x_valid,
                   'y_valid': y_valid,
                   'x_test': x_test,
                   'y_test': y_test,
                   'x_scaler': min_max_scaler_x,
                   'y_scaler': min_max_scaler_y}

    return return_dict

def inverse_x(x_matrix, min_max_scaler_x, begin_df):
    new_x = np.reshape(x_matrix[1], (89, 124))
    for i in range(2, x_matrix.shape[0]/90+1):
        next_x = np.reshape(x_matrix[i*90], (89, 124))
        new_x = np.concatenate([new_x,next_x])
    reverse_scale = min_max_scaler_x.inverse_transform(new_x[:,1:44])
    reverse_x_df = pd.DataFrame(reverse_scale)
    reverse_x_df.columns=begin_df.columns
    inverse_df_x = inverse_df(begin_df, reverse_x_df, reverse_x_df.columns)
    return inverse_df_x

def inverse_y(y_matrix, min_max_scaler_y, input_predict):
    new_y = np.reshape(y_matrix[1], (89, 4))
    for i in range(2, y_matrix.shape[0]/90+1):
        next_y = np.reshape(y_matrix[i*90], (89, 4))
        new_y = np.concatenate([new_y,next_y])
    reverse_scale_y = min_max_scaler_y.inverse_transform(new_y)
    reverse_y_df = pd.DataFrame(reverse_scale_y)
    reverse_y_df.columns=['open', 'high', 'low', 'close']
    inverse_df_y = inverse_df(input_predict[90:90], reverse_y_df, ['open', 'high', 'low', 'close'])
    return

def write_small_df(result_df, full_table_name):
    names = full_table_name.split(':')
    result_df.toPandas().to_gbq(destination_table=names[1],
                                project_id=names[0],
                                if_exists='append')


def generate_date_features(date_index):
    out_df = pd.DataFrame(index=date_index)
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    for i in range(len(days)):
        kwargs = {days[i]: date_index.map(lambda row: int(row.weekday() == i))}
        out_df = out_df.assign(**kwargs)

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for i in range(len(months)):
        kwargs = {months[i]: date_index.map(lambda row: int(row.month == i + 1))}
        out_df = out_df.assign(**kwargs)

    quarter = ['Q1', 'Q2', 'Q3', 'Q4']
    for i in range(len(quarter)):
        kwargs = {quarter[i]: date_index.map(lambda row: int(row.quarter == i))}
        out_df = out_df.assign(**kwargs)

    years = ['y14', 'y15', 'y16', 'y17']
    for i in range(len(years)):
        kwargs = {years[i]: date_index.map(lambda row: int(row.year == i + 2014))}
        out_df = out_df.assign(**kwargs)

    weeks = ['w{}'.format(i) for i in range(1, 54)]
    for i in range(len(weeks)):
        kwargs = {weeks[i]: date_index.map(lambda row: int(row.isocalendar()[1] == i + 1))}
        out_df = out_df.assign(**kwargs)

    # TODO: fix this
    def is_xmas_new_year(row):
        ret = ((dt.datetime(row.year, 12, 25) < row and row < dt.datetime(row.year + 1, 1, 5)) or
               (dt.datetime(row.year - 1, 12, 25) < row and row < dt.datetime(row.year, 1, 5)))
        return int(ret)

    # kwargs = {'xmas': date_index.map(is_xmas_new_year)}
    # out_df = out_df.assign(**kwargs)
    # print(out_df.head())

    dr = pd.to_datetime(pd.to_datetime(date_index))
    cal = calendar()
    holidays = cal.holidays(start=dr.min(), end=dr.max())

    out_df["holiday"] = dr.isin(holidays)
    out_df["holiday"] = out_df.holiday.astype(int)
    # out_df = out_df.assign(**kwargs)

    return out_df


def add_date_features(first_date, arr):
    date_range = pd.date_range(first_date, periods=arr.shape[0], freq='D')
    df_date_feat = generate_date_features(date_range)
    new_array = np.expand_dims(df_date_feat, axis=0)
    shape = new_array.shape
    reshaped = np.reshape(new_array, (shape[1], shape[2]))
    return np.concatenate((arr, reshaped), axis=1)


def iterate_prediction(model, x):
    x_copy = x.copy()
    y_copy_df = pd.DataFrame
    for i in range(0, 50):
        y_testing = model.predict(x_copy)
        next_y = np.reshape(y_testing[-1], (1, 4))
        last_x = x_copy[-1][1:44]
        new_x = np.reshape(np.concatenate([last_x, next_y]), (1, 4, 4))
        x_copy = np.concatenate([x_copy, new_x])
        y_copy = model.predict(x_copy)
        y_copy_df = pd.DataFrame(y_copy)
    final_y_copy = y_copy_df
    return final_y_copy

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


def inverse_series(old_series, transformed):
    values = []
    last_value = 0
    for i in range(len(transformed)):
        if i <= len(old_series)-1:
            new_value = old_series[i] + transformed[i]
            values.append(new_value)
            last_value = new_value
        else:
            new_value = last_value + transformed[i]
            last_value = new_value
            values.append(last_value)
    return pd.Series(values)


def inverse_df(old_df, new_df, column_list):
    new_data = pd.DataFrame(columns=column_list)
    for i in column_list:
        old = old_df[i]
        new = new_df[i]

        inverse_data = inverse_series(old, new)

        new_data[i] = inverse_data
    return new_data


def find_stock_features(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.price_change = (df.Close / df.Close.shift(1)) - 1
    new = stockstats.StockDataFrame.retype(df)
    new.get('macd')
    new.get('cci')
    new.get('rsi_6')
    new.get('dma')
    new.get('adx')
    new.get('trix')
    new.get('atr')
    new = new.dropna()
    return new


def normalize_data(df, columns):
    min_max_scaler = MinMaxScaler()
    df[columns] = min_max_scaler.fit_transform(df[columns].as_matrix())
    return df, min_max_scaler


def inverse_data(df, columns, scaler):
    df[columns] = scaler.inverse_transform(df[columns].as_matrix())
    return df


def set_up_df(csv):
    df = pd.read_csv(csv)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 'date']]
    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'ticker']]
    return df

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def create_stock_data(ticker, df):
    df_stock = df.iloc[df.index.get_level_values('Symbol') == ticker]
    df_stock = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']]
    df_stock = find_stock_features(df_stock)
    df_stock = df_stock.dropna()
    diff_df = df_stock.diff().dropna()
    last_date = diff_df.index[0]
    diff_df = diff_df.reset_index(drop=True)

    diff_df = diff_df[[u'open', u'high', u'low',
                       u'close', u'volume', u'close_26_ema', u'macd',
                       u'macds', u'macdh', u'middle', u'middle_14_sma',
                       u'cci', u'close_-1_s', u'close_-1_d', u'rs_6',
                       u'rsi_6', u'close_10_sma', u'close_50_sma', u'dma',
                       u'high_delta', u'um', u'low_delta', u'dm',
                       u'pdm', u'pdm_14_ema', u'pdm_14', u'tr',
                       u'atr_14', u'pdi_14', u'pdi', u'mdm',
                       u'mdm_14_ema', u'mdm_14', u'mdi_14', u'mdi',
                       u'dx_14', u'dx', u'dx_6_ema', u'adx',
                       u'adx_6_ema', u'adxr', u'trix', u'atr']]

    diff_df_matrix = clean_dataset(diff_df).as_matrix()

    return diff_df_matrix, last_date


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def find_sharpe_o_var(weights, arg_list):
    mean_returns, cov_matrix, risk_free_rate, desired_return, sharpe = arg_list
    portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return portfolio_std_dev

def max_it(stocks, weights, mean_returns, cov_matrix, risk_free_rate, sharpe=True):
    cons = ({'type': 'ineq',
             'fun': lambda weight: np.sum(weight) - 1.0})
    bds = tuple([(0, 1) for weight in weights])
    result = spo.minimize(find_sharpe_o_var, [1/len(weights) for weight in weights],
                          args=[mean_returns,
                                cov_matrix,
                                risk_free_rate,
                                True],
                          method='SLSQP',
                          bounds=bds,
                          constraints=cons)
    w_g = result.x
    mu_g = w_g * mean_returns
    var_g = np.dot(w_g * cov_matrix, w_g)

    return_df = pd.DataFrame(w_g, columns=['weights'])
    return_df['return'] = mu_g
    return_df['variance'] = var_g
    return_df['ticker'] = stocks
    return return_df


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((len(mean_returns) - 1, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def find_max_sharpe_min_vol(mean_returns, cov_matrix, num_portfolios, risk_free_rate, table):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['allocation_sharpe'])
    max_sharpe_allocation.allocation_sharpe = [round(i * 100, 2) for i in max_sharpe_allocation.allocation_sharpe]
    max_sharpe_allocation['volatility_sharpe'] = round(sdp, 2)
    max_sharpe_allocation['return_sharpe'] = round(rp, 2)

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['allocation_min'])
    min_vol_allocation.allocation_min = [round(i * 100, 2) for i in min_vol_allocation.allocation_min]
    min_vol_allocation['volatility_min'] = round(sdp_min, 2)
    min_vol_allocation['return_min'] = round(rp_min, 2)
    final = max_sharpe_allocation.join(min_vol_allocation)

    return final


def find_percent_returns(df):
    return df.pct_change()\
        .replace([np.inf, -np.inf], np.nan)\
        .fillna(0)


def set_up_stock_data(df):
    df = df.pivot_table('avg_price', 'date', 'ticker')
    df.reset_index(drop=False, inplace=True)
    df = df.fillna(0)\
        .set_index('date')
    return df


def find_avg_price(df, list):
    df['avg_price'] = (df[list[0]] + df[list[1]] + df[list[2]]) / 3
    return df

def find_max_utility(stocks, return_list, price_list, utility):

    def expected_utility(stocks, returns):
        return -sqrt(sum([stocks[x] * returns[x] for x in range(0, len(stocks))]))

    cons = ({'type': 'ineq',
             'fun': lambda stocks: utility - sum([price_list[x] * round(stocks[x], 0) for x in range(0, len(stocks))])})
    bds = tuple([(0, utility) for stock in stocks])
    result = spo.minimize(expected_utility, [5 for stock in stocks], args=return_list, method='SLSQP', bounds=bds,
                          constraints=cons)

    utility = pd.DataFrame(np.round(result.x, 0))
    utility['ticker'] = stocks
    utility.columns = ['amount_utility', 'ticker']
    utility['proportion_utility'] = utility['amount_utility'] / np.sum(utility['amount_utility'])
    utility.index = utility.ticker
    utility = utility[['amount_utility', 'proportion_utility']]
    return utility

def portfolio_already_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std, returns


def find_var(weights, arg_list):
    mean_returns, cov_matrix, risk_free_rate, sharpe = arg_list
    portfolio_std_dev, portfolio_return = portfolio_already_annualised_performance(weights, mean_returns, cov_matrix)
    return portfolio_std_dev


def min_var(stocks, mean_returns, cov_matrix, risk_free_rate, desired_return):
    p = [mean_returns,
         cov_matrix,
         risk_free_rate,
         desired_return,
         False]
    cons = ({'type': 'eq',
             'fun': lambda weight: np.sum(weight) - 1},
            {'type': 'ineq',
             'fun': lambda weight, p: np.sum(weight * p[0]) - p[3],
             'args': (p,)})
    bds = tuple([(0, None) for stock in stocks])
    result = spo.minimize(find_sharpe_o_var, [1 / len(stocks) for stock in stocks],
                          args=(p,),
                          method='SLSQP',
                          bounds=bds,
                          constraints=cons)
    w_g = np.round(result.x, 2)
    mu_g = np.matrix(np.array(w_g)) * np.reshape(np.matrix(mean_returns), (len(mean_returns), 1))
    return_df = pd.DataFrame(w_g, columns=['weights'])
    return_df['return'] = np.array(mu_g)[0][0]
    return_df['variance'] = np.sqrt(np.dot(w_g.T, np.dot(cov_matrix, w_g)))
    return_df.index = stocks
    return_df = return_df[['weights', 'return', 'variance']]

    return return_df
