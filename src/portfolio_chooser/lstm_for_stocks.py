import os
import argparse
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,TimeDistributed,Flatten,Masking
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext, SQLContext
import pandas_datareader.data as web
import pyspark.sql.functions as func
import util.util as util
import h5py


class LstmStockTrainer:
    def __init__(self, data, length):
        self.data = data
        self.length = length

    @staticmethod
    def buildLstmModel(x_train, y_train, x_valid, y_valid, epochs):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       min_delta=0, verbose=2, mode='auto')
        model = Sequential()
        # model.add(Masking(mask_value=0.0, input_shape=(89,124)))
        model.add(LSTM(256, input_shape=(89, 125), activation='sigmoid', return_sequences=True))
        model.add(Dropout(0.10))
        model.add(LSTM(145, activation='tanh', return_sequences=True))
        model.add(Dropout(0.10))
        model.add(LSTM(95, activation='tanh', return_sequences=True))
        model.add(Dropout(0.10))
        # model.add(Flatten())
        model.add(Dense(5, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid), shuffle=False)
        return model

    def train_and_predict_model(self, stock, data, start_date, end_date, epochs, iterative):
        df_stock, diff_df_matrix, last_date = util.create_stock_data(stock, data, iterative)
        x_train, y_train, x_valid, y_valid, x_test, y_test, min_max_scaler_x, min_max_scaler_y = util.create_datasets(
            diff_df_matrix, last_date, iterative)
        model = self.buildLstmModel(x_train, y_train, x_valid, y_valid, epochs)
        # model.save('%s_model_m2m_3.h5' % stock)

        last_predict = df_stock[(len(df_stock) - 1) - 89:len(df_stock)]
        if not iterative:
            last_predict.index = last_predict.index.droplevel(0)
        last_predict_diff = last_predict.diff().dropna()

        last_predict_matrix = last_predict_diff.as_matrix()
        last_predict_norm = min_max_scaler_x.transform(last_predict_matrix)
        x_norm = np.reshape(util.add_date_features(last_predict_diff.index[0], last_predict_norm), (1, 89, 125))

        last_predict_y = np.reshape(model.predict(x_norm), (89, 5))
        reverse_scale_predict = min_max_scaler_y.inverse_transform(last_predict_y)
        reverse_predict = pd.DataFrame(reverse_scale_predict)
        reverse_predict.columns = ['open', 'high', 'low', 'close', 'volume']
        inverse_df_predict = util.inverse_df(last_predict[89:90], reverse_predict,
                                             ['open', 'high', 'low', 'close', 'volume'])

        #inverse_df_predict.index = pd.date_range(start=pd.datetime(2018, 4, 13), end=pd.datetime(2018, 7, 10))
        inverse_df_predict.index = pd.date_range(start=start_date, end=end_date)
        try:
            df_stock.index = df_stock.index.droplevel(0)
        except:
            pass
        add_prediction = inverse_df_predict.append(df_stock[['open', 'high', 'low', 'close', 'volume']])
        add_prediction = add_prediction.sort_index()
        return add_prediction

    def predict_stocks(self, stock):
        print('starting %s' % stock)
        add_prediction = self.train_and_predict_model(stock, self.data,
                                                      pd.datetime(2018, 4, 13), pd.datetime(2018, 7, 10),
                                                      1, False)

        print('starting next prediction')
        print(add_prediction.tail())

        next_prediction = self.train_and_predict_model(stock, add_prediction,
                                                      pd.datetime(2018, 7, 11), pd.datetime(2018, 10, 7),
                                                      1, True)

        next_prediction['date'] = pd.to_datetime(next_prediction.index)
        next_prediction['date'] = next_prediction['date'].apply(lambda row: row.strftime('%Y-%m-%d'))
        next_prediction['ticker'] = stock
        next_prediction = next_prediction.reset_index(drop=True)
        print(next_prediction.tail())
        return next_prediction


def pandas_df_to_spark_schema(df):
    from pyspark.sql.types import _infer_schema, _merge_type
    from functools import reduce
    cols = [str(x) for x in df.columns]
    df = [r.tolist() for r in df.to_records(index=False)]
    struct = reduce(_merge_type, map(_infer_schema, df))
    for i, name in enumerate(cols):
        struct.fields[i].name = name
        struct.names[i] = name
    return struct


def main(stock_list, seq_len, result_table):
    os.environ['PYSPARK_PYTHON'] = '/Users/lex/miniconda2/envs/pysparkenv2/bin/python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/lex/miniconda2/envs/pysparkenv2/bin/python'

    fields = [StructField('open', FloatType(), True),
              StructField('high', FloatType(), True),
              StructField('low', FloatType(), True),
              StructField('close', FloatType(), True),
              StructField('volume', FloatType(), True),
              StructField('date', StringType(), True),
              StructField('ticker', StringType(), True),
              ]
    schema = StructType(fields)
    stock_list = stock_list.split(',')


    stock_data = pd.DataFrame()
    print('Predicting %s stocks' % len(stock_list))
    for x in stock_list:
        df = web.DataReader(x, 'morningstar', pd.datetime(2013,4,13), pd.datetime(2018,4,13))
        stock_data = stock_data.append(df)
        print('Have the data')

    len_comb = len(stock_list)

    seq_len = int(seq_len)

    lstm = LstmStockTrainer(stock_data, seq_len)

    keys = stock_list

    conf = SparkConf().setAppName('portfolio_chooser')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    keys = sc.parallelize(keys)
    task_rdd = keys.map(lambda stock: lstm.predict_stocks(stock)) \
                .repartition(len_comb)
    result_rdd = task_rdd \
        .flatMap(lambda r: r.values) \
        .map(lambda r: tuple(r))

    result_df = sqlContext.createDataFrame(result_rdd, schema)
    # replacing all pandas NaNs to null
    cols = [func.when(~func.col(x).isin("NaN"), func.col(x)).alias(x) for x in result_df.columns]
    result_df = result_df.select(*cols)
    result_df.show(5)
    util.write_small_df(result_df, result_table)
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--run_locally', default='True')
    parser.add_argument('--stock_list', default='FB')
    parser.add_argument('--seq_len', default='90')
    parser.add_argument('--result_table')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
