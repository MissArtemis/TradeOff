import akshare as ak
from entity import Request
from pyspark.sql import SparkSession
from functools import reduce
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from entity import Model_daily_reponse


def calculate_target(request: Request):
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    stock = ak.stock_zh_a_hist_min_em(symbol=request.ts_code, period=request.period, adjust='',
                                      start_date=request.start_time, end_date=request.end_time)
    stock_tb = spark.createDataFrame(stock).toDF('time', 'open', 'close', 'high', 'low', 'volumn', 'amount', 'price')
    stock_tb.registerTempTable('stock_tmp')
    sql = 'select ' + request.target + ' from stock_tmp'
    if len(request.conditions) > 0:
        conditions_sql = reduce(lambda x, y: x + 'and' + y, request.conditions)
        sql = sql + ' where ' + conditions_sql
    result_df = spark.sql(sql)
    result = result_df.toPandas().to_json(orient='columns')
    return result


def model_xgb_daily(ts_code: str, start_time: str, end_time: str):
    stock = ak.stock_zh_a_hist(symbol=ts_code, period="daily", adjust='qfq', start_date=start_time, end_date=end_time)
    stock.rename(columns={u'日期': 'date', u'开盘': 'open', u'收盘': 'close', u'最高': 'high', u'最低': 'low', u'成交量': 'volumn',
                          u'成交额': 'amount', u'振幅': 'amplitude', u'涨跌幅': 'chg_pct', u'涨跌额': 'chg',
                          u'换手率': 'turnover_pct'}, inplace=True)
    stock['EMA_9'] = stock['close'].ewm(9).mean().shift()
    stock['SMA_5'] = stock['close'].rolling(5).mean().shift()
    stock['SMA_10'] = stock['close'].rolling(10).mean().shift()
    stock['SMA_20'] = stock['close'].rolling(20).mean().shift()
    stock['SMA_30'] = stock['close'].rolling(30).mean().shift()
    stock['y'] = np.select([stock['chg_pct'] > 0, stock['chg_pct'] <= 0], [1, 0])
    stock['y'] = stock['y'].astype(int)
    stock['y'] = stock['y'].shift(-1)
    stock['volumn_t1'] = stock['volumn'].shift(1)
    stock['volumn_t2'] = stock['volumn'].shift(2)
    stock['volumn_t3'] = stock['volumn'].shift(3)
    stock['volumn_t4'] = stock['volumn'].shift(4)
    stock['volumn_t5'] = stock['volumn'].shift(5)
    stock['volumn_t6'] = stock['volumn'].shift(6)
    stock['volumn_t7'] = stock['volumn'].shift(7)
    stock['volumn_t8'] = stock['volumn'].shift(8)
    stock['volumn_t9'] = stock['volumn'].shift(9)
    stock['volumn_t10'] = stock['volumn'].shift(10)
    stock['close_t1'] = stock['close'].shift(1)
    stock['close_t2'] = stock['close'].shift(2)
    stock['close_t3'] = stock['close'].shift(3)
    stock['close_t4'] = stock['close'].shift(4)
    stock['close_t5'] = stock['close'].shift(5)
    stock['close_t6'] = stock['close'].shift(6)
    stock['close_t7'] = stock['close'].shift(7)
    stock['close_t8'] = stock['close'].shift(8)
    stock['close_t9'] = stock['close'].shift(9)
    stock['close_t10'] = stock['close'].shift(10)
    stock['high_t1'] = stock['high'].shift(1)
    stock['high_t2'] = stock['high'].shift(2)
    stock['high_t3'] = stock['high'].shift(3)
    stock['high_t4'] = stock['high'].shift(4)
    stock['high_t5'] = stock['high'].shift(5)
    stock['high_t6'] = stock['high'].shift(6)
    stock['high_t7'] = stock['high'].shift(7)
    stock['high_t8'] = stock['high'].shift(8)
    stock['high_t9'] = stock['high'].shift(9)
    stock['high_t10'] = stock['high'].shift(10)
    stock['low_t1'] = stock['low'].shift(1)
    stock['low_t2'] = stock['low'].shift(2)
    stock['low_t3'] = stock['low'].shift(3)
    stock['low_t4'] = stock['low'].shift(4)
    stock['low_t5'] = stock['low'].shift(5)
    stock['low_t6'] = stock['low'].shift(6)
    stock['low_t7'] = stock['low'].shift(7)
    stock['low_t8'] = stock['low'].shift(8)
    stock['low_t9'] = stock['low'].shift(9)
    stock['low_t10'] = stock['low'].shift(10)
    stock_now = stock.tail(1)
    stock.dropna(axis=0, inplace=True)
    stock['y'] = LabelEncoder().fit_transform(stock['y'])
    drop_cols = ['date']
    stock = stock.drop(drop_cols, 1)
    valid_size = 0.3
    valid_split_idx = int(stock.shape[0] * (1 - (valid_size)))
    train_df = stock.loc[:valid_split_idx].copy()
    valid_df = stock.loc[valid_split_idx + 1:].copy()
    y_train = train_df['y'].copy()
    X_train = train_df.drop(['y'], 1)
    y_valid = valid_df['y'].copy()
    X_valid = valid_df.drop(['y'], 1)
    parameters = {
        'n_estimators': [1, 2, 3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.02, 0.005, 0.007],
        'max_depth': [1, 2, 3, 4, 5, 8, 10, 12],
        'gamma': [0.01, 0.005, 0.007, 0.012, 0.015],
    }
    model = xgb.XGBClassifier(use_label_encoder=False).fit(X_train, y_train)
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    model = xgb.XGBClassifier(**clf.best_params_)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)
    now = stock_now.drop(['date', 'y'], 1).loc[:].copy()
    y_now = model.predict(now)[0]
    response = Model_daily_reponse(ts_code=ts_code, date=end_time, accuracy=str(accuracy), f1=str(f1), y_now=str(y_now))
    return response
