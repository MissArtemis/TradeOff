from functools import reduce
import akshare as ak
from pyspark.sql import SparkSession
from entity import Request
from time import sleep
import pandas as pd
from function import model_xgb_train
import datetime


def calculate_target(request: Request):
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    stock = ak.stock_zh_a_hist_min_em(symbol=request.ts_code, period=request.period, adjust='',
                                      start_date=request.start_time, end_date=request.end_time)
    stock_tb = spark.createDataFrame(stock).toDF('time', 'open', 'close', 'high', 'low', 'volumn', 'amount', 'price')
    stock_tb.createOrReplaceTempView('stock_tmp')
    sql = 'select ' + request.target + ' from stock_tmp'
    if len(request.conditions) > 0:
        conditions_sql = reduce(lambda x, y: x + 'and' + y, request.conditions)
        sql = sql + ' where ' + conditions_sql
    result_df = spark.sql(sql)
    result = result_df.toPandas().to_json(orient='columns')
    return result


def recommand_industry_offline():
    print('Offline On!!!')
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    time = str(year) + str('%02d' % month) + str('%02d' % day)
    stock_ind_list = ak.stock_board_industry_name_ths()['name'].to_list()
    # stock_ind_list =[u'传媒']
    print(stock_ind_list)
    ind_today = dict()
    ind_dataframe_list = []
    print('Getting data from internet')
    for s in stock_ind_list:
        try:
            industry = ak.stock_board_industry_index_ths(symbol=s)
        except:
            sleep(50)
            industry = ak.stock_board_industry_index_ths(symbol=s)
        print(s)
        industry.rename(columns={u'日期': 'date', u'开盘价': 'open', u'最高价': 'high', u'最低价': 'low', u'收盘价': 'close', u'成交量': 'volume', u'成交额': 'amount'}, inplace=True)
        industry['close'] = pd.to_numeric(industry['close'], errors='coerce')
        industry['open'] = pd.to_numeric(industry['open'], errors='coerce')
        industry['high'] = pd.to_numeric(industry['high'], errors='coerce')
        industry['low'] = pd.to_numeric(industry['low'], errors='coerce')
        industry['volume'] = pd.to_numeric(industry['volume'], errors='coerce')
        industry['amount'] = pd.to_numeric(industry['amount'], errors='coerce')
        industry['EMA_9'] = industry['close'].ewm(9).mean().shift()
        industry['SMA_5'] = industry['close'].rolling(5).mean().shift()
        industry['SMA_10'] = industry['close'].rolling(10).mean().shift()
        industry['SMA_12'] = industry['close'].rolling(12).mean().shift()
        industry['SMA_15'] = industry['close'].rolling(15).mean().shift()
        industry['SMA_20'] = industry['close'].rolling(20).mean().shift()
        industry['SMA_30'] = industry['close'].rolling(30).mean().shift()
        industry['volume_t1'] = industry['volume'].shift(1)
        industry['volume_t2'] = industry['volume'].shift(2)
        industry['volume_t3'] = industry['volume'].shift(3)
        industry['volume_t4'] = industry['volume'].shift(4)
        industry['volume_t5'] = industry['volume'].shift(5)
        industry['volume_t6'] = industry['volume'].shift(6)
        industry['volume_t7'] = industry['volume'].shift(7)
        industry['volume_t8'] = industry['volume'].shift(8)
        industry['volume_t9'] = industry['volume'].shift(9)
        industry['volume_t10'] = industry['volume'].shift(10)
        industry['close_t1'] = industry['close'].shift(1)
        industry['close_t2'] = industry['close'].shift(2)
        industry['close_t3'] = industry['close'].shift(3)
        industry['close_t4'] = industry['close'].shift(4)
        industry['close_t5'] = industry['close'].shift(5)
        industry['close_t6'] = industry['close'].shift(6)
        industry['close_t7'] = industry['close'].shift(7)
        industry['close_t8'] = industry['close'].shift(8)
        industry['close_t9'] = industry['close'].shift(9)
        industry['close_t10'] = industry['close'].shift(10)
        industry['high_t1'] = industry['high'].shift(1)
        industry['high_t2'] = industry['high'].shift(2)
        industry['high_t3'] = industry['high'].shift(3)
        industry['high_t4'] = industry['high'].shift(4)
        industry['high_t5'] = industry['high'].shift(5)
        industry['high_t6'] = industry['high'].shift(6)
        industry['high_t7'] = industry['high'].shift(7)
        industry['high_t8'] = industry['high'].shift(8)
        industry['high_t9'] = industry['high'].shift(9)
        industry['high_t10'] = industry['high'].shift(10)
        industry['low_t1'] = industry['low'].shift(1)
        industry['low_t2'] = industry['low'].shift(2)
        industry['low_t3'] = industry['low'].shift(3)
        industry['low_t4'] = industry['low'].shift(4)
        industry['low_t5'] = industry['low'].shift(5)
        industry['low_t6'] = industry['low'].shift(6)
        industry['low_t7'] = industry['low'].shift(7)
        industry['low_t8'] = industry['low'].shift(8)
        industry['low_t9'] = industry['low'].shift(9)
        industry['low_t10'] = industry['low'].shift(10)
        industry['amount_t1'] = industry['amount'].shift(1)
        industry['amount_t2'] = industry['amount'].shift(2)
        industry['amount_t3'] = industry['amount'].shift(3)
        industry['amount_t4'] = industry['amount'].shift(4)
        industry['amount_t5'] = industry['amount'].shift(5)
        industry['amount_t6'] = industry['amount'].shift(6)
        industry['amount_t7'] = industry['amount'].shift(7)
        industry['amount_t8'] = industry['amount'].shift(8)
        industry['amount_t9'] = industry['amount'].shift(9)
        industry['amount_t10'] = industry['amount'].shift(10)
        industry['SMA_10_rate_1'] = (industry['SMA_10'] - industry['SMA_10'].shift(1))
        industry['SMA_10_rate_2'] = (industry['SMA_10'].shift(1) - industry['SMA_10'].shift(2))
        industry['SMA_10_rate_3'] = (industry['SMA_10'].shift(2) - industry['SMA_10'].shift(3))
        industry['SMA_10_rate_4'] = (industry['SMA_10'].shift(3) - industry['SMA_10'].shift(4))
        industry['SMA_10_rate_5'] = (industry['SMA_10'].shift(4) - industry['SMA_10'].shift(5))
        industry['SMA_5_rate_1'] = (industry['SMA_5'] - industry['SMA_5'].shift(1))
        industry['SMA_5_rate_2'] = (industry['SMA_5'].shift(1) - industry['SMA_5'].shift(2))
        industry['SMA_5_rate_3'] = (industry['SMA_5'].shift(2) - industry['SMA_5'].shift(3))
        industry['SMA_5_rate_4'] = (industry['SMA_5'].shift(3) - industry['SMA_5'].shift(4))
        industry['SMA_5_rate_5'] = (industry['SMA_5'].shift(4) - industry['SMA_5'].shift(5))
        industry['y'] = (industry['close'] - industry['close_t1']) / industry['close']
        industry['y'] = industry['y'].apply(lambda x: '1' if x >= 0 else '0')
        industry['y'] = industry['y'].astype(int)
        industry['y'] = industry['y'].shift(-1)
        industry_now = industry.tail(1)
        industry.dropna(axis=0, inplace=True)
        drop_cols = ['date']
        industry = industry.drop(drop_cols, 1)
        ind_today.update({s: industry_now})
        ind_dataframe_list.append(industry)
    ind_dataframe = pd.concat(ind_dataframe_list, axis=0, ignore_index=True)
    print("Start Training Model")
    result_dict = model_xgb_train(ind_dataframe)
    model = result_dict['model']
    result_list = []
    for k in ind_today.keys():
        pred_data = ind_today[k].drop(['date', 'y'], 1).loc[:].copy()
        pred_t = model.predict_proba(pred_data)
        # print(pred_t)
        score = pred_t[0][1]
        result_list.append({'name': k, 'score': score})
    result = pd.DataFrame(result_list)
    result.sort_values('score', axis=0, ascending=False, inplace=True)
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    result_table = spark.createDataFrame(result)
    result_table.write.saveAsTable('industry.xgb_model_all' + time, mode='overwrite')
    print('Done!')


def recommend_industry_online(updatetme:str):
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    result_tb = spark.sql("select * from industry.xgb_model_all" + updatetme).toPandas()
    result = result_tb.head(3)
    return {'result': result.to_dict('records')}

