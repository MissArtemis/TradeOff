from functools import reduce

from numpy import double
import akshare as ak
from pyspark.sql import SparkSession
from entity import Request
from time import sleep
import pandas as pd
from function import model_xgb_train,feature_engineer,model_rf_train
import datetime
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import pickle



def predict_stock_online(stock_list):
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    time = str(year) + str('%02d' % month) + str('%02d' % day)
    print("predict selected stock")
    f = open("D:\\workspace\\TradeOff-1\\model\\stock_offline_rf.pickle","rb")
    model = pickle.load(f)
    result_dict = {}
    for s in stock_list:
        stock = ak.stock_zh_a_hist(symbol=s, period="daily", adjust="qfq")
        selected_cols = ['open', 'high', 'close', 'low', 'volume', 'amount', 'amplitude', 'turnover']
        stock = feature_engineer(df = stock,selected_cols=selected_cols,delay_term=30)
        drop_cols = ['date']
        stock = stock.drop(drop_cols, 1)
        scaler = RobustScaler()
        columns =stock.columns
        stock[columns] = scaler.fit_transform(stock[columns])
        stock['y'] = (stock['close'] - stock['close_t1'])/stock['close']
        stock['y'] = stock['y'].apply(lambda x: '1' if x>=0 else '0')
        stock['y'] = stock['y'].astype(int)
        stock['y'] = stock['y'].shift(-1)
        # stock_now = stock[-2:-1]
        stock_now = stock.tail(1)
        y_pred = model.predict(stock_now)
        result_dict.update({s:y_pred})
    
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv('D:\\workspace\\TradeOff-1\\result\\pred_result' + time +'.csv',mode='w+')

    return result_dict


def recommand_stock_offline():
    print("select stocks from cats!")
    now = datetime.now()
    # now = datetime.now() - timedelta(days=1)
    year = now.year
    month = now.month
    day = now.day
    time = str(year) + str('%02d' % month) + str('%02d' % day)
    cat_rank_df = pd.read_excel('D:\\workspace\\TradeOff-1\\result\\cat_result' + time +'.xlsx')
    top3_cat_list = cat_rank_df['name'].tolist()[0:3]
    print(top3_cat_list)
    cons_stock_list = []
    for c in top3_cat_list:
        stock_board_industry_cons_em_df = ak.stock_board_industry_cons_em(symbol=c)
        cons_stock_list.extend(stock_board_industry_cons_em_df['代码'].tolist())
    stock_today = dict()
    stock_dataframe_list = []
    for s in cons_stock_list:
        print(s)
        try:
            stock = ak.stock_zh_a_hist(symbol=s, period="daily", adjust="qfq")
            stock.rename(columns={u'日期': 'date', u'开盘': 'open', u'最高': 'high', u'最低': 'low', u'收盘': 'close', u'成交量': 'volume', u'成交额': 'amount', u'振幅': 'amplitude', u'换手率': 'turnover'}, inplace=True)
            selected_cols = ['open', 'high', 'close', 'low', 'volume', 'amount', 'amplitude', 'turnover']
            stock = feature_engineer(df = stock,selected_cols=selected_cols,delay_term=30)
            drop_cols = ['date']
            stock = stock.drop(drop_cols, 1)
            scaler = RobustScaler()
            columns =stock.columns
            stock[columns] = scaler.fit_transform(stock[columns])
            stock['y'] = (stock['close'] - stock['close_t1'])/stock['close']
            stock['y'] = stock['y'].apply(lambda x: '1' if x>=0 else '0')
            stock['y'] = stock['y'].astype(int)
            stock['y'] = stock['y'].shift(-1)
            # stock_now = stock[-2:-1]
            stock_now = stock.tail(1)
            stock.dropna(axis=0, inplace=True)
            stock_today.update({s: stock_now})
            stock_dataframe_list.append(stock)
        except:
            continue
    stock_dataframe = pd.concat(stock_dataframe_list, axis=0, ignore_index=True)
    print(stock_dataframe.head(5))
    print("Start Training Model")
    parameters = {
    'n_estimators':[150],
    'max_depth': [20]
    }
    result_dict = model_rf_train(stock_dataframe,parameters)
    model = result_dict['model']
    result_list = []
    for k in stock_today.keys():
        pred_data = stock_today[k].drop(['y'], 1).loc[:].copy()
        pred_data = pred_data.fillna(0)
        pred_t = model.predict_proba(pred_data)
        score = pred_t[0][1]
        result_list.append({'name': k, 'score': score})
    result = pd.DataFrame(result_list)
    result.sort_values('score', axis=0, ascending=False, inplace=True)
    result = result[['name']].head(5)
    try:
        result.to_excel('./result/stock_result' + time +'.xlsx')
    except:
        print("fail to write excel")
        result.to_csv('D:\\workspace\\TradeOff-1\\result\\stock_result' + time +'.csv',mode='w+')
    try:
        f = open("D:\\workspace\\TradeOff-1\\model\\stock_offline_rf.pickle","wb")
        pickle.dump(model,f)
    except:
        print("fail to save model")
    print('Done!')


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

def recommand_cat_offline():
    print('Offline On!!!')
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    time = str(year) + str('%02d' % month) + str('%02d' % day)
    stock_list = ak.stock_board_industry_name_em()['板块名称'].to_list()
    print(stock_list)
    cat_today = dict()
    cat_dataframe_list = []
    print('Getting data from internet')
    for s in stock_list:
        try:
            stock =ak.stock_board_industry_hist_em(symbol=s, adjust="qfq")
            print(s)
            stock.rename(columns={u'日期': 'date', u'开盘': 'open', u'最高': 'high', u'最低': 'low', u'收盘': 'close', u'成交量': 'volume', u'成交额': 'amount', u'振幅': 'amplitude', u'换手率': 'turnover'}, inplace=True)
            selected_cols = ['open', 'high', 'close', 'low', 'volume', 'amount', 'amplitude', 'turnover']
            stock = feature_engineer(df = stock,selected_cols=selected_cols,delay_term=30)
            drop_cols = ['date']
            stock = stock.drop(drop_cols, 1)
            scaler = RobustScaler()
            columns =stock.columns
            stock[columns] = scaler.fit_transform(stock[columns])
            stock['y'] = (stock['close'] - stock['close_t1'])/stock['close']
            stock['y'] = stock['y'].apply(lambda x: '1' if x>=0 else '0')
            stock['y'] = stock['y'].astype(int)
            stock['y'] = stock['y'].shift(-1)
            # stock_now = stock[-2:-1]
            stock_now = stock.tail(1)
            stock.dropna(axis=0, inplace=True)
            cat_today.update({s: stock_now})
            cat_dataframe_list.append(stock)
        except:
            continue
        
    cat_dataframe = pd.concat(cat_dataframe_list, axis=0, ignore_index=True)
    print(cat_dataframe.head(5))
    print("Start Training Model")
    parameters = {
    'n_estimators':[150],
    'max_depth': [20]
    }
    result_dict = model_rf_train(cat_dataframe,parameters)
    model = result_dict['model']
    result_list = []
    for k in cat_today.keys():
        pred_data = cat_today[k].drop(['y'], 1).loc[:].copy()
        pred_data = pred_data.fillna(0)
        # print(pred_data)
        pred_t = model.predict_proba(pred_data)
        # print(pred_t)
        score = pred_t[0][1]
        result_list.append({'name': k, 'score': score})
    result = pd.DataFrame(result_list)
    result.sort_values('score', axis=0, ascending=False, inplace=True)
    try:
        result.to_excel('./result/cat_result' + time +'.xlsx')
    except:
        print("fail to write excel")
        result.to_csv('D:\\workspace\\TradeOff-1\\result\\cat_result' + time +'.csv',mode='w+')
    # spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    # result_table = spark.createDataFrame(result)
    # result_table.write.saveAsTable('industry.xgb_model_stock' + time, mode='overwrite')
    # spark.stop()
    try:
        f = open("D:\\workspace\\TradeOff-1\\model\\cat_offline_rf.pickle","wb")
        pickle.dump(model,f)
    except:
        print("fail to save model")
    print('Done!')


def recommend_stock_online(updatetme:str):
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    result_tb = spark.sql("select * from stock.xgb_model_all" + updatetme).toPandas()
    result_tb.sort_values('score', axis=0, ascending=False, inplace=True)
    result = result_tb
    return {'result': result.to_dict('records')}


if __name__ == "__main__":
    recommand_cat_offline()