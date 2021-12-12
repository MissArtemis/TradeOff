import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from entity import Model_daily_reponse
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier



def model_rf_train(df,parameters):
    X = df.drop(['y'], axis=1)
    y = df['y'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # parameters = {
    # 'n_estimators':[150,200],
    # 'max_depth': [15,20]
    # }
    f1 = make_scorer(f1_score , average='macro')
    model = RandomForestClassifier(n_jobs=3,class_weight={0:1, 1:3},random_state=0).fit(X_train,y_train)
    clf = GridSearchCV(model, parameters, n_jobs= 3, cv = 3, scoring=f1)
    clf.fit(X_train, y_train)
    print("best params:",clf.best_params_)
    model = RandomForestClassifier(**clf.best_params_)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rf_model_result = {'model':model, 'accuracy':accuracy, 'f1':f1}
    return rf_model_result



# This is xgb model training function
# Input :  df:DataFrame
# Output: xbg model
# Default name of label column is y
def model_xgb_train(df,parameters):
    X = df.drop(['y'], axis=1)
    y = df['y'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # parameters = {
    #     'n_estimators': [20],
    #     'learning_rate': [0.1],
    #     'max_depth': [25],
    #     'gamma': [0.01],
    # }
    f1 = make_scorer(f1_score , average='macro')
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',tree_method='gpu_hist', gpu_id=0).fit(X_train, y_train)
    clf = GridSearchCV(model, parameters, n_jobs= 3, cv = 3, scoring=f1)
    clf.fit(X_train, y_train)
    print("best params:",clf.best_params_)
    model = xgb.XGBClassifier(**clf.best_params_)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.save_model("./xgb_cat_model.json")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    xgb_model_result = {'model':model, 'accuracy':accuracy, 'f1':f1}

    return xgb_model_result


def feature_engineer(df, selected_cols, delay_term):
    for c in selected_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        for t in range(1, delay_term+1):
            df[c + '_t' + str(t)] = df[c].shift(t)
        
        df[c + '_SMA_5'] = df[c].rolling(5).mean().shift()
        df[c + '_SMA_10'] = df[c].rolling(10).mean().shift()
        df[c + '_SMA_15'] = df[c].rolling(15).mean().shift()
        df[c + '_SMA_20'] = df[c].rolling(20).mean().shift()
        df[c + '_SMA_25'] = df[c].rolling(25).mean().shift()
        df[c + '_SMA_30'] = df[c].rolling(30).mean().shift()
        for i in range(5):
            df[c + '_SMA_5_rate'] = df[c + '_SMA_5'].shift(i) - df[c + '_SMA_5'].shift(i+1)
            df[c + '_SMA_10_rate'] = df[c + '_SMA_10'].shift(i) - df[c + '_SMA_10'].shift(i+1)
            df[c + '_SMA_15_rate'] = df[c + '_SMA_15'].shift(i) - df[c + '_SMA_15'].shift(i+1)
            df[c + '_SMA_20_rate'] = df[c + '_SMA_20'].shift(i) - df[c + '_SMA_20'].shift(i+1)
            df[c + '_SMA_25_rate'] = df[c + '_SMA_25'].shift(i) - df[c + '_SMA_25'].shift(i+1)
            df[c + '_SMA_30_rate'] = df[c + '_SMA_30'].shift(i) - df[c + '_SMA_30'].shift(i+1)
    # print(df)
    return df





