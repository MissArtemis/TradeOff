import akshare as ak
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from entity import Model_daily_reponse
from sklearn.model_selection import train_test_split


# This is xgb model training function
# Input :  df:DataFrame
# Output: xbg model
# Default name of label column is y
def model_xgb_train(df):
    X = df.drop(['y'], axis=1)
    y = df['y'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    parameters = {
        'n_estimators': [8, 9, 10],
        'learning_rate': [0.02, 0.01, 0.025],
        'max_depth': [8, 9, 10],
        'gamma': [0.02, 0.01, 0.025],
    }
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    model = xgb.XGBClassifier(**clf.best_params_)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    xgb_model_result = {'model':model, 'accuracy':accuracy, 'f1':f1}
    return xgb_model_result




