from sklearn.model_selection import KFold
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import plotly.offline as py
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
py.init_notebook_mode(connected=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 100)
start = time.time()


def eval_score(preds, dtrain):
    labels = dtrain.get_label()
    mse = mean_squared_error(labels,preds)
    score = 1/(1+mse)
    return 'score', score, True


def get_count(df, column, feature):
    df['idx'] = range(len(df))
    temp = df.groupby(column)['user_id'].agg([(feature, 'count')]).reset_index()
    df = df.merge(temp)
    df = df.sort_values('idx').drop('idx', axis=1).reset_index(drop=True)
    return df


columns = ['user_id', 'real_name', 'age', 'whether_college_students',
           'whether_blacklist_customer', 'whether_4G_unhealthy_customers',
           'user_network_age', 'last_payment_long', 'last_payment_amount',
           'average_consumption_value', 'all_fee', 'balance', 'whether_payment_owed',
           'call_sensitivity', 'number_people_circle', 'whether_often_shopping',
           'average_number_appearance', 'whether_visited_Wanda',
           'whether_visited_member_store', 'whether_watch_movie',
           'whether_attraction', 'whether_stadium_consumption',
           'shopping_app_usage', 'express_app_usage', 'financial_app_usage',
           'video_app_usage', 'aircraft_app_usage', 'train_app_usage',
           'tourism_app_usage', 'label']
boolean_columns = ['whether_college_students', 'whether_blacklist_customer', 'whether_4G_unhealthy_customers',
                   'whether_payment_owed', 'whether_often_shopping', 'whether_visited_Wanda',
                   'whether_visited_member_store', 'whether_watch_movie', 'whether_attraction',
                   'whether_stadium_consumption']

train = pd.read_csv('../data/train_dataset.csv')

test = pd.read_csv('../data/test_dataset.csv')

train.columns = columns
test.columns = columns[:-1]
y_train1 = np.power(1.005, train['label'])
y_train = train.pop('label')
drop_columns = []
data = pd.concat([train, test], axis=0)
data.drop(drop_columns, axis=1, inplace=True)


data['age'] = data['age'].apply(lambda x: np.nan if (x > 100) | (x == 0) else x)
data['all_fee-average_consumption_value'] = data['all_fee'] - data['average_consumption_value']
data['5_all_fee'] = data['average_consumption_value'] * 6 - data['all_fee']
data = get_count(data, 'last_payment_amount', 'count_payment')
data = get_count(data, 'all_fee', 'count_all_fee')
data = get_count(data, 'all_fee-average_consumption_value', 'count_all_fee_diff')
data = get_count(data, 'average_consumption_value', 'count_average_value')
data = get_count(data, ['all_fee', 'average_consumption_value'], 'count_all_fee_average_consumption_value')
data['user_network_age_diff'] = data.apply(lambda x: x.user_network_age % 12, axis=1)


train = data[:train.shape[0]]
test = data[train.shape[0]:]
# 去掉id
stack = pd.DataFrame()
stack['user_id'] = train['user_id']
train.drop(["user_id"], axis=1, inplace=True)
result = pd.DataFrame()
result['id'] = test.pop('user_id')

print(train.shape, test.shape)
# print(train.columns)

X_train = train.values
X_test = test.value

xgb_params = {'eta': 0.01,
              'max_depth': 6,
              'subsample': 0.8,
              'colsample_bytree': 0.7,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': True,
              'nthread': 4,
              'n_estimators': 20000,
              'gamma': 0.2,
              'min_child_weight': 25,
              'num_threads': 8,
              'alpha': 0.18,
              'lambda': 0.23,
              'colsample_bylevel': 0.8,
              }


folds_2 = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))
predictions_xgb = np.zeros(len(test))
oof_lgb1 = np.zeros(len(train))
predictions_lgb1 = np.zeros(len(test))


for fold_, (trn_idx, val_idx) in enumerate(folds_2.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)  # , feval=eval_score)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds_2.n_splits

np.save('../stack/xgb_stacking_train.npy', oof_xgb)
np.save('../stack/xgb_stacking_test.npy', predictions_xgb)


print("xgb MAE score: {:<8.8f}".format(mean_absolute_error(oof_xgb, y_train)))
print("xgb MAE CV score: {:<8.8f}".format(1/(mean_absolute_error(oof_xgb, y_train)+1)))
