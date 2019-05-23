import os
from sklearn.model_selection import KFold,StratifiedKFold
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)

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

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

train.columns = columns
test.columns = columns[:-1]
stack = pd.DataFrame()
stack['user_id'] = train['user_id']
y_train = train.pop('label')
drop_columns = []
data = pd.concat([train, test], axis=0)

data['age'] = data['age'].apply(lambda x: np.nan if (x > 100) | (x == 0) else x)
data['all_fee-average_consumption_value'] = data['all_fee'] - data['average_consumption_value']
data['5_all_fee'] = data['average_consumption_value'] * 6 - data['all_fee']
data = get_count(data, 'last_payment_amount', 'count_payment')
data = get_count(data, 'all_fee', 'count_all_fee')
data = get_count(data, 'all_fee-average_consumption_value', 'count_all_fee_diff')
data = get_count(data, 'average_consumption_value', 'count_average_value')
data = get_count(data, ['all_fee', 'average_consumption_value'], 'count_all_fee_average_consumption_value')
data['user_network_age_diff'] = data.apply(lambda x: x.user_network_age % 12, axis=1)


data.drop(drop_columns, axis=1, inplace=True)
train = data[:train.shape[0]]
test = data[train.shape[0]:]
train.drop(["user_id"], axis=1, inplace=True)
result = pd.DataFrame()
result['id'] = test.pop('user_id')

print(train.shape, test.shape)

X_train = train.values
X_test = test.values

param = {'num_leaves': 48,
         'min_data_in_leaf': 50,
         'objective': 'regression',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.55,
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
         "seed": 8888,
         "metric": 'mae',
         "lambda_l1": 0.5,
         "lambda_l2": 5,
         "verbosity": -1}

# "seed": 8888
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=8888)
# idx = y_train.argsort()
# y_lab = np.repeat(list(range(50000 // 20)), 20)
# y_lab = np.asarray(sorted(list(zip(idx, y_lab))))[:, -1].astype(np.int32)
# splits = folds.split(X_train, y_lab)

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
splits = folds.split(X_train, y_train)

for fold_, (trn_idx, val_idx) in enumerate(splits):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 20000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits


print("MAE CV score: {:<8.8f}".format(1/(mean_absolute_error(oof_lgb, y_train)+1)))
print(predictions_lgb)

np.save('val.mse_lgb.npy',oof_lgb)
np.save('test.mse_lgb.npy',predictions_lgb)
