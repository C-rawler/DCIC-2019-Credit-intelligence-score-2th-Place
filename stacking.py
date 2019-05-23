from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import time
from catboost import Pool, CatBoostRegressor
import plotly.offline as py
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import BayesianRidge, HuberRegressor
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


def data_process(data):
    transform_value_feature = ['user_network_age', 'age', 'number_people_circle', 'average_number_appearance',
                               'shopping_app_usage', 'express_app_usage', 'financial_app_usage', 'video_app_usage',
                               'aircraft_app_usage', 'train_app_usage', 'tourism_app_usage']
    user_fea = ['last_payment_amount', 'average_consumption_value', 'all_fee', 'balance']
    log_features = ['shopping_app_usage', 'financial_app_usage', 'video_app_usage']
    for col in transform_value_feature + user_fea:
        ulimit = np.percentile(train[col].values, 99.9)
        llimit = np.percentile(train[col].values, 0.1)
        train.loc[train[col] > ulimit, 'col'] = ulimit
        train.loc[train[col] < llimit, 'col'] = llimit
    for col in log_features:
        data[col] = data[col].map(lambda x: np.log(x))
    return data


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

X_train = train.values
X_test = test.values

lgb_param = {'num_leaves': 48,
             'min_data_in_leaf': 50,
             'objective': 'regression_l1',
             'max_depth': 6,
             'learning_rate': 0.01,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.55,
             "bagging_freq": 1,
             "bagging_fraction": 0.8,
             "bagging_seed": 2019,
             "metric": 'mae',
             "num_threads": 8,
             "lambda_l1": 0.5,
             "lambda_l2": 5,
             "verbosity": -1}

lgb_param1 = {'num_leaves': 48,
              'min_data_in_leaf': 50,
              'objective': 'regression',
              'max_depth': 6,
              'learning_rate': 0.01,
              "boosting": "gbdt",
              "feature_fraction": 0.55,
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 2019,
              "metric": 'mae',
              "num_threads": 8,
              "lambda_l1": 0.5,
              "lambda_l2": 5,
              "verbosity": -1}

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


folds_1 = KFold(n_splits=5, shuffle=True, random_state=2019)
folds_2 = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))
predictions_xgb = np.zeros(len(test))
oof_lgb1 = np.zeros(len(train))
predictions_lgb1 = np.zeros(len(test))
for fold_, (trn_idx, val_idx) in enumerate(folds_1.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(lgb_param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds_1.n_splits

    clf1 = lgb.train(lgb_param1, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                     early_stopping_rounds=100)
    oof_lgb1[val_idx] = clf1.predict(X_train[val_idx], num_iteration=clf1.best_iteration)
    predictions_lgb1 += clf1.predict(X_test, num_iteration=clf1.best_iteration) / folds_1.n_splits

np.save('../stack/lgb1_stacking_train.npy', oof_lgb)
np.save('../stack/lgb2_stacking_train.npy', oof_lgb1)
np.save('../stack/lgb1_stacking_test.npy', predictions_lgb)
np.save('../stack/lgb2_stacking_test.npy', predictions_lgb1)

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

print("lgb MAE score: {:<8.8f}".format(mean_absolute_error(oof_lgb, y_train)))
print("lgb MAE CV score: {:<8.8f}".format(1/(mean_absolute_error(oof_lgb, y_train)+1)))
print("xgb MAE score: {:<8.8f}".format(mean_absolute_error(oof_xgb, y_train)))
print("xgb MAE CV score: {:<8.8f}".format(1/(mean_absolute_error(oof_xgb, y_train)+1)))


folds_4 = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_cat = np.zeros(len(train))
predictions_cat = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds_4.split(X_train, y_train1)):
    print("fold n°{}".format(fold_+1))
    cat = CatBoostRegressor(n_estimators=20000,
                            use_best_model=True,
                            learning_rate=0.05,
                            depth=7,
                            bootstrap_type='Bernoulli',
                            l2_leaf_reg=50,
                            subsample=0.5,
                            loss_function='MAE',
                            verbose=1000,
                            early_stopping_rounds=200,
                            random_seed=2019,
                            thread_count=4,
                           )
    trn_data = Pool(X_train[trn_idx], label=y_train1[trn_idx])
    val_data = Pool(X_train[val_idx], label=y_train1[val_idx])
    cat.fit(trn_data, eval_set=val_data)
    oof_cat[val_idx] = cat.predict(X_train[val_idx])
    predictions_cat += cat.predict(X_test)/folds_4.n_splits
print("CV score: {:<8.8f}".format(1 / (1 + mean_absolute_error(np.round(np.log(oof_cat)/np.log(1.005)),
                                                               np.round(np.log(y_train1)/np.log(1.005))))))
oof_cat = np.log(oof_cat)/np.log(1.005)
predictions_cat = np.log(predictions_cat)/np.log(1.005)
np.save('../stack/cat_stacking_train1.npy', oof_cat)
np.save('../stack/cat_stacking_test1.npy', predictions_cat)

# stacking
train_stack = np.vstack([oof_lgb, oof_lgb1, oof_xgb, oof_cat]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_lgb1, predictions_xgb, predictions_cat]).transpose()
folds_stack = StratifiedKFold(n_splits=10, shuffle=True, random_state=8888)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    print("fold :", fold_ + 1)
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    stacking = HuberRegressor(epsilon=1.03, alpha=1e-5)

    stacking.fit(trn_data, trn_y)
    oof_stack[val_idx] = stacking.predict(val_data)
    predictions += stacking.predict(test_stack) / folds_stack.n_splits


print("stacking MAE score: {:<8.8f}".format(mean_absolute_error(oof_stack, y_train)))
print("stacking CV score: {:<8.8f}".format(1/(mean_absolute_error(oof_stack, y_train)+1)))


print(predictions_lgb.mean(), predictions_lgb1.mean(), y_train.mean(), predictions.mean())
result['score'] = predictions
result['score'] = round(result['score']).map(int)
result.to_csv('../result/stacking.csv', index=None)

