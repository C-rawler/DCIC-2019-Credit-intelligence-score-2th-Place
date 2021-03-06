from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import pandas as pd
import numpy as np
import time
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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

X_train = train.values
X_test = test.values

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
