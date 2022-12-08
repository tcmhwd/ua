import pandas as pd
import numpy as np
import pickle
import sklearn
from xgboost import XGBClassifier

ua_s = pd.read_csv('uaS.csv', encoding='cp949')
ua_s = ua_s[['Bacteria.q','WBC.q','Specific.Gravity.ua','WBC.max','Monocyte.max','Lymphocyte.max','CRP.max','dbp.max','sbp.max','age','UTI']]
ua_s = ua_s.fillna(ua_s.median())

train_y_ua = ua_s['UTI']
train_x_ua = ua_s.drop(['UTI'], axis=1)



np.random.seed(21)

if "Set" not in ua_s.columns:
  ua_s["Set"] = np.random.choice(["train","test"], p = [.8, .2], size=(ua_s.shape[0],))

train_indices = train_x_ua[ua_s.Set=="train"].index
test_indices = train_x_ua[ua_s.Set=="test"].index

nunique = ua_s.nunique()
types = ua_s.dtypes

categorical_columns = []
categorical_dims =  {}

for col in train_x_ua.columns:
    if types[col] == 'object'or nunique[col] < 50:
        print(col, train_x_ua[col].nunique())
        l_enc = LabelEncoder()
        train_x_ua[col] = train_x_ua[col].fillna('VV_likely')
        train_x_ua[col] = l_enc.fit_transform(train_x_ua[col].astype(str))
        train_x_ua[col] = l_enc.fit_transform(train_x_ua[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        training_mean = train_x_ua.loc[train_indices,col].mean()
        ua_s.fillna(training_mean, inplace=True)

# Categorical Embedding을 위해 Categorical 변수의 차원과 idxs를 담음.

features = [ col for col in train_x_ua.columns]
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

train_input = train_x_ua[features].values[train_indices]
train_target = train_y_ua.values[train_indices]

test_input = train_x_ua[features].values[test_indices]
test_target = train_y_ua.values[test_indices]

## xgboost model
uti = XGBClassifier(max_depth=8,
    tree_method = "hist",
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,)

uti.fit(train_input, train_target,
        eval_set=[(test_input, test_target)],
        verbose=10)


pickle.dump(uti, open('uti.pkl','wb'))
uti= pickle.load(open('uti.pkl','rb'))

##bsi model
ua_s = pd.read_csv('uaS.csv', encoding='cp949')
ua_s=ua_s[['Bacteria.q','WBC.q','Specific.Gravity.ua','WBC.max','Monocyte.max','Lymphocyte.max','CRP.max','dbp.max','sbp.max','age','BSI']]
ua_s = ua_s.fillna(ua_s.median())

train_y_ua=ua_s['BSI']
train_x_ua=ua_s.drop(['BSI'], axis=1)

np.random.seed(21)

if "Set" not in ua_s.columns:
  ua_s["Set"] = np.random.choice(["train","test"], p = [.8, .2], size=(ua_s.shape[0],))

train_indices = train_x_ua[ua_s.Set=="train"].index
test_indices = train_x_ua[ua_s.Set=="test"].index

nunique = ua_s.nunique()
types = ua_s.dtypes

categorical_columns = []
categorical_dims =  {}

for col in train_x_ua.columns:
    if types[col] == 'object'or nunique[col] < 50:
        print(col, train_x_ua[col].nunique())
        l_enc = LabelEncoder()
        train_x_ua[col] = train_x_ua[col].fillna('VV_likely')
        train_x_ua[col] = l_enc.fit_transform(train_x_ua[col].astype(str))
        train_x_ua[col] = l_enc.fit_transform(train_x_ua[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        training_mean = train_x_ua.loc[train_indices,col].mean()
        ua_s.fillna(training_mean, inplace=True)

# Categorical Embedding을 위해 Categorical 변수의 차원과 idxs를 담음.

features = [ col for col in train_x_ua.columns]
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

train_input = train_x_ua[features].values[train_indices]
train_target = train_y_ua.values[train_indices]

test_input = train_x_ua[features].values[test_indices]
test_target = train_y_ua.values[test_indices]

from xgboost import XGBClassifier

bsi = XGBClassifier(max_depth=8,
    tree_method = "hist",
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,)

bsi.fit(train_input, train_target,
        eval_set=[(test_input, test_target)],
        verbose=10)

pickle.dump(bsi, open('bsi.pkl','wb'))
bsi= pickle.load(open('bsi.pkl','rb'))
