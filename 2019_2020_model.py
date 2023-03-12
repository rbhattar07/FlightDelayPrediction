import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import max_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier

df_2019 = pd.read_csv('updated_2019.csv')
df_2020 = pd.read_csv('updated_2020.csv')

main_df = pd.concat([df_2019, df_2020])
main_df.to_csv('combined.csv')

# Train & Val Df
train_df, val_df = train_test_split(df_2019, test_size=0.65, random_state=42)
test_df = df_2020.copy()
# Defining input and target columns for train & test set

inputs_cols = list(train_df.columns[1:])
inputs_cols.remove('DEP_DEL15')
target_col = 'DEP_DEL15'

train_inputs = train_df[inputs_cols]
train_target = train_df[target_col]
val_inputs = val_df[inputs_cols]
val_target = val_df[target_col]
test_inputs = test_df[inputs_cols]
test_target = test_df[target_col]

# Validation Set
# Machine Learning Models
## Logistic Regression
print('Logistic Regression')
lr = LogisticRegression(max_iter=1000).fit(train_inputs,train_target)
val_preds = lr.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

## Decision Tree Classifier
print('Decision Tree Classifier')
tree = DecisionTreeClassifier().fit(train_inputs, train_target)
val_preds = tree.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

# Naive Bayes
print('GaussianNB')
gnb = GaussianNB().fit(train_inputs, train_target)
val_preds = gnb.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

print('MultinomialNB')
mnb = MultinomialNB().fit(train_inputs, train_target)
val_preds = mnb.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

# SGD Classifier
print('SGDclasifier')
sgd = SGDClassifier().fit(train_inputs,train_target)
val_preds = sgd.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

# K Nearest neighbor
print('KNN')
knn = KNeighborsClassifier().fit(train_inputs, train_target)
val_preds = knn.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

# Random Forest
print('Random Forest ')
rfc = RandomForestClassifier().fit(train_inputs, train_target)
val_preds = rfc.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

# Gradient Boosting CLassifier
print('Gradient Boosting Classifier')
gbc = GradientBoostingClassifier().fit(train_inputs, train_target)
val_preds = gbc.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

# LGBM
print('LGBM Classifier')
lgbm = LGBMClassifier().fit(train_inputs, train_target)
val_preds = lgbm.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))
print('-'*30)

# XGboost Classifier
print('XGBoost Classifier')
xgb = XGBClassifier().fit(train_inputs, train_target)
val_preds = xgb.predict(val_inputs)
print('AS:',accuracy_score(val_target, val_preds))
print('CM:', confusion_matrix(val_target, val_preds, normalize='true'))

print('-'*30)
print('-'*30)
print('-'*30)


# Test Set

## Decision Tree Classifier
print('Decision Tree Classifier')
test_preds = tree.predict(test_inputs)
print('AS:',accuracy_score(test_target, test_preds))
print('CM:', confusion_matrix(test_target, test_preds, normalize='true'))
print('-'*30)

# Naive Bayes
print('GaussianNB')
test_preds = gnb.predict(test_inputs)
print('AS:',accuracy_score(test_target, test_preds))
print('CM:', confusion_matrix(test_target, test_preds, normalize='true'))
print('-'*30)

# Random Forest
print('Random Forest ')
test_preds = rfc.predict(test_inputs)
print('AS:',accuracy_score(test_target, test_preds))
print('CM:', confusion_matrix(test_target, test_preds, normalize='true'))
print('-'*30)

# Gradient Boosting CLassifier
print('Gradient Boosting Classifier')
test_preds = gbc.predict(test_inputs)
print('AS:',accuracy_score(test_target, test_preds))
print('CM:', confusion_matrix(test_target, test_preds, normalize='true'))
print('-'*30)

# LGBM
print('LGBM Classifier')
test_preds = lgbm.predict(test_inputs)
print('AS:',accuracy_score(test_target, test_preds))
print('CM:', confusion_matrix(test_target, test_preds, normalize='true'))
print('-'*30)

# XGboost Classifier
print('XGBoost Classifier')
test_preds = xgb.predict(test_inputs)
print('AS:',accuracy_score(test_target, test_preds))
print('CM:', confusion_matrix(test_target, test_preds, normalize='true'))
print('-'*30)


models = {
    'tree':tree,
    'gnb':gnb,
    'rfc':rfc,
    'gbc':gbc,
    'lgbm':lgbm,
    'xgb':xgb,
    'input_cols':inputs_cols,
    'target_cols':target_col
}

joblib.dump(models, 'models.joblib')
