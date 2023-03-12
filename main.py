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

df = pd.read_csv('combined.csv')

models = joblib.load('models.joblib')

input_cols = models['input_cols']
target_cols = models['target_cols']

inputs = df[input_cols]
target = df[target_cols]

#models
tree = models['tree']
gnb = models['gnb']
rfc = models['rfc']
gbc = models['gbc']
lgbm = models['lgbm']
xgb = models['xgb']

# predictions
preds = tree.predict(inputs)
print('AS:',accuracy_score(target, preds))
print('CM:', confusion_matrix(target, preds, normalize='true'))
preds = gnb.predict(inputs)
print('AS:',accuracy_score(target, preds))
print('CM:', confusion_matrix(target, preds, normalize='true'))
preds = rfc.predict(inputs)
print('AS:',accuracy_score(target, preds))
print('CM:', confusion_matrix(target, preds, normalize='true'))
preds = gbc.predict(inputs)
print('AS:',accuracy_score(target, preds))
print('CM:', confusion_matrix(target, preds, normalize='true'))
preds = lgbm.predict(inputs)
print('AS:',accuracy_score(target, preds))
print('CM:', confusion_matrix(target, preds, normalize='true'))
preds = xgb.predict(inputs)
print('AS:',accuracy_score(target, preds))
print('CM:', confusion_matrix(target, preds, normalize='true'))
