import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import max_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('Jan_2020_ontime.csv')

df.drop(columns=['OP_UNIQUE_CARRIER','OP_CARRIER', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_SEQ_ID','ORIGIN',
                 'DEST_AIRPORT_SEQ_ID', 'DEST', 'TAIL_NUM'], inplace=True)

# Dealing with Missing Data
df.drop(columns=['Unnamed: 21','CANCELLED', 'DIVERTED'], inplace=True)

print(df[['DEP_TIME', 'DEP_DEL15', 'ARR_TIME', 'ARR_DEL15']])
print(df[['DEP_TIME', 'DEP_DEL15', 'ARR_TIME', 'ARR_DEL15']])
print(df.isna().sum())
df.dropna(inplace=True)

# Creating Numerical & Categorical COlumns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# Scaling Numeric Features
scaler = StandardScaler().fit(df[numerical_cols])

# Encoding Categorical Columns
dtb = {
    '0001-0559':1,
    '0600-0659':2,
    '0700-0759':3,
    '0800-0859':4,
    '0900-0959':5,
    '1000-1059':6,
    '1100-1159':7,
    '1200-1259':8,
    '1300-1359':9,
    '1400-1459':10,
    '1500-1559':11,
    '1600-1659':12,
    '1700-1759':13,
    '1800-1859':14,
    '1900-1959':15,
    '2000-2059':16,
    '2100-2159':17,
    '2200-2259':18,
    '2300-2359':19,
}

df['DEP_TIME_BLK']= df['DEP_TIME_BLK'].map(dtb)

# Creating Numerical & Categorical COlumns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# Scaling Numeric Features
scaler = StandardScaler().fit(df[numerical_cols])
print(df)
df.to_csv('updated_2020.csv')
