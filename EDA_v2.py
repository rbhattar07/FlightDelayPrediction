import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ydata_profiling import profile_report as pr
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(10,8)
matplotlib.rcParams['figure.facecolor']='#00000000'

df = pd.read_csv('Jan_2020_ontime.csv')

print(df.info())
print(df.describe())
print(df.head(25))
print(df.tail(25))
print(df.isna().sum())
print(df.columns)

categorical_cols = df.select_dtypes('object').columns.tolist()
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

df.drop(columns='Unnamed: 21', inplace=True)

categorical_cols = df.select_dtypes('object').columns.tolist()
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

def vc(cols, df):
    for i in cols:
        print()
        print(i)
        print(df[i].unique().tolist())
        print(df[i].value_counts().tolist())

vc(categorical_cols, df)

def hist1(df, columns):
    for i in columns:
        print(i)
        fig = px.histogram(df, x=i, marginal='box', nbins= 100, title=i)
        fig.update_layout(bargap=0.1)
        fig.show()

hist1(df, categorical_cols)

def scatter1(df, columns):
    for i in columns:
        fig = px.scatter(df, x=i, y='DEP_TIME',color='DAY_OF_WEEK', opacity=0.5, hover_data=['DAY_OF_MONTH', 'ARR_TIME', 'ARR_DEL15'],
                         title=i)
        fig.update_traces(marker_size=10)
        fig.show()

scatter1(df, numerical_cols)
