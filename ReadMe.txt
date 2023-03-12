This data is collected from the Bureau of Transportation Statistics, Govt. of the USA.
This data is open-sourced under U.S. Govt. Works. This dataset contains all the flights in the month of January 2019
and January 2020.
There are more than 400,000 flights in the month of January itself throughout the United States.
The features were manually chosen to do a primary time series analysis. There are several other features available on
their website.

This data could well be used to predict the flight delay at the destination airport specifically for
the month of January in upcoming years as the data is for January only.

# We have split this machine learning Problem in two part.
- First we will be going through the data of January of 2019 & 2020.
-- We will Perform an EDA, to understand the data better and the relation of the variable with the target column.
-- Then we will understand the feature or variable importance to select which variables/features are important for
the model.
-- Then we will impute the numerical columns if necessary.
-- We will scale the features
-- We will Encode the categorical columns.
-- We will split the data into train, val and test sets.
-- Then we will create a model and get the predictions.

Part 1: (We can also use the 2020 flight data as target to check out predictions using the 2019 data.)
Part 2: Overall we will use the data of 2019 & 2020 to get the predictions for out model.

In 2019 January Data:
- 22 Columns
- 583985 Rows
- Dtypes of columns:
-- 8 Float
-- 8 Integer
-- 6 Object
- Columns with NA/Null Values:
-- Tail_num
-- Dep_Time
-- Dep_Del15
-- ARR_Time
-- ARR_Del15
-- Unnamed: 21

In 2020 January Data:
- 22 Columns
- 607346 Rows
- Dtypes of Columns:
-- 8 Float
-- 8 Integer
-- 6 Object
- Columns with NA/Null Values:
-- Tail_num
-- Dep_Time
-- Dep_Del15
-- ARR_Time
-- ARR_Del15
-- Unnamed: 21


# Aim of the project is to create a fully automated system to predict if the flights would arrive the destination on
time or not?
Target Col: ARR_DEL15

After the validation of the part one of our project:
we have decided to go ahead with these models:
- Logistic Regression
- Decision tree Classifier
- GaussianNB
- KNN
- Gradient Boosting Classifier
- LGBM Classifier
- XGBoost Classifier

After checking the test set of our project 1:
The results were better and it makes sense with the confusion matrix as well.
The models are good to go.

Project 2 is also done.
There can be a scope of making the models better but it seems that everything is already running well.

AS: 0.9273860719462493
CM: [[0.9509193  0.0490807 ]
 [0.20163336 0.79836664]]
AS: 0.9246750215193382
CM: [[0.95153321 0.04846679]
 [0.22257332 0.77742668]]
AS: 0.9552260453077545
CM: [[0.98536955 0.01463045]
 [0.21003382 0.78996618]]
AS: 0.9329472010270925
CM: [[0.96466388 0.03533612]
 [0.24093751 0.75906249]]
AS: 0.9489972374576372
CM: [[0.98226345 0.01773655]
 [0.23338266 0.76661734]]
AS: 0.9521305217592049
CM: [[0.98314119 0.01685881]
 [0.21788353 0.78211647]]
