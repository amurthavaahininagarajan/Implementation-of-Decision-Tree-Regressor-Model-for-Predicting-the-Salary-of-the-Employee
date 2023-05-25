# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function. 
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset
7.Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 212222240008
RegisterNumber:  AMURTHA VAAHINI.KN
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
*/
```

## Output:
## Initial dataset:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/0f58e638-9710-4895-853b-7a1d13c62214)
## Data Info:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/8a24b439-c2a3-4320-be50-f4d7f41398aa)
## Optimization of null values:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/674f1b5e-ead4-49a2-bc9a-6958f1e35394)
## Converting string literals to numerical values using label encoder:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/f09cd7df-ab02-4044-84b3-e4c67a2aa7af)
## Assigning x and y values:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/c60bad84-e5a7-4f14-93e6-c539d3266612)
## Mean Squared Error:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/6668e9fa-46b2-44b3-9bbe-6466f9661431)
## R2 (variance):
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/c50e5c92-d785-4a30-b344-64ddcf8cad0d)
## Prediction:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118679102/acc5d9f5-e988-45fa-999c-a7dd0efa5ec5)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
