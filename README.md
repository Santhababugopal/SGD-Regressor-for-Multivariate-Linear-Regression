# SGD-Regressor-for-Multivariate-Linear-Regression
NAME :   SANTHABABU  G


REGISTER NUMBER: 212224040292

## AIM:


To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:


1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Load and Prepare Dataset Load California housing dataset using fetch_california_housing().
Convert the data to a DataFrame.

Add HousingPrice (target) as a new column.

2.Define Features (x) and Targets (y) Remove AveOccup and HousingPrice from features — keep them as targets.
So:

x = all features except AveOccup and HousingPrice

y = AveOccup and HousingPrice (multi-output regression)

3.Train-Test Split Split the data into 80% training and 20% testing using train_test_split().

4.Feature Scaling Use StandardScaler to normalize:

x_train and x_test (feature scaling)

y_train and y_test (target scaling — crucial for SGD)

5.Train Multi-Output Regressor Create SGDRegressor model for linear regression using stochastic gradient descent.
Wrap it in MultiOutputRegressor to handle multiple target variables.

6.Train the model using scaled x_train and y_train.

7.Make Predictions Predict on x_test, getting scaled predictions for both outputs.
Inverse transform the predictions and actual values using scaler_y to return to original scale.

8.Evaluate the Model Compute MSE between actual and predicted values (for both outputs).
Print the MSE and show the first 5 predictions.. 

## Program:
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset= fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

x=df.drop(columns=['AveOccup','HousingPrice'])
y=df[['AveOccup','HousingPrice']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred=multi_output_sgd.predict(x_test)

y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)

mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])


```

## Output:
<img width="375" height="193" alt="image" src="https://github.com/user-attachments/assets/4dcf139f-fa80-4691-8a5f-05a0039b7008" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
