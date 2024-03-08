# EX-3 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot.
2. Trace the best fit line and calculate the cost function.
3. Calculate the gradient descent and plot the graph for it.
4. Predict the profit for two population sizes.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DEEPIKA S
RegisterNumber:  212222230028
*/
```
```
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #Calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #Update theto using gradient descent
        theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("/content/50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:
#### data.head()
![intro to m1 1](https://github.com/deepikasrinivasans/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393935/14230684-595b-4f45-8e21-dde74a5c0dc1)

![intro to m1 2](https://github.com/deepikasrinivasans/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393935/0c6b1ee8-a194-4548-9eec-44cffd8a1cbd)
#### Predicted Value
![intro to m1 3](https://github.com/deepikasrinivasans/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393935/2797cf29-0b35-4b01-8864-1a4b356caf4b)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
