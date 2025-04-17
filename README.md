# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S Rajath
RegisterNumber: 212224240127
*/
```

```py
import numpy as np
import pandas as pd

dataset=pd.read_csv('Placement_Data.csv')
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,Y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 -h))

def gradient_descent(theta,X,Y,alpha,num_iteration):
    m=  len(y)
    for i in range(num_iteration):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X,Y,alpha=0.01,num_iteration=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    
    y_pred= np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)

accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy",accuracy)

print("Y_pred=",y_pred)
print("Y=",Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew= predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew= predict(theta,xnew)
print(y_prednew)


```

## Output:

### Dataset
![image](https://github.com/user-attachments/assets/3726d7f1-d86e-49c1-9696-039c7991d56e)

![image](https://github.com/user-attachments/assets/311d2586-801b-4233-834e-d2b63787ca92)

![image](https://github.com/user-attachments/assets/2ff07a31-289c-4f28-869b-9cec039571c5)

### Accuracy
![image](https://github.com/user-attachments/assets/f53d4b08-37d1-40e8-8f2f-95592a2593da)

![image](https://github.com/user-attachments/assets/00795d1d-d4f7-4307-9d7d-d08c4f40bd91)

![image](https://github.com/user-attachments/assets/e9077dbb-b9c1-494b-91fd-1bafdd188f77)

![image](https://github.com/user-attachments/assets/69be4f51-143e-42de-863e-0d0549ef75b3)

![image](https://github.com/user-attachments/assets/4a8fb925-6c8c-4236-8b79-2acbcb8ac401)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

