# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Collect the dataset of input (e.g., study hours) and output (marks scored).

2.Load and preprocess the data (check missing values, reshape if needed).

3.Train the Simple Linear Regression model using the training data.

4.Use the trained model to predict the marks for new input values. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("/content/score_updated.csv")
display(df.head(10))

# Visualize data
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()

x = df.iloc[:, 0:1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Correcting variable case and generating predictions
print("X_train samples:", X_train.head())
print("Y_train samples:", Y_train.head())

# Generate predictions needed for metrics
y_pred = lr.predict(X_test)

# Regression line plot
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.title('Regression Line')
plt.show()

print("Coefficient:", lr.coef_)
print("Intercept:", lr.intercept_)

# Metrics (fixed function name and defined y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

<img width="197" height="438" alt="image" src="https://github.com/user-attachments/assets/0863fdd6-d335-4a88-adad-2d3678766a5a" />

<img width="862" height="576" alt="image" src="https://github.com/user-attachments/assets/1ceb48ae-370e-4339-857a-95d08c30f3c1" />

<img width="712" height="258" alt="image" src="https://github.com/user-attachments/assets/7d3f79f2-8c53-4101-9a1d-dbd70a6d2b62" />

<img width="922" height="713" alt="image" src="https://github.com/user-attachments/assets/ddbd6775-7ebb-4dc9-bd31-56dc14f761a7" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
