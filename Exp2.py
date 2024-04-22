import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

house_prediction_dataset = pd.read_csv("D:/Coding/Programming/Application_of_ml/Housing.csv")

# Droping unnecessary columns
dropped_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]
house_prediction_dataset.drop(dropped_columns, axis=1, inplace=True)

X = house_prediction_dataset.iloc[:, 1:5].values
y = house_prediction_dataset.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=60)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


ln = LinearRegression()
ln.fit(X_train_scaled, y_train)


y_pred = ln.predict(X_test_scaled)

# Evaluate model
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)
