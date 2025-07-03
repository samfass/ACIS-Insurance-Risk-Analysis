import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

class Statistical_Modelling:
    def __init__(self, data):
        self.data = data

    def split_data(self, X, y, test_size = 0.2):
     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test 


    def model_testing(self, model, X_test, y_test):
      
        y_pred = model.predict(X_test)

        # Measure accuracy
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Output the metrics
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R-squared:", r2)

