import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Documents\MY_WORKSPACE\02_SCHOLARSHIPS\03_AI_ML_TECHYJAUNT\04_module-supervised-regression\lectures\data.csv")

# Drop yr_renovated
df = df.drop('yr_renovated', axis=1)

# Convert data to datatime
df['date'] = pd.to_datetime(df['date'])

# Drop more columns
cols_to_drop = ['sqft_lot', 'sqft_above', 'sqft_basement', 'street', 
               'city', 'statezip', 'country', 'date']

df_cleaned = df.drop(columns=cols_to_drop)
print(df_cleaned.columns)

# Drpp rows with price <= 0
df_cleaned_final = df_cleaned[df_cleaned['price'] > 0]

# Features and target
X = df_cleaned_final.drop('price', axis=1)
y = df_cleaned_final['price']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train_test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plotting Actual vs Predicted
plt.figure(figsize=(8,6))

sns.scatterplot(x=y_test, y=y_pred)
plt.plot(
    [y.min(), y.max()], 
    [y.min(), y.max()],
    color='red', linestyle='--'    # Line for perfect predictions 
) 


plt.xlabel('Actual Prices') 
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
