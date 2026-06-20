import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load dataset
df = pd.read_csv("car data.csv")

# 2. Feature Engineering
df['Car_Age'] = 2026 - df['Year']
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])

# Encode categorical features
for col in ['Fuel_Type','Selling_type','Transmission','Brand']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 3. Define features & target
X = df[['Present_Price','Driven_kms','Fuel_Type','Selling_type',
        'Transmission','Owner','Car_Age','Brand']]
y = df['Selling_Price']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 8. Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()
