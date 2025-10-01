# %% [markdown]
# Import the neccesary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# %% [markdown]
# Data Loading

# %%
try:
    df = pd.read_csv("coin_Tether.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'coin_Tether.csv' was not found")
    exit()


df

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %%
# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)

# %%
# Convert the "date" column to datetime objects
df["date"] = pd.to_datetime(df["date"])
print(df.info())

# %%
# Create a list of all relevant column feature columns
features_cols = ["high","low","open","volume","marketcap"]

# %%
# Add the Ordinal Date as a new feature
df["ordinal_date"] = df["date"].apply(lambda date : date.toordinal())

# %%
# Define the multi-feature matrix (X) by selecting all engineered and direct features
X = df[["ordinal_date"] + features_cols].values

# Y: The target variable remains the close price
y = df["close"].values.reshape(-1,1)

# %%
# Keep track of date labels for final visualization
date_labels = [pd.Timestamp.fromordinal(i) for i in df["ordinal_date"].values]

# %% [markdown]
# Data Splitting

# %%
# Split the data into training (80%) and testing (20%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# Data Scaling

# %%
# Initialize scalers for features and target
X_scaler = StandardScaler()
y_scaler = StandardScaler()

# Example usage to avoid ValueError:
# Always use fit_transform or transform, not the scaler object itself.
# X_train_scaled = X_scaler.fit_transform(X_train)
# X_test_scaled = X_scaler.transform(X_test)
# y_train_scaled = y_scaler.fit_transform(y_train)

# %%
# Fit and transform the training features
X_train_scaled = X_scaler.fit_transform(X_train)

# Transform the test features
X_test_scaled = X_scaler.transform(X_test)

# Fit and transform the target for SVR training (only needed for the target)
y_train_scaled = y_scaler.fit_transform(y_train)

# Dictionary to hold all models results (R_squared,Name)
model_results = {}

# %% [markdown]
# Visualization before training

# %%
plt.figure(figsize=(12,6))
plt.scatter(date_labels,y,s=5,color="gray",label="Actual Close Price")
plt.title("Pre-Training Visualization: Close Price vs Date")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Model Training and Evaluation

# %%
# List of models to compare
regressors = [
    ("Linear Regresson",LinearRegression()),
    ("Ridge Regression",Ridge(alpha=1.0)),
    ("Lasso Regression",Lasso(alpha=0.01)),
    ("ElasticNet",ElasticNet(alpha=0.1,l1_ratio=0.5)),
    ("Decision Tree Regressor",DecisionTreeRegressor(random_state=42)),
    ("Random Forest Regressor",RandomForestRegressor(random_state=42,n_estimators=100)),
    # SVR requires scaled data for optimal performance
    ('Support Vector Regressor (SVR)', SVR(kernel='rbf', C=100))
 ]

# Loop through all regressors to train, predict and evaluate
for name, model in regressors:
    # All models now use scaled features (X_train_scaled) for training
    if name == "Support Vector Regressor (SVR)":
        # SVR uses scaled features (X_train_scaled) and scaled target (y_train_scaled)
        model.fit(X_train_scaled,y_train_scaled.ravel())
        # Predict on scaled test features
        y_pred_scaled = model.predict(X_test_scaled).reshape(-1,1)
        # Inverse transform the prediction back to the original price scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
    else:
        # Other models use scaled features (X_train_scaled) and the original target (y_train)
        model.fit(X_train_scaled,y_train.ravel())
        # Predict using scaled test features
        y_pred = model.predict(X_test_scaled).reshape(-1,1)

    # Calcualte R-squared (Coefficient of Determination)
    r2 = r2_score(y_test,y_pred)

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))

    # Store the results for the final comparison
    model_results[name] = {"R2":r2, "RMSE":rmse, "model":model}

# %%
best_r2 = -np.inf
best_model_name = ""

for name,results in model_results.items():
    print(f"{name}: R-squared = {results['R2']:.4f}, RMSE = {results['RMSE']:.6f}") # Increased RMSE precision

    # Identify the model with the highest R-squared score
    if results["R2"] > best_r2:
        best_r2 = results["R2"]
        best_model_name = name

    print("="* 50)
    print(f"BEST MODEL SELECTED: {best_model_name} (R-squared: {best_r2:.4f})")
    print("="*50)

# %% [markdown]
# Visualization of the Best's Model Performance

# %%
# Find the best model by R2 score
best_model_name = max(model_results, key=lambda k: model_results[k]["R2"])
best_model = model_results[best_model_name]["model"]

# %%
# To visualize the fit over the whole dataset, we predict on all X data points
# First, transform the full dataset X
X_full_scaled = X_scaler.transform(X)

# Predict based on the best model
if best_model_name == "Support Vector Regression":
    # Predict using the scaled entire dataset
    y_pred_full_scaled = best_model.predict(X_full_scaled).reshape(-1,1)
    # Inverse transform to get actual price prediction
    y_pred_full = y_scaler.inverse_transform(X_full_scaled).reshape(-1,1)
else:
    # Predict using scaled entire dataset
    y_pred_full = best_model.predict(X_full_scaled).reshape(-1,1)


plt.figure(figsize=(12,6))
# Plot the actual data points
plt.scatter(date_labels,y,s=5,color="gray",alpha=0.6,label="Actual Close Price")
# Plot the best model's predictions as a line
plt.plot(date_labels,y_pred_full,color="red",linewidth=2,label=f"Predicted Price ({best_model_name})")

plt.title(f"Post-Training Visualization: {best_model_name} Fit vs Actual Data (Multi-Feature)")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# %%



