import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- 1. LOAD DATASET ----------------

data = pd.read_csv("data/tesla_stock.csv")

data["Date"] = pd.to_datetime(data["Date"])

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ---------------- 2. SELECT CLOSE PRICE ----------------

close_prices = data[["Close"]]


# ---------------- 3. SCALE DATA ----------------

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(close_prices)

print("\nData preparation completed successfully!")


# ---------------- 4. CREATE SEQUENCES ----------------

X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

print("\nSequence creation completed successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ---------------- 5. SPLIT DATA ----------------

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nData splitting completed successfully!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==================================================
# LINEAR REGRESSION MODEL
# ==================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# Convert predictions to original stock price scale

linear_predicted_prices = scaler.inverse_transform(
    linear_predictions.reshape(-1, 1)
)

actual_prices = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# Calculate Linear Regression metrics

linear_mae = mean_absolute_error(
    actual_prices,
    linear_predicted_prices
)

linear_mse = mean_squared_error(
    actual_prices,
    linear_predicted_prices
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    actual_prices,
    linear_predicted_prices
)


# ==================================================
# RANDOM FOREST REGRESSION MODEL
# ==================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)


# Convert predictions to original scale

rf_predicted_prices = scaler.inverse_transform(
    rf_predictions.reshape(-1, 1)
)


# Calculate Random Forest metrics

rf_mae = mean_absolute_error(
    actual_prices,
    rf_predicted_prices
)

rf_mse = mean_squared_error(
    actual_prices,
    rf_predicted_prices
)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(
    actual_prices,
    rf_predicted_prices
)


# ==================================================
# MODEL COMPARISON
# ==================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print("\nLinear Regression:")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R² Score:", linear_r2)

print("\nRandom Forest Regression:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)


# ---------------- 6. CREATE COMPARISON TABLE ----------------

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest Regression"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2 Score": [
        linear_r2,
        rf_r2
    ]
})

print("\nComparison Table:")
print(comparison)


# ---------------- 7. SAVE COMPARISON RESULTS ----------------

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison table saved successfully!")


# ---------------- 8. MODEL COMPARISON GRAPH ----------------

plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Stock Price"
)

plt.plot(
    linear_predicted_prices,
    label="Linear Regression Prediction"
)

plt.plot(
    rf_predicted_prices,
    label="Random Forest Prediction"
)

plt.title("Tesla Stock Price Prediction Comparison")

plt.xlabel("Test Data Points")

plt.ylabel("Stock Price")

plt.legend()

plt.savefig(
    "results/model_comparison_graph.png"
)

plt.close()

print("Model comparison graph saved successfully!")


# ---------------- 9. ACTUAL VS PREDICTED GRAPH ----------------

plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Stock Price"
)

plt.plot(
    linear_predicted_prices,
    label="Predicted Stock Price"
)

plt.title("Actual vs Predicted Tesla Stock Prices")

plt.xlabel("Test Data Points")

plt.ylabel("Stock Price")

plt.legend()

plt.savefig(
    "results/actual_vs_predicted.png"
)

plt.close()

print("Actual vs predicted graph saved successfully!")


# ---------------- 10. SAVE BEST MODEL ----------------

joblib.dump(
    linear_model,
    "results/best_stock_prediction_model.pkl"
)

print("\nBest model saved successfully!")


# ---------------- 11. SAVE SCALER ----------------

joblib.dump(
    scaler,
    "results/scaler.pkl"
)

print("Scaler saved successfully!")

print("Selected Model: Linear Regression")


# ---------------- 12. FUTURE STOCK PRICE PREDICTION ----------------

last_60_days = scaled_data[-60:]

future_input = last_60_days.reshape(1, -1)

future_prediction = linear_model.predict(
    future_input
)

future_price = scaler.inverse_transform(
    future_prediction.reshape(-1, 1)
)


print("\n" + "=" * 50)
print("FUTURE STOCK PRICE PREDICTION")
print("=" * 50)

print(
    "Last available date:",
    data["Date"].iloc[-1]
)

print(
    "Predicted next stock price:",
    future_price[0][0]
)