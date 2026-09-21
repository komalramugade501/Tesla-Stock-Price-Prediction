import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "tesla_stock.csv"
MODEL_PATH = BASE_DIR / "results" / "best_stock_prediction_model.pkl"
SCALER_PATH = BASE_DIR / "results" / "scaler.pkl"
COMPARISON_PATH = BASE_DIR / "results" / "model_comparison.csv"
COMPARISON_GRAPH_PATH = BASE_DIR / "results" / "model_comparison_graph.png"
ACTUAL_PREDICTED_GRAPH_PATH = BASE_DIR / "results" / "actual_vs_predicted.png"


# ==================================================
# LOAD MODEL AND SCALER
# ==================================================

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

except Exception as e:
    st.error("Unable to load the trained model or scaler.")
    st.error(f"Error: {e}")
    st.stop()


# ==================================================
# LOAD DATASET
# ==================================================

try:
    data = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error("Unable to load the Tesla stock dataset.")
    st.error(f"Error: {e}")
    st.stop()


# ==================================================
# DATA VALIDATION
# ==================================================

if "Close" not in data.columns:
    st.error("The dataset does not contain a 'Close' column.")
    st.stop()

if len(data) < 60:
    st.error("At least 60 closing-price records are required.")
    st.stop()


# ==================================================
# MAIN TITLE
# ==================================================

st.title("📈 Tesla Stock Price Prediction System")

st.write(
    "A Machine Learning application that uses historical "
    "Tesla stock closing prices to predict the next closing price."
)

st.info(
    "The prediction is a machine-learning estimate based on historical "
    "data and should not be considered a guaranteed future stock price."
)


# ==================================================
# DATASET SECTION
# ==================================================

st.header("Tesla Stock Dataset")

st.write(
    "The application uses historical Tesla stock market data "
    "for model training and prediction."
)

st.dataframe(
    data.head(10),
    width="stretch"
)


# ==================================================
# DATASET INFORMATION
# ==================================================

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Rows",
        f"{data.shape[0]:,}"
    )

with col2:
    st.metric(
        "Total Columns",
        data.shape[1]
    )

with col3:
    st.metric(
        "Prediction Window",
        "60 Days"
    )


# ==================================================
# DATASET DATE INFORMATION
# ==================================================

if "Date" in data.columns:

    try:
        dates = pd.to_datetime(data["Date"])

        first_date = dates.min().strftime("%d-%b-%Y")
        last_date = dates.max().strftime("%d-%b-%Y")

        st.caption(
            f"Dataset period: {first_date} to {last_date}"
        )

    except Exception:
        pass


st.divider()


# ==================================================
# MODEL PERFORMANCE COMPARISON
# ==================================================

st.header("Model Performance Comparison")

try:

    comparison_df = pd.read_csv(COMPARISON_PATH)

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True
    )

except Exception as e:

    st.warning(
        f"Model comparison file could not be loaded: {e}"
    )


# ==================================================
# MODEL PERFORMANCE GRAPH
# ==================================================

st.subheader("Model Performance Visualization")

if COMPARISON_GRAPH_PATH.exists():

    st.image(
        str(COMPARISON_GRAPH_PATH),
        caption="Tesla Stock Price Prediction Model Comparison",
        width="stretch"
    )

else:

    st.warning(
        "Model comparison graph was not found."
    )


# ==================================================
# BEST MODEL
# ==================================================

st.subheader("Selected Model")

st.success(
    "Linear Regression"
)

st.write(
    "Linear Regression was selected as the final model "
    "because it achieved the best evaluation performance "
    "among the models tested in this project."
)


# ==================================================
# MODEL METRICS
# ==================================================

if "comparison_df" in locals():

    linear_model = comparison_df[
        comparison_df["Model"].str.contains(
            "Linear Regression",
            case=False,
            na=False
        )
    ]

    if not linear_model.empty:

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "MAE",
                f"{linear_model.iloc[0]['MAE']:.2f}"
            )

        with metric_col2:
            st.metric(
                "RMSE",
                f"{linear_model.iloc[0]['RMSE']:.2f}"
            )

        with metric_col3:
            st.metric(
                "R² Score",
                f"{linear_model.iloc[0]['R2 Score']:.4f}"
            )


st.divider()


# ==================================================
# ACTUAL VS PREDICTED GRAPH
# ==================================================

st.header("Actual vs Predicted Stock Prices")

st.write(
    "This graph compares the actual Tesla stock prices "
    "with the prices predicted by the selected Linear "
    "Regression model."
)

if ACTUAL_PREDICTED_GRAPH_PATH.exists():

    st.image(
        str(ACTUAL_PREDICTED_GRAPH_PATH),
        caption="Actual vs Predicted Tesla Stock Prices",
        width="stretch"
    )

else:

    st.warning(
        "Actual vs Predicted graph was not found."
    )


st.divider()


# ==================================================
# NEXT STOCK PRICE PREDICTION
# ==================================================

st.header("Next Tesla Stock Price Prediction")

st.write(
    "The model uses the previous 60 Tesla closing prices "
    "to predict the next closing price."
)


# ==================================================
# CURRENT DATA INFORMATION
# ==================================================

last_60_prices = data[["Close"]].tail(60)

last_closing_price = float(
    last_60_prices.iloc[-1]["Close"]
)

if "Date" in data.columns:

    try:
        last_available_date = pd.to_datetime(
            data["Date"].iloc[-1]
        ).strftime("%d-%b-%Y")

    except Exception:

        last_available_date = "Not available"

else:

    last_available_date = "Not available"


info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "Last Closing Price",
        f"${last_closing_price:.2f}"
    )

with info_col2:

    st.metric(
        "Last Available Date",
        last_available_date
    )

with info_col3:

    st.metric(
        "Historical Prices Used",
        "60"
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

predict_button = st.button(
    "Predict Next Tesla Stock Price",
    type="primary"
)


if predict_button:

    try:

        # --------------------------------------------------
        # STEP 1: GET LAST 60 CLOSING PRICES
        # --------------------------------------------------

        last_60_prices = data[
            ["Close"]
        ].tail(60)


        # --------------------------------------------------
        # STEP 2: SCALE THE DATA
        # --------------------------------------------------

        last_60_scaled = scaler.transform(
            last_60_prices
        )


        # --------------------------------------------------
        # STEP 3: CONVERT TO MODEL INPUT
        # --------------------------------------------------

        prediction_input = last_60_scaled.reshape(
            1,
            -1
        )


        # --------------------------------------------------
        # STEP 4: PREDICT NEXT SCALED PRICE
        # --------------------------------------------------

        prediction_scaled = model.predict(
            prediction_input
        )


        # --------------------------------------------------
        # STEP 5: CONVERT BACK TO ORIGINAL PRICE
        # --------------------------------------------------

        predicted_price = scaler.inverse_transform(
            np.array(prediction_scaled).reshape(-1, 1)
        )


        predicted_price_value = float(
            predicted_price[0][0]
        )


        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.success(
            f"Predicted Next Tesla Closing Price: "
            f"${predicted_price_value:.2f}"
        )


        # --------------------------------------------------
        # PREDICTION SUMMARY
        # --------------------------------------------------

        st.subheader("Prediction Summary")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Last Closing Price",
                f"${last_closing_price:.2f}"
            )

        with result_col2:

            st.metric(
                "Predicted Next Price",
                f"${predicted_price_value:.2f}"
            )

        with result_col3:

            difference = (
                predicted_price_value
                - last_closing_price
            )

            st.metric(
                "Estimated Change",
                f"${difference:.2f}"
            )


        st.caption(
            "This prediction is generated by the trained "
            "Linear Regression model using the previous "
            "60 closing prices."
        )


    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.error(
            f"Error details: {e}"
        )


st.divider()


# ==================================================
# PROJECT INFORMATION
# ==================================================

st.header("Project Information")

project_col1, project_col2 = st.columns(2)

with project_col1:

    st.write("**Machine Learning Models**")

    st.write(
        "• Linear Regression\n\n"
        "• Random Forest Regression"
    )

    st.write("**Prediction Method**")

    st.write(
        "The model uses the previous 60 Tesla closing "
        "prices as input features to predict the next "
        "closing price."
    )


with project_col2:

    st.write("**Final Model**")

    st.write(
        "Linear Regression"
    )

    st.write("**Application Technology**")

    st.write(
        "Python + Streamlit + Scikit-learn"
    )

    st.write("**Dataset**")

    st.write(
        "Historical Tesla stock market data"
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Tesla Stock Price Prediction System | "
    "Machine Learning Internship Minor Project"
)