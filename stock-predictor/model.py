import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta


def fetch_stock_data(ticker, period="2y"):
    """Fetch historical stock data using yfinance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    return df


def get_stock_info(ticker):
    """Get basic stock information."""
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")

    result = {
        "symbol": ticker.upper(),
        "name": info.get("longName", info.get("shortName", ticker.upper())),
        "currentPrice": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "previousClose": info.get("previousClose", 0),
        "marketCap": info.get("marketCap", 0),
        "volume": info.get("volume", info.get("regularMarketVolume", 0)),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "currency": info.get("currency", "USD"),
    }

    # Calculate change
    if result["currentPrice"] and result["previousClose"]:
        result["change"] = round(result["currentPrice"] - result["previousClose"], 2)
        result["changePercent"] = round(
            (result["change"] / result["previousClose"]) * 100, 2
        )
    else:
        result["change"] = 0
        result["changePercent"] = 0

    return result


def prepare_data(df, look_back=60):
    """Prepare data for LSTM model with sliding window."""
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back : i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler, scaled_data


def build_model(look_back=60):
    """Build LSTM model architecture."""
    model = Sequential()

    # First LSTM layer with dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))

    # Second LSTM layer with dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Dense layers
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def predict_future(ticker, days=30, look_back=60, epochs=50, batch_size=32):
    """
    Train LSTM model and predict future stock prices.

    Returns:
        dict with historical prices, predicted prices, dates, and model metrics
    """
    # Fetch data
    df = fetch_stock_data(ticker)

    # Prepare data
    X, y, scaler, scaled_data = prepare_data(df, look_back)

    # Split into train/test (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train model
    model = build_model(look_back)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    # Evaluate model
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    # Predict on test set for accuracy assessment
    test_predictions = model.predict(X_test, verbose=0)
    test_predictions = scaler.inverse_transform(test_predictions)
    actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_test - test_predictions) / actual_test)) * 100
    accuracy = max(0, 100 - mape)

    # Predict future prices
    last_sequence = scaled_data[-look_back:]
    future_predictions = []

    current_sequence = last_sequence.copy()
    for _ in range(days):
        # Reshape for prediction
        input_seq = current_sequence.reshape(1, look_back, 1)
        predicted_price = model.predict(input_seq, verbose=0)

        future_predictions.append(predicted_price[0, 0])

        # Slide window: remove first, add predicted
        current_sequence = np.append(current_sequence[1:], predicted_price, axis=0)

    # Inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    # Generate future dates (skip weekends)
    last_date = df.index[-1]
    future_dates = []
    current_date = last_date
    while len(future_dates) < days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            future_dates.append(current_date.strftime("%Y-%m-%d"))

    # Historical data for chart
    historical_dates = [d.strftime("%Y-%m-%d") for d in df.index[-90:]]
    historical_prices = df["Close"].values[-90:].tolist()

    # Build response
    result = {
        "ticker": ticker.upper(),
        "historical": {
            "dates": historical_dates,
            "prices": [round(p, 2) for p in historical_prices],
        },
        "predictions": {
            "dates": future_dates,
            "prices": [round(float(p), 2) for p in future_predictions.flatten()],
        },
        "metrics": {
            "trainLoss": round(float(train_loss), 6),
            "valLoss": round(float(val_loss), 6),
            "mape": round(float(mape), 2),
            "accuracy": round(float(accuracy), 2),
        },
        "lastPrice": round(float(df["Close"].values[-1]), 2),
        "predictedEndPrice": round(float(future_predictions[-1][0]), 2),
        "priceChange": round(
            float(future_predictions[-1][0] - df["Close"].values[-1]), 2
        ),
        "priceChangePercent": round(
            float(
                (future_predictions[-1][0] - df["Close"].values[-1])
                / df["Close"].values[-1]
                * 100
            ),
            2,
        ),
    }

    return result
