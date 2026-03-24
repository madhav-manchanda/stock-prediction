from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

from model import predict_future, get_stock_info, fetch_stock_data

app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/predict")
def predict():
    """Predict future stock prices using LSTM model."""
    ticker = request.args.get("ticker", "").strip().upper()
    days = request.args.get("days", 30, type=int)

    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    if days < 7 or days > 90:
        return jsonify({"error": "Prediction days must be between 7 and 90"}), 400

    try:
        result = predict_future(ticker, days=days, epochs=25, batch_size=32)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/stock-info")
def stock_info():
    """Get basic stock information."""
    ticker = request.args.get("ticker", "").strip().upper()

    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    try:
        info = get_stock_info(ticker)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": f"Could not fetch info for '{ticker}': {str(e)}"}), 404


@app.route("/api/history")
def history():
    """Get historical price data for charting."""
    ticker = request.args.get("ticker", "").strip().upper()
    period = request.args.get("period", "1y")

    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    try:
        df = fetch_stock_data(ticker, period=period)
        data = {
            "ticker": ticker,
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "prices": {
                "open": [round(p, 2) for p in df["Open"].tolist()],
                "high": [round(p, 2) for p in df["High"].tolist()],
                "low": [round(p, 2) for p in df["Low"].tolist()],
                "close": [round(p, 2) for p in df["Close"].tolist()],
                "volume": df["Volume"].tolist(),
            },
        }
        return jsonify(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Could not fetch history: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n>> Stock Predictor running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
