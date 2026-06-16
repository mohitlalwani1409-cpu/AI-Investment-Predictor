from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import webbrowser
from sklearn.linear_model import LinearRegression
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# 📊 GRAPH FUNCTION
def plot_graph(df):
    plt.figure()

    # Price line
    df['Close'].plot(label="Price")

    # ⭐ Moving Average
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA20'].plot(label="MA 20")

    plt.title("Stock Price with Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend()   # shows labels

    plt.savefig("static/plot.png")
    plt.close()

# 🤖 PREDICTION FUNCTION
# 🤖 PREDICTION FUNCTION
def predict_price(df):
    df = df.reset_index()

    df['Days'] = np.arange(len(df))

    X = df[['Days']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[len(df)]])
    predicted_price = model.predict(next_day)[0]

    current_price = df['Close'].iloc[-1]

    suggestion = "BUY 📈" if predicted_price > current_price else "SELL 📉"

    return round(predicted_price, 2), suggestion
def predict_price_lstm(df):
    import numpy as np

    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    last_60 = scaled_data[-60:]
    last_60 = last_60.reshape(1, 60, 1)

    predicted = model.predict(last_60)
    predicted_price = scaler.inverse_transform(predicted)[0][0]

    current_price = df['Close'].iloc[-1]

    suggestion = "BUY 📈" if predicted_price > current_price else "SELL 📉"

    return round(predicted_price, 2), suggestion

# 🔥 TOP STOCKS FUNCTION (SEPARATE)
def get_top_stocks():
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

    stocks = []

    for symbol in symbols:
        df = yf.Ticker(symbol).history(period="5d")

        current_price = round(df['Close'].iloc[-1], 2)
        previous_price = round(df['Close'].iloc[-2], 2)

        change = round(current_price - previous_price, 2)
        percent_change = round((change / previous_price) * 100, 2)

        stocks.append({
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "percent": percent_change
        })

    return stocks

# 🌐 MAIN ROUTE
@app.route('/', methods=['GET', 'POST'])
def home():
    top_stocks = get_top_stocks()
    data = None
    graph = False
    predicted_price = None
    suggestion = None
    current_price = None
    change = None
    percent_change = None

    if request.method == 'POST':
        symbol = request.form['symbol']
        df = yf.Ticker(symbol).history(period="6mo")

        # 📋 TABLE DATA
        data = df.tail().to_html(classes="table table-striped")

        # 💰 LIVE PRICE
        current_price = round(df['Close'].iloc[-1], 2)
        previous_price = round(df['Close'].iloc[-2], 2)

        change = round(current_price - previous_price, 2)
        percent_change = round((change / previous_price) * 100, 2)

        # 📊 GRAPH
        plot_graph(df)
        graph = True

        # 🤖 PREDICTION
        predicted_price, suggestion = predict_price(df)

    return render_template(
        'index.html',
        data=data,
        graph=graph,
        predicted_price=predicted_price,
        suggestion=suggestion,
        current_price=current_price,
        change=change,
        percent_change=percent_change,  
        top_stocks=top_stocks
    )


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)