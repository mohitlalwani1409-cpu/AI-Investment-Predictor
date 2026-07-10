"""
Stock AI Predictor - Flask Backend
Designed as a Resume-Grade Project for Internship Applications.

Key Interview Concepts Implemented:
1. In-Memory Caching: Caches the trending stocks list to prevent API rate limiting and speed up responses.
2. Dynamic Machine Learning Inference: Performs real-time Linear Regression fitting and LSTM model prediction.
3. API Data Ingestion & Formatting: Pulls real-time financial metrics, news feeds, and historical ranges via yfinance.
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI thread conflicts
import matplotlib.pyplot as plt
import webbrowser
import numpy as np
import time
import json
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# 🤖 LOAD PRE-TRAINED DEEP LEARNING MODEL
# The LSTM model and scaler are loaded once at server startup.
try:
    lstm_model = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.pkl")
    lstm_loaded = True
except Exception as e:
    print(f"⚠️ Warning: Could not load LSTM model or scaler. Details: {e}")
    lstm_loaded = False


# 💡 GLOBAL IN-MEMORY CACHE FOR TRENDING TICKERS
# Caching reduces the page load time on the first visit and prevents Yahoo Finance API rate limits.
cached_top_stocks = None
last_cache_time = 0
CACHE_DURATION_SEC = 600  # 10 minutes cache duration

# Popular stocks metadata (names and currencies are static to avoid slow ticker.info web requests in a loop)
POPULAR_STOCKS_METADATA = {
    "AAPL": {"name": "Apple Inc.", "currency": "$"},
    "TSLA": {"name": "Tesla Inc.", "currency": "$"},
    "MSFT": {"name": "Microsoft Corp.", "currency": "$"},
    "GOOGL": {"name": "Alphabet Inc.", "currency": "$"},
    "AMZN": {"name": "Amazon.com Inc.", "currency": "$"}
}


# 🛠️ HELPER: FORMAT CURRENCY SYMBOL
def get_currency_symbol(currency_code):
    """Maps standard currency codes to symbols (e.g. USD -> $, INR -> ₹)."""
    if not currency_code:
        return "$"
    mapping = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        'GBP': '£',
        'CAD': 'C$',
        'AUD': 'A$',
        'JPY': '¥',
    }
    return mapping.get(currency_code.upper(), currency_code.upper() + " ")


# 🛠️ HELPER: FORMAT LARGE NUMBERS (e.g. Market Cap)
def format_large_number(num):
    """Converts a raw number into a readable currency format (e.g. 1.2 Trillion)."""
    if not isinstance(num, (int, float)):
        return 'N/A'
    if num >= 1e12:
        return f"{num / 1e12:.2f} T"
    elif num >= 1e9:
        return f"{num / 1e9:.2f} B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f} M"
    else:
        return f"{num:,.2f}"


# 📊 GRAPH FUNCTION (Matplotlib fallback)
def plot_graph(df):
    """Saves a static trend image to the static/ folder. Used as a classic baseline plot."""
    plt.figure(figsize=(10, 5))
    df['Close'].plot(label="Price", color='#0d6efd', linewidth=2)

    # 20-Day Moving Average to smooth out short-term price fluctuations
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA20'].plot(label="MA 20 (Short Term Trend)", color='#ffc107', linestyle='--')

    plt.title("Stock Price & 20-Day Moving Average", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/plot.png", dpi=150)
    plt.close()


# 🤖 ML MODEL 1: LINEAR REGRESSION (Supervised Statistical Model)
def predict_price_lr(df):
    """
    Fits a linear trend line (y = mx + c) over the stock's historical close prices.
    Predicts the next trading day's price.
    """
    df_temp = df.reset_index()
    # Create an independent variable 'Days' (0, 1, 2, ...)
    df_temp['Days'] = np.arange(len(df_temp))

    X = df_temp[['Days']]
    y = df_temp['Close']

    # Initialize and train the model dynamically on the current stock's data
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    # Predict the next day (length of dataset)
    next_day = np.array([[len(df_temp)]])
    predicted_price = lr_model.predict(next_day)[0]

    current_price = df['Close'].iloc[-1]
    suggestion = "BUY 📈" if predicted_price > current_price else "SELL 📉"

    return round(predicted_price, 2), suggestion


# 🤖 ML MODEL 2: LSTM (Recurrent Neural Network for Deep Learning)
def predict_price_lstm(df):
    """
    Executes a pre-trained Long Short-Term Memory (LSTM) network.
    LSTM is a Recurrent Neural Network designed to learn order dependence in sequences.
    Looks at the last 60 days of closing prices to predict the next price.
    """
    if not lstm_loaded:
        raise ValueError("LSTM model was not loaded successfully at startup.")

    # 1. Shape and scale the raw inputs using the pre-saved MinMaxScaler
    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    # 2. Extract the last 60 trading days
    last_60 = scaled_data[-60:]
    last_60 = last_60.reshape(1, 60, 1)  # Reshape to matching LSTM input tensor [batch_size, time_steps, features]

    # 3. Model inference
    predicted_scaled = lstm_model.predict(last_60)
    
    # 4. Inverse transform to convert the 0-1 scale back to actual stock currency values
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    current_price = df['Close'].iloc[-1]
    suggestion = "BUY 📈" if predicted_price > current_price else "SELL 📉"

    return round(predicted_price, 2), suggestion


# 🔥 POPULAR STOCKS LIST WITH MEMORY CACHE
def get_top_stocks():
    """Retrieves and caches the top stock prices to reduce load times."""
    global cached_top_stocks, last_cache_time
    current_time = time.time()

    # Serve from cache if available and fresh
    if cached_top_stocks and (current_time - last_cache_time < CACHE_DURATION_SEC):
        return cached_top_stocks

    stocks = []
    for symbol, meta in POPULAR_STOCKS_METADATA.items():
        try:
            # yfinance history is used to grab the latest closing and previous closing price
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d")
            if df.empty or len(df) < 2:
                continue

            current_price = round(df['Close'].iloc[-1], 2)
            previous_price = round(df['Close'].iloc[-2], 2)
            change = round(current_price - previous_price, 2)
            percent_change = round((change / previous_price) * 100, 2)

            stocks.append({
                "symbol": symbol,
                "name": meta["name"],
                "price": current_price,
                "change": change,
                "percent": percent_change,
                "currency_symbol": meta["currency"]
            })
        except Exception as e:
            print(f"Could not load cache for {symbol}: {e}")

    if stocks:
        cached_top_stocks = stocks
        last_cache_time = current_time
        return stocks

    # Fallback to expired cache if internet is down or API fails
    return cached_top_stocks or []


# 🌐 MAIN WEB APPLICATION ROUTE
@app.route('/', methods=['GET', 'POST'])
def home():
    top_stocks = get_top_stocks()
    
    # Define active variables
    symbol = None
    period = '1y'  # Default period chosen to ensure we have enough data (60+ days) for the LSTM model
    error_msg = None
    
    # Stock info variables
    company_name = None
    sector = 'N/A'
    industry = 'N/A'
    description = 'No company description available.'
    currency_symbol = '$'
    
    # Key Fundamentals
    market_cap = 'N/A'
    pe_ratio = 'N/A'
    dividend_yield = 'N/A'
    volume = 'N/A'
    avg_volume = 'N/A'
    fifty_two_week_high = 'N/A'
    fifty_two_week_low = 'N/A'
    day_open = 'N/A'
    
    # Live stats
    current_price = None
    change = None
    percent_change = None
    
    # Charts & Tables
    data_html = None
    chart_data_json = None
    graph_rendered = False
    
    # Machine Learning Predictions
    predicted_price_lr = None
    suggestion_lr = None
    
    predicted_price_lstm = None
    suggestion_lstm = None
    lstm_error_msg = None

    news_articles = []

    # Get target symbol from either POST (Search) or GET (Watchlist click/Default load)
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip().upper()
        period = request.form.get('period', '1y')
    else:
        symbol = request.args.get('symbol', '').strip().upper()
        # On first load, default to Apple (AAPL) to show off a pre-loaded, populated dashboard
        if not symbol:
            symbol = 'AAPL'

    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            # Fetch historical dataframe
            df = ticker.history(period=period)
            
            if df.empty or len(df) < 5:
                error_msg = f"No historical market data found for ticker '{symbol}'. Please double check the symbol."
                symbol = None
            else:
                # 📋 EXTRACT RAW DATA FOR PREVIEW TABLE
                # Reorder to show newest records first
                data_html = df.tail(10).iloc[::-1].to_html(
                    classes="table table-dark table-hover table-striped border-secondary mb-0 align-middle",
                    formatters={'Close': lambda x: f"{x:.2f}", 'Open': lambda x: f"{x:.2f}", 
                                'High': lambda x: f"{x:.2f}", 'Low': lambda x: f"{x:.2f}", 
                                'Volume': lambda x: f"{int(x):,}"}
                )

                # 💰 LIVE PRICE STATISTICS
                current_price = round(df['Close'].iloc[-1], 2)
                previous_price = round(df['Close'].iloc[-2], 2)
                change = round(current_price - previous_price, 2)
                percent_change = round((change / previous_price) * 100, 2)

                # 📊 PREPARE INTERACTIVE CHART DATA (Chart.js JSON Integration)
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                
                chart_data = {
                    "dates": df.index.strftime('%Y-%m-%d').tolist(),
                    "close": [None if np.isnan(x) else round(x, 2) for x in df['Close'].tolist()],
                    "open": [None if np.isnan(x) else round(x, 2) for x in df['Open'].tolist()],
                    "high": [None if np.isnan(x) else round(x, 2) for x in df['High'].tolist()],
                    "low": [None if np.isnan(x) else round(x, 2) for x in df['Low'].tolist()],
                    "volume": [0 if np.isnan(x) else int(x) for x in df['Volume'].tolist()],
                    "ma20": [None if np.isnan(x) else round(x, 2) for x in df['MA20'].tolist()],
                    "ma50": [None if np.isnan(x) else round(x, 2) for x in df['MA50'].tolist()],
                }
                chart_data_json = json.dumps(chart_data)

                # 📊 MATPLOTLIB STATIC PLOT (Generated in background as backup)
                try:
                    plot_graph(df)
                    graph_rendered = True
                except Exception as chart_err:
                    print(f"Matplotlib generation error: {chart_err}")

                # 🧠 MODEL 1: LINEAR REGRESSION RUN
                predicted_price_lr, suggestion_lr = predict_price_lr(df)

                # 🧠 MODEL 2: LSTM TIME-SERIES MODEL RUN
                if lstm_loaded:
                    if len(df) >= 60:
                        try:
                            predicted_price_lstm, suggestion_lstm = predict_price_lstm(df)
                        except Exception as lstm_err:
                            lstm_error_msg = f"LSTM Prediction failed: {str(lstm_err)}"
                    else:
                        lstm_error_msg = "LSTM requires at least 60 trading days. Please select a larger timeframe."
                else:
                    lstm_error_msg = "LSTM model file was not successfully initialized on startup."

                # 🏛️ INGEST COMPANY FUNDAMENTAL DETAILS
                try:
                    info = ticker.info
                    if info and isinstance(info, dict):
                        company_name = info.get('longName') or info.get('shortName') or symbol
                        sector = info.get('sector', 'N/A')
                        industry = info.get('industry', 'N/A')
                        description = info.get('longBusinessSummary', 'No description available.')
                        
                        currency_val = info.get('currency', 'USD')
                        currency_symbol = get_currency_symbol(currency_val)

                        # Standard technical indicators
                        market_cap = format_large_number(info.get('marketCap'))
                        pe_ratio = round(info.get('trailingPE'), 2) if info.get('trailingPE') else 'N/A'
                        
                        div_yield = info.get('dividendYield')
                        dividend_yield = f"{div_yield * 100:.2f}%" if div_yield else 'N/A'
                        
                        volume = f"{info.get('volume', 0):,}" if info.get('volume') else 'N/A'
                        avg_volume = f"{info.get('averageVolume', 0):,}" if info.get('averageVolume') else 'N/A'
                        
                        fifty_two_week_high = round(info.get('fiftyTwoWeekHigh'), 2) if info.get('fiftyTwoWeekHigh') else 'N/A'
                        fifty_two_week_low = round(info.get('fiftyTwoWeekLow'), 2) if info.get('fiftyTwoWeekLow') else 'N/A'
                        day_open = round(info.get('open'), 2) if info.get('open') else 'N/A'
                except Exception as info_err:
                    print(f"yfinance info retrieval failed: {info_err}")
                
                # Fallbacks from dataframe if info API was rate-limited/failed
                if not company_name:
                    company_name = symbol
                if volume == 'N/A':
                    volume = f"{int(df['Volume'].iloc[-1]):,}"
                if avg_volume == 'N/A':
                    avg_volume = f"{int(df['Volume'].mean()):,}"
                if fifty_two_week_high == 'N/A':
                    fifty_two_week_high = round(df['High'].max(), 2)
                if fifty_two_week_low == 'N/A':
                    fifty_two_week_low = round(df['Low'].min(), 2)
                if day_open == 'N/A':
                    day_open = round(df['Open'].iloc[-1], 2)

                # 📰 LIVE NEWS FEED INGESTION
                try:
                    raw_news = ticker.news
                    if raw_news:
                        for article in raw_news[:6]:  # Show top 6 news items
                            content = article.get('content', {})
                            if not content or not content.get('title'):
                                continue
                            
                            # Clean article URL
                            article_url = (content.get('clickThroughUrl', {}).get('url') or 
                                           content.get('canonicalUrl', {}).get('url') or '#')
                            
                            # Extract clean source name
                            source_name = content.get('provider', {}).get('displayName', 'Financial News')
                            
                            news_articles.append({
                                'title': content.get('title'),
                                'publisher': source_name,
                                'url': article_url,
                                'date': content.get('pubDate', '').split('T')[0],  # Get date segment
                                'summary': content.get('summary') or content.get('description', '')
                            })
                except Exception as news_err:
                    print(f"News collection failed: {news_err}")
                    
        except Exception as general_err:
            error_msg = f"Server failed to fetch details for '{symbol}': {str(general_err)}"
            symbol = None

    return render_template(
        'index.html',
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        industry=industry,
        description=description,
        currency_symbol=currency_symbol,
        
        # Fundamentals
        market_cap=market_cap,
        pe_ratio=pe_ratio,
        dividend_yield=dividend_yield,
        volume=volume,
        avg_volume=avg_volume,
        fifty_two_week_high=fifty_two_week_high,
        fifty_two_week_low=fifty_two_week_low,
        day_open=day_open,

        # Real-time Status
        current_price=current_price,
        change=change,
        percent_change=percent_change,
        
        # Data & charts
        data_html=data_html,
        chart_data_json=chart_data_json,
        graph_rendered=graph_rendered,
        period=period,
        
        # Predictions
        predicted_price_lr=predicted_price_lr,
        suggestion_lr=suggestion_lr,
        predicted_price_lstm=predicted_price_lstm,
        suggestion_lstm=suggestion_lstm,
        lstm_error_msg=lstm_error_msg,
        
        # Markets & details
        top_stocks=top_stocks,
        news_articles=news_articles,
        error_msg=error_msg
    )


if __name__ == "__main__":
    # Open local browser dynamically to make testing simpler for the developer
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)