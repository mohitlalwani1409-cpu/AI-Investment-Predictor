# 📈 AI Stock Predictor & Analytics Dashboard

A responsive, recruiter-grade web application that leverages **statistical modeling** and **recurrent deep learning** to analyze stock prices, pull real-time fundamentals, and forecast future market trends on-demand.

---

## 🌟 Key Features

1. **Dual Machine Learning Predictions**:
   - **Linear Regression**: Dynamic baseline trend analysis based on supervised statistical curve fitting.
   - **LSTM Recurrent Neural Network (RNN)**: A pre-trained deep learning network designed to recognize chronological patterns in 60-day historical time-series sequences.
2. **Interactive Charting (Chart.js)**: High-performance line/area charts with toggleable moving average curve overlays (MA20/MA50) and interactive hover-tooltips, replacing static Matplotlib graphs.
3. **Real-time API Ingestion**: Pulls company profiles, financials, and live news streams dynamically using `yfinance`.
4. **In-Memory Caching (System Design Best Practice)**: Caches popular stock lists for 10 minutes, optimizing load times (under 0.1s for cached hits) and safeguarding against external API rate-limiting blocks.
5. **Client-Side State Persistence**: Saves theme options and user watchlists locally in browser `localStorage`.
6. **Polished Glassmorphic UI**: Beautiful dark-mode-first dashboard utilizing Bootstrap 5 and custom CSS with smooth transitions and responsive layouts.

---

## 🛠️ Technology Stack

- **Backend**: Python, Flask (Web framework)
- **Data Ingestion**: Yahoo Finance API (`yfinance`)
- **Machine Learning**: TensorFlow / Keras (Deep Learning LSTM), Scikit-Learn (Linear Regression), NumPy (Data manipulation)
- **Frontend**: HTML5, Vanilla CSS3 (Custom variables, glassmorphism shadows, keyframes), Bootstrap 5 (Layout & responsiveness), Chart.js (Data visualization)

---

## 📂 Project Architecture

```
stock_project/
│
├── app.py              # Main Flask server (Routes, cache controls, ML inference)
├── train_model.py      # LSTM Model training script (Demonstrates modeling workflow)
├── lstm_model.h5       # Pre-trained TensorFlow LSTM weights file
├── scaler.pkl          # MinMaxScaler parameters saved during training
│
├── templates/
│   └── index.html      # Responsive dashboard interface (Chart.js & state controllers)
│
├── static/
│   ├── plot.png        # Matplotlib static plot fallback
│   └── (Dynamic plots generated during runtime)
│
└── README.md           # Project documentation & interview cheat sheet
```

---

## ⚙️ Quick Installation & Setup

1. **Clone or Open the Project Directory**
   Ensure all project files are in the active folder.

2. **Install Required Libraries**
   Run the following command in your terminal to install the necessary packages:
   ```bash
   pip install flask yfinance scikit-learn tensorflow joblib numpy matplotlib
   ```

3. **Launch the Server**
   Start the application:
   ```bash
   python app.py
   ```

4. **Access the Dashboard**
   The application will automatically launch your default browser. If not, open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 🎓 Recruiter Interview Q&A Cheat Sheet

Review these quick questions and answers to feel 100% prepared when explaining this project in an internship interview:

### 💬 Q1: "Why did you build this project, and how does the data pipeline work?"
> **Answer**: I wanted to create a practical dashboard that connects live financial market feeds with machine learning models. When a user inputs a stock ticker, the backend makes an on-demand REST API call to Yahoo Finance via the `yfinance` library. It extracts historical closing prices, volumes, and news feeds. This raw data is then processed in the backend, packaged into JSON formats, and sent to the frontend to drive interactive Chart.js visualizations.

### 💬 Q2: "What is the difference between your two prediction models (Linear Regression vs. LSTM)?"
> **Answer**:
> - **Linear Regression** is a statistical, supervised learning model. It treats time (days) as an independent variable and fits a line $y = mx + c$ that minimizes the sum of squared differences from the actual prices. It is simple, highly interpretable, and tells us the overall directional trajectory.
> - **LSTM (Long Short-Term Memory)** is a recurrent deep learning architecture. Unlike standard regression, LSTMs have internal memory gates (input, forget, and output gates) designed to learn time-series sequences. It looks at the last 60 days of closing prices (representing a sequence) and makes a non-linear prediction by recognizing historical patterns.

### 💬 Q3: "I notice your LSTM model uses a pre-saved scaler file (`scaler.pkl`). Why is that?"
> **Answer**: LSTMs are sensitive to scale; passing raw prices like \$150 or \$3000 causes gradient instability. We scale the values into a `[0,1]` range using a `MinMaxScaler`. It is crucial to use the *same* scaler instance (`scaler.pkl`) that was fitted on the training data so that the model evaluates the scaled numbers on the exact same distribution it was trained on.

### 💬 Q4: "What optimizations did you implement for performance?"
> **Answer**:
> 1. **Backend Caching**: I implemented an in-memory caching mechanism. The homepage lists five trending stocks; querying Yahoo Finance for five different symbols in real-time on every load takes 3-5 seconds and can lead to API rate limiting. Caching these results in memory for 10 minutes reduces subsequent page load times to under 0.1 seconds.
> 2. **Stateless Watchlist**: To keep the backend server stateless and light, I built the watchlist feature using the browser's `localStorage` API. The stock list persists on the client's browser, eliminating database overhead.

### 💬 Q5: "What are the limitations of your machine learning models?"
> **Answer**: The LSTM model was pre-trained on Apple (`AAPL`) closing prices. In a production environment, we would train individual models for each ticker or use normalized percentage changes rather than absolute prices to generalize the model across different price levels. This project is a great proof-of-concept showing how to deploy models in web applications, and in the future, I plan to expand this to run dynamic training jobs.
