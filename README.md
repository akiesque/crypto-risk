# 📊 Crypto Risk & Volatility Dashboard

A machine learning–powered dashboard that predicts **short-term cryptocurrency volatility** and assigns a **risk level** using historical market data.  
Built with **XGBoost**, **Python**, and **Gradio** for fast experimentation and interactive visualization.

---

## 🚀 Overview

Cryptocurrency markets are highly volatile and difficult to model due to noise, regime shifts, and non-stationarity.  
This project explores **data-driven volatility estimation** using engineered financial indicators and a gradient boosting regression model.

The application allows users to:
- Select a cryptocurrency
- Predict its **7-day volatility**
- View a corresponding **risk classification**
- Inspect **feature importance** driving the prediction

---

## 🧠 Model & Approach

- **Model:** XGBoost Regressor  
- **Target:** 7-day rolling volatility  
- **Learning Strategy:**  
  - Low learning rate + large number of estimators  
  - Conservative regularization to reduce overfitting in noisy markets  

### Final Model Parameters
```python
XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42
)
```

## 📈 Features Used

The model leverages both **price-based** and **statistical volatility indicators**, including:

- Price, volume, and market capitalization  
- Daily returns and cumulative returns  
- Moving averages (7-day, 30-day)  
- Lagged volatility values  
- Rolling standard deviation of returns  
- Return skewness and kurtosis  
- Seasonal encoding (month sine/cosine)  
- Market cap rank  

These features help capture **trend, momentum, dispersion, and market regime effects**.

---

## 📊 Evaluation Metrics

Final validation performance:

- **RMSE:** 2.98  
- **MAE:** 1.77  
- **R²:** 0.17  

While crypto volatility remains inherently unpredictable, the model demonstrates **meaningful signal extraction** and performs significantly better than naïve baselines.

---

## 🖥️ Dashboard (Gradio)

The Gradio interface provides:

- Coin selection dropdown  
- Predicted 7-day volatility value  
- Risk classification:
  - 🟢 Low  
  - 🟡 Medium  
  - 🔴 High  
- Feature importance visualization  

This makes the model **interpretable, interactive, and demo-ready**.

---

## 📂 Dataset

- **Source:** Kaggle – Top 100 Cryptocurrencies Daily Price Data (2025)  
- **Time Range:** December 2024 – Present  
- **Granularity:** Daily  

---

## 🛠️ Tech Stack

- Python  
- XGBoost  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- Gradio  

---

## 📌 Use Cases

- Risk awareness for retail traders  
- Exploratory financial data science  
- Volatility modeling experiments  
- Portfolio-grade ML demo project  

---

## 🔮 Future Improvements

- Multi-horizon volatility prediction (7 / 14 / 30 days)  
- Quantile regression for uncertainty bands  
- Coin embeddings or regime-aware modeling  
- Ensemble models (XGBoost + LightGBM)  
- Live price ingestion via APIs  

---

## 📜 Disclaimer

This project is for **educational and research purposes only** and does not constitute financial advice.



