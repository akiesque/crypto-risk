# =============================
# GRADIO DASHBOARD FOR CRYPTO VOLATILITY
# =============================

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import joblib  # for saving/loading model
import json
import os

# -----------------------------
# 1. Load your trained model
# -----------------------------
try:
    model_data = joblib.load("model/volatility_model.pkl")
    
    # Extract model and features from dict
    if isinstance(model_data, dict):
        model = model_data["model"]
        FEATURES = model_data["features"]
    else:
        model = model_data
        # Load features from metadata
        with open("model/model_metadata.json", "r") as f:
            metadata = json.load(f)
            FEATURES = metadata["features"]
except Exception as e:
    raise

# -----------------------------
# 2. Load dataset for dropdown
# -----------------------------
df = pd.read_csv("data/crypto_historical_365days.csv")
df["date"] = pd.to_datetime(df["date"])

# Unique coins for dropdown
coins = df["symbol"].unique().tolist()

# -----------------------------
# 2b. Load coin descriptions
# -----------------------------
try:
    desc_df = pd.read_csv("data/crypto_descriptions.csv", engine="python")
    # Normalize symbols to uppercase strings for robust lookup
    desc_df["symbol"] = desc_df["symbol"].astype(str).str.upper()
    desc_df["description_en"] = desc_df["description_en"].fillna("")
    SYMBOL_TO_DESC = {
        row["symbol"]: str(row["description_en"])
        for _, row in desc_df.iterrows()
    }
except Exception:
    # Fallback: no descriptions available
    SYMBOL_TO_DESC = {}

# -----------------------------
# 3. Feature engineering function
# -----------------------------
def prepare_features(df_coin):
    # lagged/rolling features (use most recent row)
    row = df_coin.iloc[-1:].copy()
    
    try:
        # Lagged volatility
        if len(df_coin) > 1:
            row['volatility_7d_lag1'] = df_coin['volatility_7d'].iloc[-2] if len(df_coin) > 1 else 0
        if len(df_coin) > 7:
            row['volatility_7d_lag7'] = df_coin['volatility_7d'].iloc[-8] if len(df_coin) > 7 else 0
        
        # Realized volatility
        if len(df_coin) >= 7:
            row['realized_vol_7d'] = df_coin['daily_return'].iloc[-7:].std() * np.sqrt(7)
        if len(df_coin) >= 30:
            row['realized_vol_30d'] = df_coin['daily_return'].iloc[-30:].std() * np.sqrt(30)
            row['realized_vol_7d_lag1'] = df_coin['daily_return'].iloc[-8:-1].std() * np.sqrt(7) if len(df_coin) > 7 else 0
        
        # Rolling volatility means
        if len(df_coin) >= 7:
            row['volatility_7d_ma7'] = df_coin['volatility_7d'].iloc[-7:].mean()
        if len(df_coin) >= 30:
            row['volatility_7d_ma30'] = df_coin['volatility_7d'].iloc[-30:].mean()
        
        # Returns
        if len(df_coin) >= 7:
            row['return_7d'] = df_coin['price'].iloc[-1] / df_coin['price'].iloc[-8] - 1 if len(df_coin) > 7 else 0
        if len(df_coin) >= 14:
            row['return_14d'] = df_coin['price'].iloc[-1] / df_coin['price'].iloc[-15] - 1 if len(df_coin) > 14 else 0
        if len(df_coin) >= 30:
            row['return_30d'] = df_coin['price'].iloc[-1] / df_coin['price'].iloc[-31] - 1 if len(df_coin) > 30 else 0
        
        # Momentum
        if len(df_coin) > 1:
            row['return_momentum'] = df_coin['daily_return'].iloc[-1] - df_coin['daily_return'].iloc[-2]
        if len(df_coin) > 2:
            prev_momentum = df_coin['daily_return'].iloc[-2] - df_coin['daily_return'].iloc[-3]
            row['return_acceleration'] = row['return_momentum'].iloc[0] - prev_momentum if len(df_coin) > 2 else 0
        
        # Volume features
        if len(df_coin) > 1:
            row['volume_change'] = (df_coin['volume'].iloc[-1] / df_coin['volume'].iloc[-2] - 1) if df_coin['volume'].iloc[-2] != 0 else 0
        if len(df_coin) >= 7:
            row['volume_ma7'] = df_coin['volume'].iloc[-7:].mean()
            row['volume_momentum_7d'] = (df_coin['volume'].iloc[-1] / df_coin['volume'].iloc[-8] - 1) if len(df_coin) > 7 and df_coin['volume'].iloc[-8] != 0 else 0
        if len(df_coin) >= 30:
            row['volume_ma30'] = df_coin['volume'].iloc[-30:].mean()
        
        # Trend ratio - extract scalars
        if 'price_ma7' in row.columns and 'price_ma30' in row.columns:
            price_ma7_val = row['price_ma7'].iloc[0]
            price_ma30_val = row['price_ma30'].iloc[0]
            row['ma_ratio'] = price_ma7_val / (price_ma30_val + 1e-8)
        
        # Volatility momentum
        if len(df_coin) > 1:
            row['volatility_momentum'] = df_coin['volatility_7d'].iloc[-1] - df_coin['volatility_7d'].iloc[-2]
        
        # Price range
        if len(df_coin) >= 7:
            price_window = df_coin['price'].iloc[-7:]
            row['price_range_7d'] = (price_window.max() - price_window.min()) / (price_window.mean() + 1e-8)
        if len(df_coin) >= 30:
            price_window = df_coin['price'].iloc[-30:]
            row['price_range_30d'] = (price_window.max() - price_window.min()) / (price_window.mean() + 1e-8)
        
        # Market-wide features (simplified - would need full dataset)
        row['market_volatility'] = df_coin['volatility_7d'].iloc[-1]  # Placeholder
        row['market_volatility_lag7'] = df_coin['volatility_7d'].iloc[-8] if len(df_coin) > 7 else 0
        row['market_realized_vol'] = row['realized_vol_7d'].iloc[0] if 'realized_vol_7d' in row.columns else 0
        
        # Relative volatility - extract scalars using .iloc[0] to avoid Series comparison
        if 'volatility_7d' in row.columns:
            vol_7d_val = row['volatility_7d'].iloc[0]
            market_vol_val = row['market_volatility'].iloc[0]
            row['volatility_relative'] = vol_7d_val / (market_vol_val + 1e-8)
        else:
            row['volatility_relative'] = 0
        
        if 'realized_vol_7d' in row.columns:
            realized_vol_val = row['realized_vol_7d'].iloc[0]
            market_realized_val = row['market_realized_vol'].iloc[0]
            row['realized_vol_relative'] = realized_vol_val / (market_realized_val + 1e-8) if market_realized_val != 0 else 0
        else:
            row['realized_vol_relative'] = 0
        
        # Volume-volatility relationships - extract scalars
        if 'volatility_7d' in row.columns:
            vol_7d_val = row['volatility_7d'].iloc[0]
            volume_val = row['volume'].iloc[0]
            row['volume_volatility_ratio'] = volume_val / (vol_7d_val + 1e-8)
            
            if 'volume_change' in row.columns:
                daily_return_val = row['daily_return'].iloc[0]
                volume_change_val = row['volume_change'].iloc[0]
                row['price_volume_divergence'] = (daily_return_val - volume_change_val) ** 2
            else:
                row['price_volume_divergence'] = 0
        else:
            row['volume_volatility_ratio'] = 0
            row['price_volume_divergence'] = 0
        
        # Downside volatility
        if len(df_coin) >= 7:
            downside = df_coin['daily_return'].iloc[-7:].where(df_coin['daily_return'].iloc[-7:] < 0, 0)
            row['downside_vol_7d'] = downside.std() * np.sqrt(7)
        
        # Extreme returns
        if len(df_coin) >= 7:
            returns_abs = df_coin['daily_return'].iloc[-7:].abs()
            threshold = returns_abs.quantile(0.95)
            row['extreme_return_count_7d'] = (returns_abs > threshold).sum()
        
        # Time encoding
        if 'date' in row.columns:
            date_val = pd.to_datetime(row['date'].iloc[0])
            row['month_numeric'] = date_val.month
            row['month_sin'] = np.sin(2 * np.pi * date_val.month / 12)
            row['month_cos'] = np.cos(2 * np.pi * date_val.month / 12)
            row['day_of_week'] = date_val.dayofweek
            row['day_sin'] = np.sin(2 * np.pi * date_val.dayofweek / 7)
            row['day_cos'] = np.cos(2 * np.pi * date_val.dayofweek / 7)
        
    except Exception as e:
        raise
    
    # Ensure all features exist
    for f in FEATURES:
        if f not in row.columns:
            row[f] = 0
    
    return row[FEATURES]

# -----------------------------
# 4. Risk label function
# -----------------------------
def risk_label(vol):
    if vol < 0.25:
        return "Low 🟢"
    elif vol < 0.6:
        return "Medium 🟡"
    return "High 🔴"

# -----------------------------
# 5a. Description lookup function (lightweight, no model needed)
# -----------------------------
def get_coin_description(coin_symbol):
    """Return description for a coin symbol immediately (no prediction)."""
    if not coin_symbol:
        return ""
    symbol_key = str(coin_symbol).upper()
    desc = SYMBOL_TO_DESC.get(symbol_key, "No description available for this asset yet.")
    
    # Strip any leading/trailing triple quotes that might be in the CSV data
    desc = desc.strip()
    if desc.startswith('"""') and desc.endswith('"""'):
        desc = desc[3:-3].strip()
    elif desc.startswith('"') and desc.endswith('"'):
        desc = desc[1:-1].strip()
    
    return desc

# -----------------------------
# 5b. Prediction function (heavy computation)
# -----------------------------
def predict_volatility(coin_symbol):
    try:
        df_coin = df[df["symbol"] == coin_symbol].copy()
        
        if len(df_coin) == 0:
            raise ValueError(f"No data found for symbol: {coin_symbol}")
        
        X = prepare_features(df_coin)
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Predict
        vol_pred = model.predict(X_array)[0]
        
        risk = risk_label(vol_pred)
        
        # Feature importance plot
        importances = model.feature_importances_
        feat_names = FEATURES[:len(importances)]
        
        plt.figure(figsize=(8,6))
        top_features = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
        sns.barplot(x="importance", y="feature", data=top_features)
        plt.title("Top 15 Feature Importance")
        plt.tight_layout()
        
        # Save plot to buffer and convert to PIL Image
        import io
        from PIL import Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return float(vol_pred), risk, img
        
    except Exception as e:
        raise

with gr.Blocks(title="Crypto Volatility & Risk Dashboard") as iface:
    gr.Markdown(
        "## Crypto Volatility & Risk Dashboard\n"
        "Predicts 7-day volatility for a selected cryptocurrency and shows the risk level and feature importance."
    )

    with gr.Row():
        with gr.Column(scale=1):
            coin_input = gr.Dropdown(choices=coins, label="Select Cryptocurrency")
            predict_button = gr.Button("Predict Volatility")

            output_description = gr.Textbox(
                label="Asset Description",
                lines=10,
                interactive=False,
            )

        with gr.Column(scale=1):
            vol_output = gr.Number(label="Predicted 7-Day Volatility")
            risk_output = gr.Textbox(label="Risk Level")
            img_output = gr.Image(type="pil", label="Feature Importance")

    # Update description immediately when coin is selected (no prediction needed)
    coin_input.change(
        fn=get_coin_description,
        inputs=coin_input,
        outputs=output_description,
    )

    # Run prediction only when button is clicked
    predict_button.click(
        fn=predict_volatility,
        inputs=coin_input,
        outputs=[vol_output, risk_output, img_output],
    )

if __name__ == "__main__":
    iface.launch(theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]))
