# =============================
# CRYPTO VOLATILITY FORECAST
# XGBoost Regression
# Dataset: top-100 cryptocurrencies daily
# =============================

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "data", "crypto_historical_365days.csv")
df = pd.read_csv(csv_path)  

# Ensure timestamp/date is datetime
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["symbol", "date"])
df["target_vol_7d"] = df.groupby("symbol")["volatility_7d"].shift(-7)

# Drop last rows per coin with NaN target
df = df.dropna(subset=["target_vol_7d"])


df["return_7d"] = df.groupby("symbol")["price"].pct_change(7)
df["return_14d"] = df.groupby("symbol")["price"].pct_change(14)
df["return_30d"] = df.groupby("symbol")["price"].pct_change(30)

# Volume dynamics
df["volume_change"] = df.groupby("symbol")["volume"].pct_change()
df["volume_ma7"] = df.groupby("symbol")["volume"].rolling(7).mean().reset_index(0, drop=True)
df["volume_ma30"] = df.groupby("symbol")["volume"].rolling(30).mean().reset_index(0, drop=True)
df["volume_momentum_7d"] = df.groupby("symbol")["volume"].pct_change(7)

# Trend strength
df["ma_ratio"] = df["price_ma7"] / (df["price_ma30"] + 1e-8)

# Lagged volatility features (CRITICAL for volatility clustering)
df["volatility_7d_lag1"] = df.groupby("symbol")["volatility_7d"].shift(1)
df["volatility_7d_lag7"] = df.groupby("symbol")["volatility_7d"].shift(7)
df["volatility_7d_ma7"] = df.groupby("symbol")["volatility_7d"].rolling(7).mean().reset_index(0, drop=True)
df["volatility_7d_ma30"] = df.groupby("symbol")["volatility_7d"].rolling(30).mean().reset_index(0, drop=True)

# Realized volatility (more accurate than rolling std)
df["realized_vol_7d"] = df.groupby("symbol")["daily_return"].rolling(7).std().reset_index(0, drop=True) * np.sqrt(7)
df["realized_vol_30d"] = df.groupby("symbol")["daily_return"].rolling(30).std().reset_index(0, drop=True) * np.sqrt(30)
df["realized_vol_7d_lag1"] = df.groupby("symbol")["realized_vol_7d"].shift(1)

# Price range features (proxy for intraday volatility)
df["price_range_7d"] = df.groupby("symbol")["price"].rolling(7).apply(
    lambda x: (x.max() - x.min()) / (x.mean() + 1e-8), raw=False
).reset_index(0, drop=True)
df["price_range_30d"] = df.groupby("symbol")["price"].rolling(30).apply(
    lambda x: (x.max() - x.min()) / (x.mean() + 1e-8), raw=False
).reset_index(0, drop=True)

# Momentum and acceleration
df["return_momentum"] = df.groupby("symbol")["daily_return"].diff()
df["return_acceleration"] = df.groupby("symbol")["return_momentum"].diff()
df["volatility_momentum"] = df.groupby("symbol")["volatility_7d"].diff()

# Market-wide features (cross-asset volatility)
df["market_volatility"] = df.groupby("date")["volatility_7d"].transform("mean")
df["market_volatility_lag7"] = df.groupby("date")["volatility_7d"].transform("mean").shift(7)
df["market_realized_vol"] = df.groupby("date")["realized_vol_7d"].transform("mean")

# Relative volatility (coin vs market)
df["volatility_relative"] = df["volatility_7d"] / (df["market_volatility"] + 1e-8)
df["realized_vol_relative"] = df["realized_vol_7d"] / (df["market_realized_vol"] + 1e-8)

# Volume-volatility relationships
df["volume_volatility_ratio"] = df["volume"] / (df["volatility_7d"] + 1e-8)
df["price_volume_divergence"] = (df["daily_return"] - df["volume_change"]) ** 2

# Downside volatility (more relevant for risk)
df["downside_returns"] = df["daily_return"].where(df["daily_return"] < 0, 0)
df["downside_vol_7d"] = df.groupby("symbol")["downside_returns"].rolling(7).std().reset_index(0, drop=True) * np.sqrt(7)

# Extreme returns indicator
df["extreme_return"] = (df["daily_return"].abs() > df.groupby("symbol")["daily_return"].transform(
    lambda x: x.abs().quantile(0.95)
)).astype(int)
df["extreme_return_count_7d"] = df.groupby("symbol")["extreme_return"].rolling(7).sum().reset_index(0, drop=True)

# Time encoding - extract numeric month from date
df["month_numeric"] = pd.to_datetime(df["date"]).dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month_numeric"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_numeric"] / 12)

# Day of week (crypto markets have weekly patterns)
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

FEATURES = [
    # Price and market fundamentals
    "price",
    "market_cap",
    "volume",
    "daily_return",
    "price_ma7",
    "price_ma30",
    "market_cap_rank",
    
    # Returns
    "return_7d",
    "return_14d",
    "return_30d",
    "return_momentum",
    "return_acceleration",
    "cumulative_return",
    
    # Volume features
    "volume_change",
    "volume_ma7",
    "volume_ma30",
    "volume_momentum_7d",
    "volume_volatility_ratio",
    "price_volume_divergence",
    
    # Trend and momentum
    "ma_ratio",
    "volatility_momentum",
    
    # CRITICAL: Lagged volatility (volatility clustering)
    "volatility_7d_lag1",
    "volatility_7d_lag7",
    "volatility_7d_ma7",
    "volatility_7d_ma30",
    
    # Realized volatility
    "realized_vol_7d",
    "realized_vol_30d",
    "realized_vol_7d_lag1",
    
    # Price range (intraday volatility proxy)
    "price_range_7d",
    "price_range_30d",
    
    # Market-wide features
    "market_volatility",
    "market_volatility_lag7",
    "market_realized_vol",
    "volatility_relative",
    "realized_vol_relative",
    
    # Risk features
    "downside_vol_7d",
    "extreme_return_count_7d",
    
    # Time features
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos"
]

TARGET = "target_vol_7d"

# Drop rows with NaN features (from pct_change / rolling)
df = df.dropna(subset=FEATURES)

# Replace infinite values with NaN and drop those rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES)

def split_coin(df_coin):
    n = len(df_coin)
    train = df_coin.iloc[:int(0.7*n)]
    val   = df_coin.iloc[int(0.7*n):int(0.85*n)]
    test  = df_coin.iloc[int(0.85*n):]
    return train, val, test

train_list, val_list, test_list = [], [], []

for coin in df["symbol"].unique():
    coin_df = df[df["symbol"] == coin]
    t, v, te = split_coin(coin_df)
    train_list.append(t)
    val_list.append(v)
    test_list.append(te)

train_df = pd.concat(train_list)
val_df   = pd.concat(val_list)
test_df  = pd.concat(test_list)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val, y_val     = val_df[FEATURES], val_df[TARGET]
X_test, y_test   = test_df[FEATURES], test_df[TARGET]

model = XGBRegressor(
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

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results_str = f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}"
    print(results_str)
    return results_str, rmse, mae, r2

y_pred_test = model.predict(X_test)
eval_results, rmse, mae, r2 = evaluate(y_test, y_pred_test)

importances = model.feature_importances_
feat_imp_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
feat_imp_df = feat_imp_df.sort_values("importance", ascending=False)

feature_importance_header = "\n" + "="*60 + "\n"
feature_importance_title = "TOP 15 MOST IMPORTANT FEATURES:\n"
feature_importance_separator = "="*60 + "\n"
feature_importance_table = feat_imp_df.head(15).to_string(index=False)
feature_importance_footer = "="*60 + "\n"

print(feature_importance_header + feature_importance_title + feature_importance_separator)
print(feature_importance_table)
print(feature_importance_footer)

results_dir = os.path.join(project_root, "model")
os.makedirs(results_dir, exist_ok=True)

results_file = os.path.join(results_dir, "model_results.txt")

with open(results_file, "w") as f:
    f.write("="*60 + "\n")
    f.write("CRYPTO VOLATILITY FORECAST - MODEL RESULTS\n")
    f.write("="*60 + "\n\n")
    
    f.write("MODEL CONFIGURATION:\n")
    f.write("-"*60 + "\n")
    f.write(f"Model: XGBoost Regressor\n")
    f.write(f"Number of Estimators: {model.n_estimators}\n")
    f.write(f"Max Depth: {model.max_depth}\n")
    f.write(f"Learning Rate: {model.learning_rate}\n")
    f.write(f"Subsample: {model.subsample}\n")
    f.write(f"Colsample by Tree: {model.colsample_bytree}\n")
    f.write(f"Min Child Weight: {model.min_child_weight}\n")
    f.write(f"Gamma: {model.gamma}\n")
    f.write(f"Random State: {model.random_state}\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write("-"*60 + "\n")
    f.write(f"Total Features: {len(FEATURES)}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Validation Samples: {len(X_val)}\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"Number of Cryptocurrencies: {df['symbol'].nunique()}\n\n")
    
    f.write("EVALUATION METRICS:\n")
    f.write("-"*60 + "\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n\n")
    
    f.write(feature_importance_header)
    f.write(feature_importance_title)
    f.write(feature_importance_separator)
    f.write(feature_importance_table + "\n")
    f.write(feature_importance_footer)
    
    f.write("\nALL FEATURES (sorted by importance):\n")
    f.write("-"*60 + "\n")
    f.write(feat_imp_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*60 + "\n")
    f.write("END OF RESULTS\n")
    f.write("="*60 + "\n")

print(f"\nResults saved to: {results_file}")

model_file = os.path.join(results_dir, "volatility_model.json")
model.save_model(model_file)
print(f"Model saved to: {model_file}")

metadata = {
    "features": FEATURES,
    "target": TARGET,
    "model_type": "XGBRegressor",
    "model_params": {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "learning_rate": model.learning_rate,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
        "min_child_weight": model.min_child_weight,
        "gamma": model.gamma,
        "random_state": model.random_state
    },
    "feature_importance": feat_imp_df.to_dict("records"),
    "evaluation_metrics": {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    },
    "data_info": {
        "n_features": len(FEATURES),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_cryptocurrencies": int(df['symbol'].nunique())
    }
}

metadata_file = os.path.join(results_dir, "model_metadata.json")
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Model metadata saved to: {metadata_file}")

model_pickle_file = os.path.join(results_dir, "volatility_model.pkl")
with open(model_pickle_file, "wb") as f:
    pickle.dump({
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metadata": metadata
    }, f)
print(f"Model (pickle format) saved to: {model_pickle_file}")

print("\n" + "="*60)
print("MODEL SAVED SUCCESSFULLY!")
print("="*60)
print(f"Model file: {model_file}")
print(f"Metadata file: {metadata_file}")
print(f"Pickle file: {model_pickle_file}")
print("\nTo load the model in Gradio:")
print("  import xgboost as xgb")
print("  model = xgb.XGBRegressor()")
print("  model.load_model('model/volatility_model.json')")
print("="*60 + "\n")

