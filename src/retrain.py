"""
Quick retrain script — runs all models and saves the best one.
Run: python src/retrain.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import load_data, split_features_target, split_data, scale_features
from src.feature_engineering import add_composite_features
from src.train import MODELS, train_model, cross_validate_model, save_model
from src.evaluate import evaluate_model, build_comparison_table, print_results

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = load_data("data/raw/dataset.csv")
df = add_composite_features(df)
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

cv_results, test_results, trained_models = {}, {}, {}

for name, model in MODELS.items():
    print(f"Training {name}...")
    X_tr = X_train_sc if name == "Linear Regression" else X_train
    X_te = X_test_sc  if name == "Linear Regression" else X_test
    X_cv = X_train_sc if name == "Linear Regression" else X_train

    trained = train_model(model, X_tr, y_train)
    trained_models[name] = trained
    cv_results[name]   = cross_validate_model(model, X_cv, y_train, cv=5)
    test_results[name] = evaluate_model(trained, X_te, y_test)

    print(f"  RMSE {test_results[name]['RMSE']}  R2 {test_results[name]['R2']}")

combined = {n: {**test_results[n], **cv_results[n]} for n in MODELS}
results_df = build_comparison_table(combined)
results_df.to_csv("results/model_comparison.csv")
print_results(combined)

# Save best model
best = min(test_results, key=lambda m: test_results[m]["RMSE"])
print(f"\nBest model: {best}")
save_model(trained_models[best], "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved.")

# Update model comparison chart
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, (metric, color) in zip(axes, [("RMSE","#e74c3c"),("MAE","#e67e22"),("R2","#2ecc71")]):
    vals = [test_results[m][metric] for m in MODELS]
    bars = ax.bar(list(MODELS.keys()), vals, color=color, edgecolor="white")
    ax.set_title(metric)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
plt.suptitle("Model Comparison on Test Set", fontsize=13)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
print("Chart saved.")
