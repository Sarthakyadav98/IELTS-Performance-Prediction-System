import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbosity=0,
    ),
}


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def cross_validate_model(model, X, y, cv: int = 5):
    neg_mse = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return {
        "cv_rmse_mean": round(rmse_scores.mean(), 4),
        "cv_rmse_std": round(rmse_scores.std(), 4),
        "cv_r2_mean": round(r2_scores.mean(), 4),
        "cv_r2_std": round(r2_scores.std(), 4),
    }


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)
