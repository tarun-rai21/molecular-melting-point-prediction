import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Trains XGBoost regression model with log-transformed target.
    """

    # Log-transform target (important for stability)
    y_log = np.log1p(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=test_size, random_state=random_state
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",

        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.4,

        learning_rate=0.05,
        n_estimators=800,

        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Validation performance (convert back to original scale)
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    y_val_original = np.expm1(y_val)

    mae = mean_absolute_error(y_val_original, val_preds)

    print(f"Validation MAE: {mae:.4f}")

    return model, mae


def predict(model, X):
    """
    Predicts and converts back from log-scale.
    """
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    return preds
