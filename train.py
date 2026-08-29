import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("salary_processed.csv")

X = df.drop("Salary", axis=1)
y = df["Salary"]


# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ===========================
# Models
# ===========================

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge":
        Ridge(),

    "Lasso":
        Lasso(),

    "Decision Tree":
        DecisionTreeRegressor(random_state=42),

    "Random Forest":
        RandomForestRegressor(random_state=42),

    "Extra Trees":
        ExtraTreesRegressor(random_state=42),

    "Gradient Boosting":
        GradientBoostingRegressor(random_state=42),

    "AdaBoost":
        AdaBoostRegressor(random_state=42),

    "XGBoost":
        XGBRegressor(
            random_state=42,
            verbosity=0
        ),

    "CatBoost":
        CatBoostRegressor(
            random_state=42,
            verbose=0
        )
}


# ===========================
# Train Models
# ===========================

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

print("All Models Trained Successfully.")


# ===========================
# Model Evaluation
# ===========================

results = []

for name, model in trained_models.items():

    prediction = model.predict(X_test)

    r2 = r2_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)

    results.append({

        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse

    })

results = pd.DataFrame(results)

results = results.sort_values(
    by="R2 Score",
    ascending=False
)

print("\nModel Comparison\n")
print(results)


# ===========================
# Best Model
# ===========================

best_model_name = results.iloc[0]["Model"]

print("\nBest Model :", best_model_name)

best_model = trained_models[best_model_name]


# ===========================
# Cross Validation
# ===========================

scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nCross Validation Scores")
print(scores)

print("\nAverage Score :", scores.mean())
print("Standard Deviation :", scores.std())


# =====================================================
# Hyperparameter Tuning
# =====================================================

if best_model_name == "XGBoost":

    params = {

        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]

    }

    search = RandomizedSearchCV(

        estimator=XGBRegressor(
            random_state=42,
            verbosity=0
        ),

        param_distributions=params,

        n_iter=20,

        cv=5,

        scoring="r2",

        random_state=42,

        n_jobs=-1

    )

    search.fit(X_train, y_train)

    print("\nBest Parameters")
    print(search.best_params_)

    print("\nBest Cross Validation Score")
    print(search.best_score_)

    best_model = search.best_estimator_


# ===========================
# Final Evaluation
# ===========================

prediction = best_model.predict(X_test)

print("\nFinal Model Performance")

print("R2 Score :", r2_score(y_test, prediction))
print("MAE :", mean_absolute_error(y_test, prediction))
print("RMSE :", np.sqrt(mean_squared_error(y_test, prediction)))


# ===========================
# Save Model
# ===========================

joblib.dump(best_model, "salary_model.pkl")

print("\nModel Saved Successfully.")


# ===========================
# Feature Names
# ===========================

print("\nFeature Order")
print(X.columns)