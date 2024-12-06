import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import pandas as pd
import missingno as msno
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

import warnings

warnings.filterwarnings("ignore")

from helpers import (
    load_data,
    create_submission,
    plot_y_yhat_single,
    fit_predict_evaluate,
    my_train_test_split,
    error_metric,
)


def polynomial_regression_pipeline(
    X_train, y_train, X_test, y_test, degrees, scoring="neg_mean_squared_error", cv=5
):
    """
    Performs hyperparameter search for Polynomial Regression.

    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - degrees: List of polynomial degrees to try
    - scoring: Scoring metric for cross-validation
    - cv: Number of cross-validation folds

    Returns:
    - Best model pipeline
    - Train error
    - Test error
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures()),
            ("model", LinearRegression()),
        ]
    )

    param_grid = {
        "poly__degree": degrees,
    }

    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters for Polynomial Regression: {grid_search.best_params_}")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_error = error_metric(
        y_train, y_pred_train, np.zeros_like(y_train)
    )  # Assuming c=0 for simplicity
    test_error = error_metric(y_test, y_pred_test, np.zeros_like(y_test))

    return best_model, train_error, test_error, grid_search.best_params_


def hyperparameter_search_polynomial_model(df):
    # Define hyperparameter ranges
    degrees = [1, 2, 3, 4, 5]

    # Split the data
    X_train, X_test, y_train, y_test, c_train, c_test = my_train_test_split(
        df[
            [
                "Age",
                "Gender",
                "Stage",
                "GeneticRisk",
                "TreatmentType",
                "ComorbidityIndex",
                "TreatmentResponse",
            ]
        ],
        df[["SurvivalTime"]],
        df[["Censored"]],
        test_size=0.2,
        random_state=42,
        dropXnan=True,
        dropYnan=True,
    )

    # Polynomial Regression
    poly_model, poly_train_error, poly_test_error, best_poly_params = (
        polynomial_regression_pipeline(
            X_train, y_train, X_test, y_test, degrees=degrees
        )
    )

    print(
        f"Polynomial Regression Train Error: {poly_train_error}, Test Error: {poly_test_error}"
    )
    best_degree = best_poly_params["poly__degree"]
    return poly_model, best_degree


def knn_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    n_neighbors,
    p_values,
    scoring="neg_mean_squared_error",
    cv=5,
):
    """
    Performs hyperparameter search for k-Nearest Neighbors.

    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - n_neighbors: List of neighbors to try
    - p_values: List of p values for distance metric
    - scoring: Scoring metric for cross-validation
    - cv: Number of cross-validation folds

    Returns:
    - Best model pipeline
    - Train error
    - Test error
    """
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", KNeighborsRegressor())]
    )

    param_grid = {"model__n_neighbors": n_neighbors, "model__p": p_values}

    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters for kNN: {grid_search.best_params_}")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_error = error_metric(
        y_train, y_pred_train, np.zeros_like(y_train)
    )  # Assuming c=0 for simplicity
    test_error = error_metric(y_test, y_pred_test, np.zeros_like(y_test))

    return best_model, train_error, test_error, grid_search.best_params_


def hyperparameter_search_knn_model(df):
    n_neighbors = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 30, 35, 40, 50, 60, 70]
    p_values = [1, 2]

    # Split the data
    X_train, X_test, y_train, y_test, c_train, c_test = my_train_test_split(
        df[
            [
                "Age",
                "Gender",
                "Stage",
                "GeneticRisk",
                "TreatmentType",
                "ComorbidityIndex",
                "TreatmentResponse",
            ]
        ],
        df[["SurvivalTime"]],
        df[["Censored"]],
        test_size=0.2,
        random_state=42,
        dropXnan=True,
        dropYnan=True,
    )

    # kNN
    knn_model, knn_train_error, knn_test_error, best_knn_params = knn_pipeline(
        X_train, y_train, X_test, y_test, n_neighbors=n_neighbors, p_values=p_values
    )

    print(f"kNN Train Error: {knn_train_error}, Test Error: {knn_test_error}")

    best_neighbour = best_knn_params["model__n_neighbors"]
    best_p = best_knn_params["model__p"]
    return knn_model, best_neighbour, best_p


def evaluate_polynomial_model(df, df_test, best_degree):

    best_poly_model = Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=best_degree)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    fit_predict_evaluate(
        "best_poly_model",
        df,
        df_test,
        best_poly_model,
        dropXnan=True,
        dropYnan=True,
        y_impute_function=None,
        test_size=0.2,
        random_state=42,
        plot=True,
        submission=True,
    )


def evaluate_knn_model(df, df_test, best_neighbour, best_p):
    best_knn_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=best_neighbour, p=best_p)),
        ]
    )

    fit_predict_evaluate(
        "best_knn_model",
        df,
        df_test,
        best_knn_model,
        dropXnan=True,
        dropYnan=True,
        y_impute_function=None,
        test_size=0.2,
        random_state=42,
        plot=True,
        submission=True,
    )


def main2():
    df, df_test = load_data()
    poly_model, best_degree = hyperparameter_search_polynomial_model(df)
    knn_model, best_neighbour, best_p = hyperparameter_search_knn_model(df)
    evaluate_polynomial_model(df, df_test, best_degree)
    evaluate_knn_model(df, df_test, best_neighbour, best_p)


if __name__ == "__main__":
    main2()
