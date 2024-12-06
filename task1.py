import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import pandas as pd
import missingno as msno
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

from helpers import (
    load_data,
    create_submission,
    plot_y_yhat_single,
    fit_predict_evaluate,
)


def describe_data(df):
    print("Statistics of the different outcomes of the features:")
    for col in df.columns:
        if col == "Age" or col == "SurvivalTime":
            print(f"{col}: {df[col].describe()}")
        else:
            print(f"{col}: {df[col].unique()}")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    msno.bar(df, ax=axs[0, 0])
    msno.heatmap(df, ax=axs[0, 1])
    msno.dendrogram(df, ax=axs[1, 0])
    msno.matrix(df, ax=axs[1, 1])
    plt.tight_layout()
    # save figure
    plt.savefig("images/msno.png")
    # Count number of missing values
    print("Number of missing values:")
    print(df.isnull().sum())

    # get dataframe where none of the values are missing
    df_no_missing = df.dropna()
    valid_datapoints = len(df_no_missing[df_no_missing["Censored"] == 0])
    print(f"Number of valid datapoints: {valid_datapoints}")


def baseline_model(df, df_test):
    # Create a pipeline with StandardScaler and the model
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

    fit_predict_evaluate(
        "linear_regression",
        df,
        df_test,
        pipeline,
        dropXnan=True,
        dropYnan=True,
        y_impute_function=None,
        test_size=0.2,
        random_state=42,
        plot=True,
        submission=True,
    )


def explore_randomness_of_random_seed(df, df_test):
    test_errors = []
    train_errors = []
    for random_seed in np.arange(1, 10, 1):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        train_error, test_error = fit_predict_evaluate(
            f"linear_regression_{random_seed}",
            df,
            df_test,
            pipeline,
            dropXnan=True,
            dropYnan=True,
            y_impute_function=None,
            test_size=0.2,
            random_state=random_seed,
            plot=False,
            submission=False,
        )

        test_errors.append(test_error)
        train_errors.append(train_error)

    # print errors
    print("Errors over different random seeds:")
    print("Test errors:")
    print(test_errors)
    print("Train errors:")
    print(train_errors)

    fig, ax = plt.subplots()
    # plot two bars next to each other
    barWidth = 0.3
    r1 = np.arange(len(test_errors))
    r2 = [x + barWidth for x in r1]
    ax.bar(
        r1, test_errors, color="b", width=barWidth, edgecolor="grey", label="Test error"
    )
    ax.bar(
        r2,
        train_errors,
        color="r",
        width=barWidth,
        edgecolor="grey",
        label="Train error",
    )
    ax.set_xticks([r + barWidth for r in range(len(test_errors))])
    ax.set_xticklabels(np.arange(1, 10, 1))
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Error")
    ax.legend()
    ax.set_title("Error over different random seeds")
    plt.savefig(f"images/Error_over_random_seeds.png")
    fig.show()


class GradientDescentModel:
    def __init__(
        self, learning_rate=0.0002, epochs=5000, regularization=None, alpha=0.01
    ):
        """
        Parameters:
        - learning_rate: Step size for gradient descent.
        - epochs: Number of iterations.
        - regularization: 'lasso', 'ridge', or None.
        - alpha: Regularization strength (lambda).
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y, c):
        X = np.array(X)
        y = np.array(y)
        c = np.array(c)

        n, d = X.shape
        self.weights = np.zeros(d)  # Initialize weights

        for epoch in range(self.epochs):
            y_hat = np.dot(X, self.weights)
            gradient = np.zeros(d)

            for i in range(n):
                err = y[i] - y_hat[i]
                grad = -2 * (1 - c[i]) * err
                if c[i] == 1 and y[i] > y_hat[i]:
                    grad += -2 * c[i] * err
                gradient += grad * X[i]

            gradient /= n

            # Add regularization to the gradient
            if self.regularization == "ridge":  # L2 Regularization
                gradient += 2 * self.alpha * self.weights
            elif self.regularization == "lasso":  # L1 Regularization
                gradient += self.alpha * np.sign(self.weights)

            self.weights -= self.learning_rate * gradient

            # (Optional) Track loss for debugging
            if epoch % 1000 == 0:
                loss = np.mean(
                    (1 - c) * (y - y_hat) ** 2 + c * np.maximum(0, y - y_hat) ** 2
                )
                if self.regularization == "ridge":
                    loss += self.alpha * np.sum(self.weights**2)
                elif self.regularization == "lasso":
                    loss += self.alpha * np.sum(np.abs(self.weights))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained. Call `fit` first.")
        X = np.array(X)
        return np.dot(X, self.weights)


def learn_with_gradient_descent(df, df_test):
    # Train and predict
    model = GradientDescentModel(
        learning_rate=0.0002, epochs=20000, regularization="lasso", alpha=0.02
    )

    train_error, test_error = fit_predict_evaluate(
        "gradient_descent_lasso",
        df,
        df_test,
        model,
        dropXnan=True,
        dropYnan=True,
        y_impute_function=None,
        test_size=0.2,
        random_state=42,
        plot=True,
        submission=True,
        fit_with_c=True,
    )


def main1():
    # Load data
    df, df_test = load_data()

    # Describe data
    describe_data(df)

    # Baseline model
    baseline_model(df, df_test)

    # Explore randomness of random seed
    explore_randomness_of_random_seed(df, df_test)

    # Learn with gradient descent
    learn_with_gradient_descent(df, df_test)


if __name__ == "__main__":
    main1()
