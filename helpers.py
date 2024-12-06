import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('data/train_data.csv', index_col=0)
    df_test = pd.read_csv('data/test_data.csv', index_col=0)
    return df, df_test

def create_submission(df, pipeline, name, drop_na_X, handle_cat=False):

    idx_with_nan = df.isnull().any(axis=1).to_numpy().nonzero()[0]

    if handle_cat:
        # Ensure categorical features are strings
        for cat_feature in df.columns:
            df[cat_feature] = df[cat_feature].astype(str)

    # predict row by row if not in idx_with_nan
    predictions = []
    for idx, row in df.iterrows():
        if idx in idx_with_nan:
            predictions.append(np.nan)
        else:
            X = row
            X = pd.DataFrame(X).T
            # Ensure categorical features are strings
            prediction = pipeline.predict(X)[0]
            predictions.append(prediction)
    submission = pd.DataFrame(predictions, columns=["SurvivalTime"])
    submission["id"] = np.arange(0, len(submission))
    submission = submission[["id", "SurvivalTime"]]

    # if SurvivalTime has nan values replace them with mean
    survival_mean = np.round(submission["SurvivalTime"].mean(),6)
    submission["SurvivalTime"] = submission["SurvivalTime"].fillna(survival_mean)
    submission.to_csv(f"submissions/{name}.csv", index=False)
    return submission


def plot_y_yhat_single(y_test: pd.Series, y_pred: np.ndarray, plot_title="plot"):
    """
    Plot y_test vs y_pred for single-target regression.

    Parameters:
    - y_test: True target values (pd.Series)
    - y_pred: Predicted target values (np.ndarray)
    - plot_title: Title of the plot
    """

    # Subset for plotting
    y_test_sample = y_test
    y_pred_sample = y_pred

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test_sample - y_pred_sample) ** 2))

    # Determine min and max for the axes
    x0 = min(y_test_sample.min(), y_pred_sample.min())
    x1 = max(y_test_sample.max(), y_pred_sample.max())

    # Create the plot
    plt.figure(figsize=(4, 4))
    plt.suptitle(f"{plot_title} (RMSE: {rmse:.4f})", fontsize=16, weight="bold")
    plt.scatter(
        y_test_sample, y_pred_sample, edgecolors="b", facecolors="none", alpha=0.7
    )
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.plot([x0, x1], [x0, x1], color="red", linestyle="--", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis("square")
    plt.xlim(x0, x1)
    plt.ylim(x0, x1)

    # Save the plot
    plt.savefig(f"images/{plot_title}.png")

    # Show the plot
    plt.show()


def error_metric(y, y_hat, c):

    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err) ** 2
    return np.sum(err) / err.shape[0]


def my_train_test_split(X, y, c, test_size=0.2, random_state=42, dropXnan = None, dropYnan = None):

    if dropXnan:
        X = X.dropna()
        y = y.loc[X.index]
        c = c.loc[X.index]

    if dropYnan:
        y = y.dropna()
        X = X.loc[y.index]
        c = c.loc[y.index]


    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=test_size, random_state=random_state)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    c_train = c_train.reset_index(drop=True)
    c_test = c_test.reset_index(drop=True)

    # reshape
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    c_train = c_train.values.ravel()
    c_test = c_test.values.ravel()

    return X_train, X_test, y_train, y_test, c_train, c_test

def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err) ** 2
    return float(np.sum(err) / err.shape[0])


def fit_predict_evaluate(run_name, df, df_test, pipeline, dropXnan, dropYnan, y_impute_function, test_size, random_state, plot=True, submission=False, fit_with_c=False):
    # Split the dataframe into features and target
    X = df[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType',
        'ComorbidityIndex', 'TreatmentResponse']]
    y = df[['SurvivalTime']]
    c = df[['Censored']]

    # Drop all rows of X and y and c where X has missing values
    if dropXnan:
        print("Dropping rows with missing X values")
        X = X.dropna()
        y = y.loc[X.index]
        c = c.loc[X.index]

    if dropYnan:
        print("Dropping rows with missing y values")
        y = y.dropna()
        X = X.loc[y.index]
        c = c.loc[y.index]

    if y_impute_function is not None:
        print("Imputing missing y values")
        y = y_impute_function(y)

    # Train-test split
    X_train, X_test, y_train, y_test, c_train, c_test = my_train_test_split(X, y, c, test_size=test_size, random_state=random_state)
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    # Fit the pipeline
    print("Fitting the pipeline")
    if fit_with_c:
        pipeline.fit(X_train, y_train, c_train)
    else:
        pipeline.fit(X_train, y_train)

    # Predict on the test set
    print("Predicting on the test set")
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    train_error = error_metric(y_train, pipeline.predict(X_train), c_train)
    test_error = error_metric(y_test, y_pred, c_test)

    errors = np.abs(y_test-y_pred)

    min_error = np.min(errors)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    if plot:
        print(f"Train error: {train_error}")
        print(f"Test error: {test_error}")
        plot_y_yhat_single(y_test, y_pred, run_name)

        # small bar chart of rounded train and test error
        plt.bar(["Train", "Test"], [round(train_error, 4), round(test_error, 4)])
        plt.title("Train and Test Error (custom error metric)")
        plt.ylabel("Error")
        plt.savefig(f"images/{run_name}_error.png")
        plt.show()

        # plot min max mean and std of test error
        plt.bar(["Min", "Max", "Mean", "Std"], [min_error, max_error, mean_error, std_error])
        plt.title("Min, Max, Mean and Std of Test Error")
        plt.ylabel("Error")
        plt.savefig(f"images/{run_name}_min_max_mean_std_error.png")
        plt.show()



    if submission:
        create_submission(df_test, pipeline, run_name, dropXnan)

    return train_error, test_error

