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
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool

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


def mean_impute_X(df,df_test):
    # Create a pipeline with SimpleImputer for X and StandardScaler
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing X values with mean
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    # Evaluate the pipeline
    train_error_mean_impue, test_error_mean_impute = fit_predict_evaluate(
        "linear_regression_with_X_impute_mean", 
        df,
        df_test,
        pipeline, 
        dropXnan=False,  # Keep rows with NaN in X; handled by the imputer
        dropYnan=True,   # Drop rows with NaN in y
        y_impute_function=None, 
        test_size=0.2, 
        random_state=42, 
        plot=True, 
        submission=False
    )
    return train_error_mean_impue, test_error_mean_impute

def median_impute_X(df,df_test):
    # Create a pipeline with SimpleImputer for X and StandardScaler
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Impute missing X values with median
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    # Evaluate the pipeline
    train_error_median_impue, test_error_median_impute = fit_predict_evaluate(
        "linear_regression_with_X_impute_median", 
        df,
        df_test,
        pipeline, 
        dropXnan=False,  # Keep rows with NaN in X; handled by the imputer
        dropYnan=True,   # Drop rows with NaN in y
        y_impute_function=None, 
        test_size=0.2, 
        random_state=42, 
        plot=True, 
        submission=False
    )
    return train_error_median_impue, test_error_median_impute

def knn_impute_X(df, df_test):
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    # Evaluate the pipeline
    train_error_knn_impute, test_error_knn_impute = fit_predict_evaluate(
        "linear_regression_with_X_impute_knn", 
        df, 
        df_test,
        pipeline, 
        dropXnan=False,  # Keep rows with NaN in X; handled by the imputer
        dropYnan=True,   # Drop rows with NaN in y
        y_impute_function=None, 
        test_size=0.2, 
        random_state=42, 
        plot=True, 
        submission=False
    )
    return train_error_knn_impute, test_error_knn_impute

def impute_mean_X_and_y(df, df_test):
    def y_impute_mean(y):
        return y.fillna(y.mean())

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    # Evaluate the pipeline
    train_error_y_mean_impute, test_error_y_mean_impute = fit_predict_evaluate(
        "linear_regression_with_X_y_impute_mean", 
        df, 
        df_test,
        pipeline, 
        dropXnan=False,  # Drop rows with NaN in X
        dropYnan=False,  # Keep rows with NaN in y; handled by the imputer
        y_impute_function=y_impute_mean, 
        test_size=0.2, 
        random_state=42, 
        plot=True, 
        submission=False
    )
    return train_error_y_mean_impute, test_error_y_mean_impute

def impute_median_X_and_y(df, df_test):
    def y_impute_median(y):
        return y.fillna(y.median())

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    # Evaluate the pipeline
    train_error_y_mean_impute, test_error_y_mean_impute = fit_predict_evaluate(
        "linear_regression_with_X_y_impute_mean", 
        df, 
        df_test,
        pipeline, 
        dropXnan=False,  # Drop rows with NaN in X
        dropYnan=False,  # Keep rows with NaN in y; handled by the imputer
        y_impute_function=y_impute_median, 
        test_size=0.2, 
        random_state=42, 
        plot=True, 
        submission=False
    )
    return train_error_y_mean_impute, test_error_y_mean_impute

def knn_impute_X_and_y(df, df_test):
    def y_impute_knn(y):
        imputer = KNNImputer(n_neighbors=5)
        return pd.DataFrame(imputer.fit_transform(y))

    pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)), 
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    # Evaluate the pipeline
    train_error_y_knn_impute, test_error_y_knn_impute = fit_predict_evaluate(
        "linear_regression_with_X_y_impute_knn", 
        df, 
        df_test,
        pipeline, 
        dropXnan=False,  # Drop rows with NaN in X
        dropYnan=False,  # Keep rows with NaN in y; handled by the imputer
        y_impute_function=y_impute_knn, 
        test_size=0.2, 
        random_state=42, 
        plot=True, 
        submission=True
    )

    return train_error_y_knn_impute, test_error_y_knn_impute

def compare_imputers(test_error_mean_impute, test_error_median_impute, test_error_knn_impute, test_error_y_mean_impute, test_error_y_median_impute, test_error_y_knn_impute):
    # bar chart of all imputations of X and X,y 

    # plot min max mean and std of test error
    plt.bar(["Mean", "Median", "KNN"], [test_error_mean_impute, test_error_median_impute, test_error_knn_impute])
    plt.title("Test Error of X Imputation Methods")
    plt.ylabel("Error")
    plt.savefig(f"images/X_imputation_methods.png")


    plt.bar(["Mean", "Median", "KNN"], [test_error_y_mean_impute, test_error_y_mean_impute, test_error_y_knn_impute])
    plt.title("Test Error of Y Imputation Methods")
    plt.ylabel("Error")
    plt.savefig(f"images/Y_imputation_methods.png")

    # print them in a table
    print("Test errors of different imputation methods:")
    print("X imputation methods:")
    print(f"Mean: {test_error_mean_impute}")
    print(f"Median: {test_error_median_impute}")
    print(f"KNN: {test_error_knn_impute}")
    print("X and Y imputation methods:")
    print(f"Mean: {test_error_y_mean_impute}")
    print(f"Median: {test_error_y_median_impute}")
    print(f"KNN: {test_error_y_knn_impute}")

def build_best_model(df, df_test):
    def y_impute_knn(y):
        imputer = KNNImputer(n_neighbors=5)
        return pd.DataFrame(imputer.fit_transform(y))

    pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)), 
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor())
    ])

    # Evaluate the pipeline
    train_error_y_knn_impute, test_error_y_knn_impute = fit_predict_evaluate(
        "linear_regression_with_y_impute_knn", 
        df,
        df_test,
        pipeline, 
        dropXnan=False,  # Drop rows with NaN in X
        dropYnan=False,  # Keep rows with NaN in y; handled by the imputer
        y_impute_function=y_impute_knn, 
        test_size=0.2, 
        random_state=42, 
        plot=True, 
        submission=True
    )

def build_decision_tree(df, df_test):

    # Create a pipeline with DecisionTreeRegressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DecisionTreeRegressor(random_state=42))
    ])

    # Evaluate the pipeline
    fit_predict_evaluate(
        "decision_tree",
        df,
        df_test,
        pipeline,
        dropXnan=True,  # Keep rows with missing values in X
        dropYnan=True,   # Drop rows with missing values in y
        y_impute_function=None,
        test_size=0.2,
        random_state=42,
        plot=True,
        submission=False,
    )

def build_hist_gradient(df, df_test):

    # Create a pipeline with HistGradientBoostingRegressor
    pipeline = Pipeline([
        ('model', HistGradientBoostingRegressor(random_state=42))
    ])

    # Evaluate the pipeline
    fit_predict_evaluate(
        "hist_gradient_boosting",
        df,
        df_test,
        pipeline,
        dropXnan=False,  # Keep rows with missing values in X
        dropYnan=True,   # Drop rows with missing values in y
        y_impute_function=None,
        test_size=0.2,
        random_state=42,
        plot=True,
        submission=False
    )

def create_submission_no_target(df, pipeline, name, handle_cat=False):
    """
    Create a submission for a test set without SurvivalTime or Censored columns.
    
    Parameters:
    - df: The test dataframe containing features for prediction.
    - pipeline: The trained CatBoost pipeline/model.
    - name: The name for the output CSV file.
    - handle_cat: Whether to preprocess categorical features (convert to strings).
    
    Returns:
    - submission: A dataframe containing predictions.
    """
    if handle_cat:
        # Ensure categorical features are strings
        for cat_feature in df.columns:
            df[cat_feature] = df[cat_feature].astype(str)

    # Predict survival times for all rows
    predictions = pipeline.predict(df)
    
    # Create the submission dataframe
    submission = pd.DataFrame({
        "id": np.arange(len(df)),
        "SurvivalTime": predictions
    })
    
    # Save to CSV
    submission.to_csv(f"submissions/{name}.csv", index=False)
    print(f"Submission saved to 'submissions/{name}.csv'")
    
    return submission


def try_catboost(df, df_test):
    # Assuming df_new is already loaded
    X = df[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 
                'ComorbidityIndex', 'TreatmentResponse']]
    y = df['SurvivalTime']
    c = df['Censored']  # 1 if censored, 0 if uncensored

    # Encode categorical variables (if any) or let CatBoost handle it
    categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType']


    #median_survival = y.median()
    #y = y.fillna(median_survival)
    # remove data where y is missing
    y = y.dropna()
    X = X.loc[y.index]
    c = c.loc[y.index]

    # Create a weight column: assign lower weights to censored data
    weights = np.where(c == 1, 0.5, 1.0)

    # Ensure categorical features are strings
    for cat_feature in categorical_features:
        X[cat_feature] = X[cat_feature].astype(str)

    # Create CatBoost Pool
    train_pool = Pool(data=X, label=y, cat_features=categorical_features, weight=weights)

    # Initialize CatBoost Regressor
    catboost_model = CatBoostRegressor(
        iterations=10000,
        l2_leaf_reg=100,     
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100,
    )

    # Train the model
    catboost_model.fit(train_pool)

    # Predictions
    predictions = catboost_model.predict(X)

    # Evaluate predictions on uncensored data
    uncensored_mask = c == 0  # Mask for uncensored records
    y_true = y[uncensored_mask]
    y_pred = predictions[uncensored_mask]

    # Calculate MAE or RMSE
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"Mean Absolute Error (MAE) on uncensored data: {mae}")

    # plot y yhat
    print(y_true.shape)
    print(y_pred.shape)

    plot_y_yhat_single(y_true, y_pred, "catboost_regressor")

    # create submission
    create_submission_no_target(df_test, catboost_model, "handle-missing-submission-02", handle_cat=True)

# Robust preprocessing to handle NaNs
def preprocess_survival_data(df):
    # Create a copy to avoid modifying the original DataFrame
    df_cleaned = df.copy()
    
    # Fill or drop NaNs based on your domain knowledge
    # Example strategies:
    # 1. Drop rows with NaNs in critical columns
    df_cleaned = df_cleaned.dropna(subset=['SurvivalTime', 'Censored'])
    
    # 2. Convert Censored to integer, replacing NaNs
    df_cleaned['Censored'] = df_cleaned['Censored'].fillna(0).astype(int)
    
    # 3. Ensure SurvivalTime is numeric and handle any remaining NaNs
    df_cleaned['SurvivalTime'] = pd.to_numeric(df_cleaned['SurvivalTime'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['SurvivalTime'])
    
    return df_cleaned

# Comprehensive NaN checking function
def check_nan_values(df):
    print("NaN Value Checks:")
    print("-" * 40)
    
    # Check overall NaNs
    print("Total NaNs in each column:")
    print(df.isna().sum())

def catboost_aft(df, df_test):
    
    # Check specific columns of interest
    columns_to_check = ['SurvivalTime', 'Censored', 'Age', 'Gender', 'Stage', 
                        'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']
    
    # Detailed NaN checking
    for col in columns_to_check:
        print(f"\nColumn: {col}")
        print(f"Total NaNs: {df[col].isna().sum()}")
        print(f"Unique non-NaN values: {df[col].dropna().unique()}")
        
    # Check data types
    print("\nData Types:")
    print(df[columns_to_check].dtypes)

    # Perform NaN checking
    check_nan_values(df)


    # Preprocess the data
    df_cleaned = preprocess_survival_data(df)

    # Recheck NaNs after preprocessing
    check_nan_values(df_cleaned)

    # Prepare features and targets
    X = df_cleaned[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 
                    'ComorbidityIndex', 'TreatmentResponse']]
    y = df_cleaned['SurvivalTime']
    c = df_cleaned['Censored']

    # Ensure categorical features are properly encoded
    categorical_features = ['Gender', 'Stage', 'TreatmentType']
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Prepare AFT target
    lower_bound = y.values
    upper_bound = np.where(c.values == 1, y.max() * 10, y.values)

    # Construct AFT target
    aft_target = list(zip(lower_bound, upper_bound))

    # Extensive target validation
    print("\nAFT Target Validation:")
    print(f"Number of targets: {len(aft_target)}")
    print(f"Sample of first 5 targets: {aft_target[:5]}")
    print(f"Any NaNs in targets: {any(np.isnan(t[0]) or np.isnan(t[1]) for t in aft_target)}")

    # Create CatBoost Pool
    train_pool = Pool(data=X, label=aft_target, cat_features=categorical_features)

    # Initialize and train CatBoost Regressor
    catboost_aft_model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.05,
        depth=6,
        loss_function='SurvivalAft:dist=Normal',
        eval_metric='SurvivalAft',
        random_seed=42,
        verbose=100
    )

    # Fit the model
    catboost_aft_model.fit(train_pool)


    # Predictions
    predictions = catboost_aft_model.predict(X)

    # Evaluate predictions on uncensored data
    uncensored_mask = c == 0  # Mask for uncensored records
    y_true = y[uncensored_mask]
    y_pred = predictions[uncensored_mask]

    # Calculate MAE or RMSE
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"Mean Absolute Error (MAE) on uncensored data: {mae}")

    # plot y yhat
    plot_y_yhat_single(y_true, y_pred, "catboost_regressor")

    # create submission
    create_submission_no_target(df_test, catboost_aft_model, "handle-missing-submission-02", handle_cat=True)



def main3():
    # Load the training data
    df ,df_test = load_data()

    _, test_error_mean_impute = mean_impute_X(df,df_test)
    _, test_error_median_impute = median_impute_X(df,df_test)
    _, test_error_knn_impute = knn_impute_X(df, df_test)
    _, test_error_y_mean_impute = impute_mean_X_and_y(df, df_test)
    _, test_error_y_median_impute = impute_median_X_and_y(df, df_test)
    _, test_error_y_knn_impute = knn_impute_X_and_y(df, df_test)
    compare_imputers(test_error_mean_impute, test_error_median_impute, test_error_knn_impute, test_error_y_mean_impute, test_error_y_median_impute, test_error_y_knn_impute)
    build_best_model(df, df_test)
    build_decision_tree(df, df_test)
    build_hist_gradient(df, df_test)
    try_catboost(df, df_test)
    catboost_aft(df, df_test)

if __name__ == "__main__":
    main3()