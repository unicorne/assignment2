from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from helpers import plot_y_yhat_single, load_data

class GroupMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_column, target_columns):
        """
        Parameters:
        - group_column: str, the column used to define the groups (e.g., 'Censored').
        - target_columns: list of str, the columns to impute based on group means.
        """
        self.group_column = group_column
        self.target_columns = target_columns
        self.group_means_ = {}

    def fit(self, X, y=None):
        """
        Compute the mean for each group in the specified target columns.
        """
        # Ensure X is a pandas DataFrame
        X = pd.DataFrame(X)

        # Calculate group means for each target column
        self.group_means_ = {}
        for col in self.target_columns:
            self.group_means_[col] = (
                X.groupby(self.group_column)[col].mean().to_dict()
            )
        return self

    def transform(self, X):
        """
        Impute missing values in the target columns based on group means.
        """
        # Ensure X is a pandas DataFrame
        X = pd.DataFrame(X)

        # Fill missing values with group means
        for col in self.target_columns:
            group_means = self.group_means_.get(col, {})
            X[col] = X.apply(
                lambda row: group_means.get(row[self.group_column], np.nan)
                if pd.isna(row[col])
                else row[col],
                axis=1,
            )
        return X



def semi_supervised_approach(df, df_test):

    # Ensure SurvivalTime is numeric
    df['SurvivalTime'] = pd.to_numeric(df['SurvivalTime'], errors='coerce')

    # Debugging: Check for NaN in SurvivalTime
    print(f"Total NaN values in SurvivalTime: {df['SurvivalTime'].isna().sum()}")

    # Instantiate and fit the imputer
    cols = [c for c in df.columns if c!='SurvivalTime']
    target_columns2 = [c for c in cols if c!='Censored']
    imputer = GroupMeanImputer(group_column='Censored', target_columns=cols)
    imputer.fit(df[cols])
    df[cols] = imputer.transform(df[cols])


    # Separate labeled and unlabeled data
    df_labeled = df[df['SurvivalTime'].notna()]  # Rows with a target
    df_unlabeled = df[df['SurvivalTime'].isna()]  # Rows without a target

    #imputer = GroupMeanImputer(group_column='Censored', target_columns=target_columns)
    #imputer.fit(X_unlabeled)
    #X_unlabeled = imputer.transform(X_unlabeled)

    # Define labeled data
    X_labeled = df_labeled.drop(columns=['SurvivalTime'])  # Features for labeled data
    y_labeled = df_labeled['SurvivalTime']  # Target for labeled data

    # Define unlabeled data (features only)


    X_unlabeled = df_unlabeled.drop(columns=['SurvivalTime'])  # Features for unlabeled data
    target_columns = [c for c in X_unlabeled.columns if c!='Censored']

    # Combine labeled and unlabeled data for imputation and dimensionality reduction
    X_combined = pd.concat([X_labeled, X_unlabeled], axis=0)

    # Debugging: Check shapes and column consistency
    print(f"Shape of X_labeled: {X_labeled.shape}")
    print(f"Shape of X_unlabeled: {X_unlabeled.shape}")
    print(f"Shape of X_combined: {X_combined.shape}")

    # Ensure numeric data for combined features
    X_combined = X_combined.apply(pd.to_numeric, errors='coerce')

    # Step 1: Impute Missing Values
    #imputer = SimpleImputer(strategy="mean")  # Use the best imputation strategy from Task 3.1
    #X_combined_imputed = imputer.fit_transform(X_combined)
    X_combined_imputed = X_combined

    # Ensure no NaN remains in imputed data
    X_combined_imputed = np.nan_to_num(X_combined_imputed)
    print(f"NaN in combined data after imputation: {np.isnan(X_combined_imputed).sum()}")

    # Extract labeled data after imputation
    X_labeled_imputed = X_combined_imputed[:len(X_labeled)]

    # Train Baseline Linear Regression
    lr = LinearRegression()
    lr.fit(X_labeled_imputed, y_labeled)

    # Evaluate baseline Linear Regression model
    y_pred = lr.predict(X_labeled_imputed)
    baseline_rmse = np.sqrt(mean_squared_error(y_labeled, y_pred))
    print(f"Baseline RMSE (Linear Regression on imputed data): {baseline_rmse}")

    # Step 3: Train Isomap for Dimensionality Reduction
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined_imputed)

    # Train Isomap on combined data
    isomap = Isomap(n_components=2)  # Experiment with n_components as needed
    isomap.fit(X_combined_scaled)

    # Wrap Isomap in FrozenTransformer for pipeline integration
    class FrozenTransformer(BaseEstimator):
        def __init__(self, fitted_transformer):
            self.fitted_transformer = fitted_transformer

        def __getattr__(self, name):
            return getattr(self.fitted_transformer, name)

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return self.fitted_transformer.transform(X)

        def fit_transform(self, X, y=None):
            return self.fitted_transformer.transform(X)

    frozen_isomap = FrozenTransformer(isomap)

    # Step 4: Build Pipeline with Isomap and Train Linear Regression
    pipe = make_pipeline(
        #SimpleImputer(strategy="mean"),  # Imputation
        GroupMeanImputer(group_column='Censored', target_columns=target_columns),
        StandardScaler(),  # Scaling
        frozen_isomap,  # Dimensionality Reduction
        LinearRegression()  # Linear Regression
    )

    # Train the pipeline on labeled data
    pipe.fit(X_labeled, y_labeled)

    # Evaluate the pipeline
    y_pred_pipe = pipe.predict(X_labeled)
    pipeline_rmse = np.sqrt(mean_squared_error(y_labeled, y_pred_pipe))
    print(f"Pipeline RMSE (Linear Regression with Isomap): {pipeline_rmse}")

    # Compare Baseline and Pipeline Results
    print(f"Baseline RMSE: {baseline_rmse}")
    print(f"Pipeline RMSE: {pipeline_rmse}")

    # plot y yhat
    plot_y_yhat_single(y_labeled, y_pred_pipe, "linear_regression_with_isomap")

def main4():
    # Load data
    df, df_test = load_data()
    semi_supervised_approach(df, df_test)

if __name__ == "__main__":
    main4()