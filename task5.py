
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from helpers import plot_y_yhat_single, load_data


# Custom Stacking Model
class StackingModel(BaseEstimator, RegressorMixin):
    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model

    def fit(self, X, y):
        # Fit base models
        self.model_predictions = []
        for model in self.models:
            model.fit(X, y)
            self.model_predictions.append(model.predict(X))

        # Create new feature set for meta model
        stacked_features = np.column_stack(self.model_predictions)
        self.meta_model.fit(stacked_features, y)
        return self

    def predict(self, X):
        # Predict with base models
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        # Stack predictions for meta model
        stacked_features = np.column_stack(predictions)
        return self.meta_model.predict(stacked_features)

# Step 1: Data Preprocessing
def preprocess_data(df):
    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Standardize features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    return df_imputed

# Step 2: Define Base Models
def build_base_models():
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    xgb = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=500, random_state=42)
    return [rf, xgb, mlp]

# Step 3: Define Meta Model
def build_meta_model():
    return Ridge(alpha=1.0)

# Step 4: Train Model
def train_advanced_model(X, y):
    # Preprocess Data
    X_preprocessed = preprocess_data(X)

    # Define Models
    base_models = build_base_models()
    meta_model = build_meta_model()

    # Build Stacking Model
    stacking_model = StackingModel(models=base_models, meta_model=meta_model)

    # Train Stacking Model
    stacking_model.fit(X_preprocessed, y)

    return stacking_model

# Step 5: Evaluate Model
def evaluate_model(model, X, y):
    X_preprocessed = preprocess_data(X)
    y_pred = model.predict(X_preprocessed)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse

# Step 6: Submission
def create_submission(df, model, name, training_columns):
    data = preprocess_data(df[training_columns])

    # Predict SurvivalTime
    predictions = model.predict(data)

    # Replace NaNs with mean of predictions
    survival_mean = np.round(np.nanmean(predictions), 6)
    predictions = np.nan_to_num(predictions, nan=survival_mean)

    # Create Submission
    submission = pd.DataFrame({
        "id": data.index,
        "SurvivalTime": predictions
    })

    submission.to_csv(f"submissions/{name}.csv", index=False)
    return submission

def own_model(df, df_test):
    # Main Script
    # Split labeled data
    df['SurvivalTime'] = pd.to_numeric(df['SurvivalTime'], errors='coerce')
    df_labeled = df[df['SurvivalTime'].notna()]
    X_labeled = df_labeled.drop(columns=['SurvivalTime', 'Censored'], errors='ignore')
    y_labeled = df_labeled['SurvivalTime']

    # Train-Validation Split
    X_train, X_valid, y_train, y_valid = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

    # Train Model
    stacking_model = train_advanced_model(X_train, y_train)

    # Evaluate Model
    rmse = evaluate_model(stacking_model, X_valid, y_valid)
    print(f"Validation RMSE: {rmse}")

    # plot y yhat
    X_valid_preprocessed =  preprocess_data(X_valid)
    plot_y_yhat_single(y_valid, stacking_model.predict(X_valid_preprocessed), "stacking_model")

    # Generate Submission
    training_columns = X_labeled.columns
    submission = create_submission(df_test, stacking_model, "optional-submission-01", training_columns)


def main5():
    df, df_test = load_data()
    own_model(df, df_test)

if __name__ == "__main__":
    main5()