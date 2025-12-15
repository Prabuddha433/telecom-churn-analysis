# src/model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import os
import joblib
from data_prep import load_raw, clean_telco, encode_features, split_data
from feature_engineering import feature_engineering
from explainability import plot_feature_importance, shap_summary
from utils import save_model, evaluate_model



def train_and_save(raw_csv_path, output_model_path='model/churn_model.pkl'):
    # Load and prep
    df_raw = load_raw(raw_csv_path)
    df = clean_telco(df_raw)
    df = feature_engineering(df)
    df_enc, classes = encode_features(df, label='Churn')  # churn becomes 0/1
    print("Checking encoded data types:")
    print(df_enc.dtypes.value_counts())

    print("Checking missing values:")
    print(df_enc.isna().sum().sort_values(ascending=False).head())

    # split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_enc, target='Churn')
    # Model: Random Forest with small grid search
    param_grid = {
        'n_estimators': [100],
        'max_depth': [6, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    g = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    g.fit(X_train, y_train)
    best = g.best_estimator_
    # Evaluate on validation and test
    val_metrics = evaluate_model(best, X_val, y_val)
    test_metrics = evaluate_model(best, X_test, y_test)
    print("Best params:", g.best_params_)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    plot_feature_importance(best, X_train)
    X_shap_sample = X_train.sample(200, random_state=42)
    shap_summary(best, X_shap_sample)
    # Save model and the columns used (so dashboard can use same features)
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump({'model': best, 'columns': X_train.columns.tolist(), 'classes': classes}, output_model_path)
    print("Saved model to", output_model_path)
    return best, X_test, y_test, test_metrics

if __name__ == '__main__':
    model, X_test, y_test, metrics = train_and_save('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv', output_model_path='../model/churn_model.pkl')
