# src/data_prep.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_raw(path):
    df = pd.read_csv(path)
    return df

def clean_telco(df):
    df = df.copy()
    # Standard cleanup for Telco dataset
    # Drop customerID
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    # Convert TotalCharges to numeric (some are spaces)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    # Convert senior citizen if encoded as 0/1 to int
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    # Strip spaces from object columns
    obj_cols = df.select_dtypes(include='object').columns
    for c in obj_cols:
        df[c] = df[c].str.strip()
    # Replace "No internet service" and "No phone service" to "No"
    df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
    return df

def encode_features(df, label='Churn'):
    df = df.copy()
    le = LabelEncoder()

    if label in df.columns:
        df[label] = le.fit_transform(df[label])
        churn_classes = list(le.classes_)
    else:
        churn_classes = None

    # One-hot encode object columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ---- FINAL SAFETY CONVERSION ----

    # Convert category columns (like TenureGroup) to numeric
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].cat.codes

    # Convert bool to int (True/False -> 1/0)
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # Fill any remaining missing values
    df = df.fillna(0)

    return df, churn_classes


def split_data(df, target='Churn', test_size=0.2, val_size=0.1, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y)
    # split X_temp into validation and test
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1-relative_val_size), random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    # quick local test
    df = load_raw('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = clean_telco(df)
    df_enc, classes = encode_features(df)
    df_enc.to_csv('../data/churn_clean.csv', index=False)
    print("Saved cleaned data to ../data/churn_clean.csv")
