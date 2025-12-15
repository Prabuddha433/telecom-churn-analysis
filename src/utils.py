# src/utils.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:,1]
    except:
        y_proba = None
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    return metrics
