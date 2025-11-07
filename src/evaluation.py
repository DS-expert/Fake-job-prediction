import pandas as pd
import numpy as np
from src.train import train_adaboost
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model on the test dataset and return accuracy, ROC AUC Score and classification report.

    Parameters:
    model: Trained machine learning model
    X_test: Feature of the test dataset
    y_test: True labels of the test dataset

    *returns: accuracy_score, roc_auc_score, classification_report
    """
    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    return {
        "Accuracy Score": acc_score,
        "ROC AUC Score ": roc,
        "Classification Report" : cr
    }
