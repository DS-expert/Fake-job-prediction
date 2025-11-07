import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

def train_adaboost(X_train, y_train):
    """
    Train an AdaBoost Classifier on the provided training data.
    Parameters:
    X_train (pd.DataFrame or np.ndarray): The training feature set.
    y_train (pd.DataFrame or np.ndarray): The training target / label set.

    *return: Return the adaboost trained model.
    """
    adaboost = AdaBoostClassifier()
    adaboost.fit(X_train, y_train)
    return adaboost

def train_logistic(X_train, y_train):

    """
    Train an Logistic Regression on the provided training data.
    Parameters:
    X_train (pd.DataFrame or np.ndarray): The training feature set.
    y_train (pd.DataFrame or np.ndarray): The training target / label set.

    *return: Return the Logistic Regression trained model.
    """

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)

    return logistic