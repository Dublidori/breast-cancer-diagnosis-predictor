"""
model.py -- Breast cancer classification model.

Loads the sklearn breast cancer dataset, trains a Random Forest classifier
inside a Pipeline (StandardScaler + RandomForestClassifier), evaluates it,
and exposes prediction functions.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from dataclasses import dataclass


@dataclass
class ModelArtifacts:
    """Container for everything the app needs from the trained model."""
    pipeline: Pipeline
    feature_names: list
    target_names: list
    feature_importances: np.ndarray
    accuracy: float
    classification_report: str
    feature_means: np.ndarray
    feature_mins: np.ndarray
    feature_maxs: np.ndarray
    dataset: pd.DataFrame  # full dataset with features + diagnosis
    labels: np.ndarray     # 0=malignant, 1=benign


def train_model(random_state=42, test_size=0.2):
    """
    Load data, split, train pipeline, evaluate, return artifacts.
    Designed to be called once via st.cache_resource.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0=malignant, 1=benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=data.target_names,
    )

    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_means = np.mean(data.data, axis=0)
    feature_mins = np.min(data.data, axis=0)
    feature_maxs = np.max(data.data, axis=0)

    dataset = X.copy()
    dataset["diagnosis"] = ["Malignant" if v == 0 else "Benign" for v in y]

    return ModelArtifacts(
        pipeline=pipeline,
        feature_names=list(data.feature_names),
        target_names=list(data.target_names),
        feature_importances=importances,
        accuracy=accuracy,
        classification_report=report,
        feature_means=feature_means,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        dataset=dataset,
        labels=y,
    )


def predict(artifacts, input_values):
    """
    Make a prediction on a single sample.

    Returns:
        (label, confidence) where label is 'Benign' or 'Malignant'
        and confidence is a float in [0, 1].
    """
    input_2d = input_values.reshape(1, -1)
    prediction = artifacts.pipeline.predict(input_2d)[0]
    probabilities = artifacts.pipeline.predict_proba(input_2d)[0]

    label = "Benign" if prediction == 1 else "Malignant"
    confidence = probabilities[prediction]

    return label, confidence
