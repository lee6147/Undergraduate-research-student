import time
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def _make_model(model_name: str, random_state: int = 42):
    models = {
        "SVM": lambda rs: SVC(kernel="rbf", random_state=rs),
        "MLP": lambda rs: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=rs),
        "RF": lambda rs: RandomForestClassifier(n_estimators=100, random_state=rs),
    }
    return models[model_name](random_state)


def train_classical_model(model_name: str, X_train, y_train, X_test, y_test, random_state: int = 42):
    model = _make_model(model_name, random_state)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "model": model_name,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_f1": f1_score(y_train, y_pred_train, average="macro"),
        "test_f1": f1_score(y_test, y_pred_test, average="macro"),
        "train_time": train_time,
    }
