from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_list,
        y_pred_list,
        average="binary",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_list,
        y_pred_list,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true_list, y_pred_list)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "true_poetry_pred_poetry": int(matrix[0, 0]),
        "true_poetry_pred_rap": int(matrix[0, 1]),
        "true_rap_pred_poetry": int(matrix[1, 0]),
        "true_rap_pred_rap": int(matrix[1, 1]),
    }
