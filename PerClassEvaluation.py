import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
def PerClassEvaluation(Preds, Labels):
    classes = set(Labels)
    per_class_metrics = [[] for idx in range(len(classes))]
    for c in classes:
        y_true = [1 if l == c else 0 for l in Labels]
        y_pred = [1 if p == c else 0 for p in Preds]
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='binary')
        rec = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        per_class_metrics[c] = [acc, bacc, prec, rec, f1]
    return per_class_metrics


