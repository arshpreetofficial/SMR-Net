import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred)
    spe_list = []

    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spe = tn / (tn + fp + 1e-8)
        spe_list.append(spe)

    spe = np.mean(spe_list)

    return acc, sen, spe, f1
