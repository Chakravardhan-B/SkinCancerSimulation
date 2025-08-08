import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm, tqdm_gui,tqdm_notebook
from PerClassEvaluation import PerClassEvaluation
from ROC_DCA import roc

def test_model(model, device, val_loader):
    TestingMetrics = []
    TestingMetricsALL = []
    Preds = []
    Probs = []
    Labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):  # 加载测试数据集
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 同上
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            Preds.append(predicted.cpu().numpy())
            Probs.append(outputs.cpu().numpy())
            Labels.append(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()

    print(f'Testing Accuracy: {100 * correct / total}%')
    Preds = np.concatenate(Preds)
    Labels = np.concatenate(Labels)
    report = classification_report(Labels, Preds, digits = 4)
    return report
    
    # Preds = np.concatenate(Preds)
    # Probs = np.concatenate(Probs)
    # Labels = np.concatenate(Labels)
    # TestingMetrics.append(PerClassEvaluation(Preds, Labels))
    # bacc = balanced_accuracy_score(Labels, Preds)
    # pre = precision_score(Labels, Preds, average='weighted')
    # rec = recall_score(Labels, Preds, average='weighted')
    # f1 = f1_score(Labels, Preds, average='weighted')
    # tpr, fpr, auc = roc(Labels, Probs)
    # auc = [auc]
    # # tpr, fpr, auc = roc(Labels, Probs)
    # TestingMetricsALL.append([100 * correct / total, bacc * 100, pre * 100, rec * 100, f1 * 100])

    # return TestingMetrics, TestingMetricsALL, tpr, fpr, auc


def test_SavedModel(model, device, val_loader):
    TestingMetrics = []
    TestingMetricsALL = []
    Preds = []
    Probs = []
    Labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):  # 加载测试数据集
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 同上
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            Preds.append(predicted.cpu().numpy())
            Probs.append(outputs.cpu().numpy())
            Labels.append(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()

    print(f'Testing Accuracy: {100 * correct / total}%')
    Preds = np.concatenate(Preds)
    Probs = np.concatenate(Probs)
    Labels = np.concatenate(Labels)
    tpr, fpr, auc = roc(Labels, Probs)


    return tpr, fpr , auc
