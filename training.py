import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, tqdm_gui,tqdm_notebook
from PerClassEvaluation import PerClassEvaluation
from ROC_DCA import roc
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--num_classes', type=int, default=3)
args = parser.parse_args()

def train_model(model, device, train_loader, val_loader, criterion, optimizer,network, num_epochs, data, opt):
    TotalLoss = []
    best_acc = 0
    TestingMetrics = []
    TestingMetricsALL = []
    TPR, FPR, AUC = 0, 0, []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # break
        model.eval() #切换到测试模型
        correct = 0
        total = 0
        Preds = []
        Labels = []
        Probs = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader): #加载测试数据集
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) #同上
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                Preds.append(predicted.cpu().numpy())
                Probs.append(outputs.cpu().numpy())
                Labels.append(labels.cpu().numpy())
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Val Accuracy: {100 * correct / total}%')
        if (100 * correct / total) > best_acc:
            best_acc = 100 * correct / total
            if not os.path.exists('./SavedModel/' + data + network + '/' + opt + '/'):
                os.makedirs('./SavedModel/' + data + network + '/' + opt + '/')
            torch.save(model.state_dict(),'./SavedModel/' + data + network + '/' + opt + '/BestValModel.pth')

        TotalLoss.append([running_loss / len(train_loader)])
        Preds = np.concatenate(Preds)
        Probs = np.concatenate(Probs)
        Labels = np.concatenate(Labels)
        TestingMetrics.append(PerClassEvaluation(Preds, Labels))
        bacc = balanced_accuracy_score(Labels,Preds)
        pre = precision_score(Labels,Preds, average='weighted')
        rec = recall_score(Labels, Preds,average='weighted')
        f1 = f1_score(Labels,Preds,average='weighted')
        print(Labels.shape, Probs.shape)
        tpr, fpr, auc = roc(Labels, Probs)
        TestingMetricsALL.append([100 * correct / total, bacc*100,pre*100,rec*100,f1*100])

    return TotalLoss, TestingMetrics, TestingMetricsALL

