import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_scoreimport numpy as np
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
        tpr, fpr, auc = roc(Labels, Probs)
    return TotalLoss, TestingMetrics, TestingMetricsALL
from tqdm import tqdm
from PerClassEvaluation import PerClassEvaluation
from ROC_DCA import roc
from data_loader import get_loader
from models import get_model

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')
args = parser.parse_args()

def train_model(model, device, train_loader, val_loader, criterion, optimizer, network, num_epochs, data, opt):
    TotalLoss = []
    best_acc = 0
    TestingMetrics = []
    TestingMetricsALL = []
    TPR, FPR, AUC = 0, 0, []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        Preds, Labels, Probs = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                Preds.append(predicted.cpu().numpy())
                Probs.append(outputs.cpu().numpy())
                Labels.append(labels.cpu().numpy())
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f'Loss: {running_loss / len(train_loader):.4f}, Val Accuracy: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f'./SavedModel/{data}{network}/{opt}/'
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'BestValModel.pth'))

        TotalLoss.append([running_loss / len(train_loader)])
        Preds = np.concatenate(Preds)
        Probs = np.concatenate(Probs)
        Labels = np.concatenate(Labels)
        TestingMetrics.append(PerClassEvaluation(Preds, Labels))
        bacc = balanced_accuracy_score(Labels, Preds)
        pre = precision_score(Labels, Preds, average='weighted')
        rec = recall_score(Labels, Preds, average='weighted')
        f1 = f1_score(Labels, Preds, average='weighted')
        tpr, fpr, auc = roc(Labels, Probs)
        TestingMetricsALL.append([val_acc, bacc*100, pre*100, rec*100, f1*100])

    return TotalLoss, TestingMetrics, TestingMetricsALL

def main():
    print("Preparing for training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    print("Loading dataset from:", args.data_path)
    train_loader, val_loader = get_loader(args.data_path)

    # Load model
    print("Creating model with", args.num_classes, "output classes")
    model = get_model(args.num_classes)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training
    train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        network='CustomModel',
        num_epochs=args.epochs,
        data='FPV',
        opt='Adam'
    )

if __name__ == '__main__':
    main()
