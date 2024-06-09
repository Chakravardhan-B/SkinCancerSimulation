import torch
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
from data_loader import CustomDataLoader
from models import CustomCNN
from training import train_model
import numpy as np
from GetLog import GetLog
import os
from testing import test_model

def main(network):
    classes = 7
    epochs = 50
    data = 'HAM10000/'
    opt = 'AdamW'
    model = CustomCNN(num_classes=classes, network=network).to('cuda')

    num_params = sum(p.numel() for p in model.parameters())
    print(f"The network has {num_params} parameters.")

    # print(model)

    train_loader, val_loader, test_loader = CustomDataLoader('./data/' + data, batches=32)

    # optimizer = torch.optim.SGD(params=[{'params':model.features.parameters()}], lr=0.01, weight_decay=5e-4,momentum=0.9)
    # optimizer = torch.optim.Adam(params=[{'params':model.features.parameters()}], lr=0.001, weight_decay=0.02, betas=(0.99,0.98))
    # optimizer = torch.optim.Adam(params=[{'params':model.features.parameters()}], lr=0.001, weight_decay=0.02)
    optimizer = torch.optim.AdamW(params=[{'params': model.features.parameters()}], betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss()

    Losses, ValMetrics, ValMetricsALL = train_model(model, 'cuda', train_loader, val_loader, criterion, optimizer,
                                                    network, epochs, data, opt)
    model.load_state_dict(torch.load('./SavedModel/' + data + network + '/' + opt + '/BestValModel.pth'))
    TestingMetrics, TestingMetricsALL, tpr, fpr, auc = test_model(model, 'cuda', test_loader)

    ACC = GetLog(classes)
    BACC = GetLog(classes)
    PRE = GetLog(classes)
    REC = GetLog(classes)
    F1 = GetLog(classes)
    Test_ACC = GetLog(classes)
    Test_BACC = GetLog(classes)
    Test_PRE = GetLog(classes)
    Test_REC = GetLog(classes)
    Test_F1 = GetLog(classes)
    Val_ALL = GetLog(5)
    Test_ALL = GetLog(5)
    TPR_ALL = GetLog(epochs)
    FPR_ALL = GetLog(epochs)
    AUC_ALL = GetLog(epochs)
    # print(ValMetrics[0][6])
    for j in range(0, epochs):
        for i in range(0, classes):
            ACC[i].append(ValMetrics[j][i][0])
            BACC[i].append(ValMetrics[j][i][1])
            PRE[i].append(ValMetrics[j][i][2])
            REC[i].append(ValMetrics[j][i][3])
            F1[i].append(ValMetrics[j][i][4])
        Val_ALL[0].append(ValMetricsALL[j][0])
        Val_ALL[1].append(ValMetricsALL[j][1])
        Val_ALL[2].append(ValMetricsALL[j][2])
        Val_ALL[3].append(ValMetricsALL[j][3])
        Val_ALL[4].append(ValMetricsALL[j][4])
        # TPR_ALL[j].append(ValMetricsALL[j][5])
        # FPR_ALL[j].append(ValMetricsALL[j][6])
        # AUC_ALL[j].append(ValMetricsALL[j][7])

    for i in range(0, classes):
        Test_ACC[i].append(TestingMetrics[0][i][0])
        Test_BACC[i].append(TestingMetrics[0][i][1])
        Test_PRE[i].append(TestingMetrics[0][i][2])
        Test_REC[i].append(TestingMetrics[0][i][3])
        Test_F1[i].append(TestingMetrics[0][i][4])
    Test_ALL[0].append(TestingMetricsALL[0][0])
    Test_ALL[1].append(TestingMetricsALL[0][1])
    Test_ALL[2].append(TestingMetricsALL[0][2])
    Test_ALL[3].append(TestingMetricsALL[0][3])
    Test_ALL[4].append(TestingMetricsALL[0][4])

    Hold = [ACC, BACC, PRE, REC, F1, Val_ALL[0], Val_ALL[1], Val_ALL[2], Val_ALL[3], Val_ALL[4],
            TPR_ALL, FPR_ALL, AUC_ALL, Test_ACC, Test_BACC, Test_PRE, Test_REC, Test_F1, Test_ALL[0],
            Test_ALL[1], Test_ALL[2], Test_ALL[3], Test_ALL[4]]
    Map = ['VAL_ACC', 'VAL_BACC', 'VAL_PRE', 'VAL_REC', 'VAL_F1', 'VAL_ACC_ALL', 'VAL_BACC_ALL', 'VAL_PRE_ALL',
           'VAL_REC_ALL',
           'VAL_F1_ALL', 'TPR_ALL', 'FPR_ALL', 'AUC_ALL', 'Test_ACC', 'Test_BACC', 'Test_PRE', 'Test_REC', 'Test_F1',
           'Test_ACC_ALL', 'Test_BACC_ALL', 'Test_PRE_ALL', 'Test_REC_ALL',
           'Test_F1_ALL']
    for i in range(0, len(Map)):
        np_log = np.array(Hold[i], dtype=float)
        if 'VAL' in Map[i]:
            if not os.path.exists('./results/' + data + network + '/' + opt + '/Val/'):
                os.makedirs('./results/' + data + network + '/' + opt + '/Val/')
            loc = './results/' + data + network + '/' + opt + '/Val/' + Map[i] + '.csv'
            np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')
        if 'Test' in Map[i]:
            if not os.path.exists('./results/' + data + network + '/' + opt + '/Test/'):
                os.makedirs('./results/' + data + network + '/' + opt + '/Test/')
            loc = './results/' + data + network + '/' + opt + '/Test/' + Map[i] + '.csv'
            np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')

    TPR = np.array(tpr, dtype=float)
    FPR = np.array(fpr, dtype=float)
    AUC = np.array(auc, dtype=float)
    np.savetxt('./results/' + data + network + '/' + opt + '/Test/' + 'TPR.csv', TPR, delimiter=',', fmt='%.6f')
    np.savetxt('./results/' + data + network + '/' + opt + '/Test/' + 'FPR.csv', FPR, delimiter=',', fmt='%.6f')
    np.savetxt('./results/' + data + network + '/' + opt + '/Test/' + 'AUC.csv', AUC, delimiter=',', fmt='%.6f')

main(network = 'convnext_base')
main(network = 'convnext_small')
main(network = 'convnext_large')
main(network = 'convnext_base')
main(network = 'maxvit_t')
main(network = 'swin_b')
main(network = 'swin_s')
main(network = 'swin_t')
main(network = 'swin_v2_b')
main(network = 'swin_v2_s')
main(network = 'swin_v2_t')
main(network = 'vit_l_16')