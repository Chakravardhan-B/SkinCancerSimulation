import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

random_seed = 42
torch.manual_seed(random_seed)

def CustomDataLoader(path, batches):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.RandomVerticalFlip(),  # 随机垂直翻转
        # transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ])  # 归一化
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ])
    ])

    train_dataset = datasets.ImageFolder(root= path + 'train', transform=train_transform)

    train_size = int(0.8 * len(train_dataset))  # 80%作为训练集
    val_size = len(train_dataset) - train_size  # 剩余作为测试集
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                               generator=torch.Generator().manual_seed(random_seed))

    test_dataset = datasets.ImageFolder(root= path + 'test', transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batches, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batches, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batches, shuffle=False)

    return train_loader, val_loader, test_loader