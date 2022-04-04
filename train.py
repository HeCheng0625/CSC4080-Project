import time
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

from my_dataset import MyDataset
from my_model import myResNet50

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    augment_transforms = [transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=180, resample=False, expand=False, center=None, fill=None),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])]

    train_data = MyDataset('train_images', 'train.csv', True, transform, True, augment_transforms)
    valid_data = MyDataset('train_images', 'train.csv', False, transform)

    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, BATCH_SIZE)

    net = myResNet50
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)

    print('\nTraining start!\n')
    start = time.time()
    max_acc = 0 # resnet18 pretrained and data augmentation
    reached = 0  # which epoch reached the max accuracy

    loss_list = []
    acc_list = []
    train_acc_list = []

    for epoch in range(1, MAX_EPOCH + 1):

        loss_mean = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)   # (B, C)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # results
            # torch.max return value: (B, ), index: (B, )
            _, predicted = torch.max(outputs.data, 1)   # (B, )

            # calculate the accuracy of this training iteration
            val_true = labels.view(-1).cpu().numpy().tolist()
            val_pred = predicted.view(-1).cpu().numpy().tolist()
            train_acc = accuracy_score(val_true, val_pred)

            if (i==0 or i==11 or i==23):
                train_acc_list.append(train_acc)

            # print log
            loss_mean += loss.item()
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, train_acc))
                loss_mean = 0.

        # validate the model
        if epoch % val_interval == 0:

            loss_val = 0.
            net.eval()
            val_true, val_pred = [], []
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)

                    loss_val += loss.item()

                    val_true.extend(labels.view(-1).cpu().numpy().tolist())
                    val_pred.extend(predicted.view(-1).cpu().numpy().tolist())

                val_acc = accuracy_score(val_true, val_pred)
                print(classification_report(val_true, val_pred, zero_division=0))

                if val_acc > max_acc:
                    max_acc = val_acc
                    reached = epoch
                    torch.save(net.state_dict(), 'resnet50_pretrained')

                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, val_acc))

            loss_list.append(loss_val)
            acc_list.append(val_acc)

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(
        MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(
        max_acc, reached))

    print(loss_list)
    print(acc_list)
    print(train_acc_list)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')

    # hyper-parameters
    seed = 42
    MAX_EPOCH = 50
    BATCH_SIZE = 64
    LR = 0.001
    weight_decay = 1e-3
    log_interval = 2
    val_interval = 1

    set_seed(seed)
    print('random seed:', seed)
    main()