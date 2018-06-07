import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.append('../')
from data.data_loder import ScutFBPDataset, HotOrNotDataset
from util.utils import mkdirs_if_not_exist
from config.cfg import cfg


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25,
                inference=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not inference:
        model.train()
        print('Start training Bi-CNN...')
        for epoch in range(num_epochs):
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data['image'], data['score']

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                inputs = inputs.float()
                # labels = labels.float().view(cfg['batch_size'], 5)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished training Bi-CNN...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'exp-bi-cnn.pth'))
        print('Bi-CNN has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/exp-bi-cnn.pth')))

    model.eval()

    predicted_labels = []
    gt_labels = []
    for data in test_dataloader:
        images, labels = data['image'], data['score']
        labels = labels.to(device)
        images = images.to(device)
        outputs = model.forward(images)
        # _, predicted = torch.max(outputs.data, 1)

        for out in F.softmax(outputs).to("cpu"):
            tmp = 0
            for i in range(1, 6, 1):
                tmp += out[i - 1] * i
            predicted_labels.append(float(tmp.detach().numpy()))

        gt_labels += labels.cpu().numpy().tolist()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_lr = round(mean_absolute_error(np.array(gt_labels), np.array(predicted_labels).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(gt_labels), np.array(predicted_labels).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_labels), np.array(predicted_labels).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of Bi-CNN is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Bi-CNN is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Bi-CNN is {0}===================='.format(pc))

    col = ['gt', 'pred']
    df = pd.DataFrame([gt_labels, predicted_labels], columns=col)
    df.to_excel("./output_eccv.xlsx", sheet_name='Output', index=False)


def run_bicnn_scutfbp():
    """
    train ResNet18 with 5-Class output
    :return:
    """
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    df = pd.read_excel('./cvsplit/SCUT-FBP.xlsx', sheet_name='Sheet1')
    X_train, X_test, y_train, y_test = train_test_split(df['Image'].tolist(), df['Attractiveness label'],
                                                        test_size=0.2, random_state=0)

    train_dataset = ScutFBPDataset(f_list=X_train, f_labels=y_train, transform=transforms.Compose([
        transforms.ColorJitter(),
        transforms.Resize(224),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ]))

    test_dataset = ScutFBPDataset(f_list=X_test, f_labels=y_test, transform=transforms.Compose([
        transforms.ColorJitter(),
        transforms.Resize(224),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4)

    train_model(model=model_ft, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=50, inference=False)


def run_bicnn_eccv(cv_split):
    """
    train and test ECCV HotOrNot dataset
    :param cv_split:
    :return:
    """
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 7)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    train_dataset = HotOrNotDataset(cv_split=cv_split, train=True, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ]))

    test_dataset = HotOrNotDataset(cv_split=cv_split, train=False, transform=transforms.Compose([
        transforms.ColorJitter(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4)

    train_model(model=model_ft, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=50, inference=False)


if __name__ == '__main__':
    # run_bicnn_scutfbp()
    run_bicnn_eccv(cv_split=1)
