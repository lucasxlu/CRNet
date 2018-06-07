import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.append('../')
from model.models import CRNet
from model.losses import CRLoss
from data.data_loder import ScutFBPDataset, HotOrNotDataset
from util.utils import mkdirs_if_not_exist
from config.cfg import cfg


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25,
                inference=False):
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not inference:
        print('Start training CRNet...')
        for epoch in range(num_epochs):
            model.train()
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, scores, classes = data['image'], data['score'], data['class']

                inputs = inputs.to(device)
                scores = scores.to(device)
                classes = classes.to(device)

                optimizer.zero_grad()

                inputs = inputs.float()
                scores = scores.float().view(cfg['batch_size'], 1)
                # classes = classes.int().view(cfg['batch_size'], 3)

                reg_out, cls_out = model(inputs)
                loss = criterion(cls_out, classes, reg_out, scores)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished training CRNet...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'crnet.pth'))
        print('CRNet has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/crnet.pth')))

    model.eval()

    print('Start testing CRNet...')
    predicted_labels = []
    gt_labels = []
    filenames = []
    for data in test_dataloader:
        images, scores, classes, filename = data['image'], data['score'], data['class'], data['filename']
        images = images.to(device)

        reg_out, cls_out = model.forward(images)

        # bat_list = []
        # for out in F.softmax(cls_out).to("cpu"):
        #     tmp = 0
        #     for i in range(0, 3, 1):
        #         tmp += out[i] * (i - 1)
        #     bat_list.append(float(tmp.detach().numpy()))

        # predicted_labels += (0.6 * reg_out.to("cpu").detach().numpy() + 0.4 * np.array(bat_list)).tolist()

        predicted_labels += reg_out.to("cpu").detach().numpy().tolist()
        gt_labels += scores.to("cpu").detach().numpy().tolist()
        filenames += filename

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_lr = round(mean_absolute_error(np.array(gt_labels), np.array(predicted_labels).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(gt_labels), np.array(predicted_labels).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_labels), np.array(predicted_labels).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of CRNet is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of CRNet is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of CRNet is {0}===================='.format(pc))

    col = ['filename', 'gt', 'pred']
    df = pd.DataFrame([[filenames[i], gt_labels[i], predicted_labels[i][0]] for i in range(len(gt_labels))],
                      columns=col)
    df.to_excel("./output.xlsx", sheet_name='Output', index=False)
    print('Output Excel has been generated~')


def run_crnet_scutfbp(model, epoch=30):
    criterion = CRLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    df = pd.read_excel('./cvsplit/SCUT-FBP.xlsx', sheet_name='Sheet1')
    X_train, X_test, y_train, y_test = train_test_split(df['Image'].tolist(), df['Attractiveness label'],
                                                        test_size=0.2, random_state=0)

    print('start loading SCUT-FBP dataset...')
    train_dataset = ScutFBPDataset(f_list=X_train, f_labels=y_train, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[1, 1, 1])
    ]))

    test_dataset = ScutFBPDataset(f_list=X_test, f_labels=y_test, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[1, 1, 1])
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4)

    print('finish loading SCUT-FBP dataset...')
    train_model(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epoch,
                inference=False)


def run_crnet_eccv(model, cv_split=1, epoch=30):
    """
    train and test ECCV HotOrNot dataset
    :param cv_split:
    :return:
    """
    criterion = CRLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    print('start loading ECCV HotOrNot dataset...')
    train_dataset = HotOrNotDataset(cv_split=cv_split, train=True, transform=transforms.Compose([
        transforms.Resize(227),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[1, 1, 1])
    ]))

    test_dataset = HotOrNotDataset(cv_split=cv_split, train=False, transform=transforms.Compose([
        transforms.Resize(227),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[1, 1, 1])
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4, drop_last=True)

    print('finish loading ECCV HotOrNot dataset...')

    train_model(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epoch,
                inference=False)


if __name__ == '__main__':
    run_crnet_scutfbp(model=CRNet(), epoch=30)
    # run_crnet_eccv(model=CRNet(), cv_split=1, epoch=30)
