import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from skimage import io
from skimage.transform import resize

sys.path.append('../')
from model.models import CRNet
from config.cfg import cfg


def prepare_data(model):
    """
    prepare training and test set
    :param model:
    :return:
    """
    df = pd.read_csv('./cvsplit/jaffe.csv', sep=',')

    files = [os.path.join(cfg['jaffe_dir'], df['PIC'][i].replace('-', '.') + '.' + str(df['#'][i]) + '.tiff') for i
             in range(len(df['#'].tolist()))]
    labels = []
    for index, row in df.iterrows():
        labels.append(np.argmax(np.array(row[1: 7].tolist())))

    print(files)
    print(labels)

    X = []
    y = []
    for i in range(len(files)):
        if os.path.exists(files[i]):
            x = deep_ft(files[i], model)
            X.append(x)
            y.append(labels[i])

    return X, y


def deep_ft(imgfile, model):
    """
    extract deep features from pretrained model
    :param imgfile:
    :param model:
    :return:
    """
    print(imgfile)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image = resize(io.imread(imgfile), (224, 224), mode='constant')
    # image = cv2.imread(imgfile)
    # image = cv2.resize(image, (224, 224)).astype(np.float32)
    if image.shape == (224, 224, 3):
        image[:, :, 0] -= np.mean(image[:, :, 0])
        image[:, :, 1] -= np.mean(image[:, :, 1])
        image[:, :, 2] -= np.mean(image[:, :, 2])
    else:
        tmp = np.zeros([224, 224, 3])
        image -= np.mean(image)
        tmp[:, :, 0] = image
        tmp[:, :, 1] = image
        tmp[:, :, 2] = image
        image = tmp

    image = np.transpose(image, [2, 0, 1])
    x = torch.from_numpy(image).unsqueeze(0).float()
    x = x.to(device)

    for name, module in model.named_children():
        if name == 'model':
            for nm, mod in module.named_children():
                if nm != 'fc':
                    x = mod(x)
                else:
                    break
        elif name == 'regressor':
            x = x.view(-1, 512)

            return x.to('cpu').detach().numpy().ravel().tolist()

            # for nm, mod in module.named_children():
            #     if nm != 'fc1':
            #         x = mod.forward(x)
            #     else:
            #         return x.to('cpu').detach().numpy().ravel().tolist()


def fer_jaffe(X, y):
    """
    train and test on JAFFE with SVM
    :param X:
    :param y:
    :return:
    """
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    X_train, X_test, y_train, y_test = train_test_split(np.array(X, dtype=np.float64), np.array(y), test_size=0.3,
                                                        random_state=42)

    svc = svm.SVC(kernel='rbf')
    print('start training SVM with RBF kernel...')
    svc.fit(X_train, y_train)
    print('finish training SVM with RBF kernel...\nstart evaluating...')
    acc = svc.score(X_test, y_test)
    print('Accuracy is ', acc)


if __name__ == '__main__':
    model = CRNet()
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    print('Loading pre-trained model...')
    model.load_state_dict(torch.load(os.path.join('./model/crnet.pth')))
    model.eval()

    X, y = prepare_data(model)
    fer_jaffe(X, y)
