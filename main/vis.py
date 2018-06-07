import os
import sys

import numpy as np
import scipy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import cv2
from PIL import Image
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

sys.path.append('../')
from model.models import CRNet
from config.cfg import cfg


def viz(img_path, model=CRNet()):
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('./model/crnet.pth'))
    model = model.to(device)
    model.eval()

    image = resize(io.imread(img_path), (224, 224), mode='constant')
    image[:, :, 0] -= np.mean(image[:, :, 0])
    image[:, :, 1] -= np.mean(image[:, :, 1])
    image[:, :, 2] -= np.mean(image[:, :, 2])

    image = np.transpose(image, [2, 0, 1])
    input = torch.from_numpy(image).unsqueeze(0).float()
    input = input.to(device)

    for idx, module in model.named_children():
        if idx == 'model':
            for idx, mod in module.named_children():
                # print(idx)
                if idx != 'avgpool':
                    input = mod(input)
                else:
                    # mat = np.transpose(input[0, [1, 1, 1], :, :].data.cpu().numpy(), [1, 2, 0])
                    mat = np.transpose(input[0, :, :, :].data.cpu().numpy(), [1, 2, 0])
                    mat = np.mean(mat, axis=2).reshape([mat.shape[0], mat.shape[0], 1])
                    print(mat.shape)
                    # mat = resize(mat, (224, 224), mode='constant')
                    mat = cv2.resize(mat, (224, 224))
                    org = resize(io.imread(img_path), (224, 224), mode='constant')

                    dst = np.zeros([224, 224, 3])
                    dst[:, :, 0] = 0.2 * org[:, :, 0] + 0.8 * mat
                    dst[:, :, 1] = 0.2 * org[:, :, 1] + 0.8 * mat
                    dst[:, :, 2] = 0.2 * org[:, :, 2] + 0.8 * mat

                    # dst = org + mat

                    # plt.figure("Image")
                    # plt.imshow(org)
                    # plt.axis('on')
                    # plt.title('image')
                    # plt.show()

                    # cv2.imshow('ft', dst)
                    # cv2.waitKey()

                    if not os.path.exists('./feature_viz/'):
                        os.makedirs('./feature_viz/')

                    scipy.misc.imsave('./feature_viz/' + os.path.basename(img_path).split('.')[0] + '.jpg', dst)
                    break


if __name__ == '__main__':
    filelist = [cfg['scut_fbp_dir'] + "/SCUT-FBP-%d.jpg" % _ for _ in [101, 57, 242, 380, 192, 469, 241, 174]]
    for f in filelist:
        viz(f, CRNet())
