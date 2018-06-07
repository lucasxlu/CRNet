import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# class BiCNN(nn.Module):
#     """
#     definition of BiCNN based on VGG-11
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
#                      'std': [1, 1, 1],
#                      'imageSize': [224, 224, 3]}
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
#         self.relu1 = nn.ReLU()
#         self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
#         self.relu2 = nn.ReLU()
#         self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
#         self.relu3 = nn.ReLU()
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
#         self.relu4 = nn.ReLU()
#         self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
#         self.relu5 = nn.ReLU()
#         self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
#         self.relu6 = nn.ReLU()
#         self.mpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn7 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
#         self.relu7 = nn.ReLU()
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn8 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
#         self.relu8 = nn.ReLU()
#         self.mpool8 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.regressor = Regressor()
#         self.classifier = Classifier()
#
#         # self.fc1 = nn.Linear(512 * 7 * 7, 2048)
#         # self.fcrelu1 = nn.ReLU()
#         # self.fc2 = nn.Linear(2048, 512)
#         # self.fcrelu2 = nn.ReLU()
#         # self.fc3 = nn.Linear(512, 1)
#
#     def forward(self, x):
#         x1 = self.relu1(self.bn1(self.conv1(x)))
#         x2 = self.mpool1(x1)
#
#         x3 = self.relu2(self.bn2(self.conv2(x2)))
#         x4 = self.mpool2(x3)
#
#         x5 = self.relu3(self.bn3(self.conv3(x4)))
#         x6 = self.relu4(self.bn4(self.conv4(x5)))
#         x7 = self.mpool4(x6)
#
#         x8 = self.relu5(self.bn5(self.conv5(x7)))
#         x9 = self.relu6(self.bn6(self.conv6(x8)))
#         x10 = self.mpool6(x9)
#
#         x11 = self.relu7(self.bn7(self.conv7(x10)))
#         x12 = self.relu8(self.bn8(self.conv8(x11)))
#         x13 = self.mpool8(x12)
#
#         x13 = x13.view(-1, self.num_flat_features(x13))
#
#         reg_out = self.regressor(x13)
#         cls_out = self.classifier(x13)
#
#         return reg_out, cls_out
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#
#         return num_features
#
#
# class Regressor(nn.Module):
#
#     def __init__(self):
#         super(Regressor, self).__init__()
#
#         self.fc1 = nn.Linear(512 * 7 * 7, 2048)
#         self.fcrelu1 = nn.ReLU()
#         self.fc2 = nn.Linear(2048, 512)
#         self.fcrelu2 = nn.ReLU()
#         self.fc3 = nn.Linear(512, 1)
#
#     def forward(self, x):
#         x1 = self.fcrelu1(self.fc1(x))
#         x2 = self.fcrelu2(self.fc2(x1))
#         x3 = self.fc3(x2)
#
#         return x3
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#
#         return num_features
#
#
# class Classifier(nn.Module):
#
#     def __init__(self, num_cls=5):
#         super(Classifier, self).__init__()
#
#         self.fc1 = nn.Linear(512 * 7 * 7, 2048)
#         self.fcrelu1 = nn.ReLU()
#         self.fc2 = nn.Linear(2048, 512)
#         self.fcrelu2 = nn.ReLU()
#         self.fc3 = nn.Linear(512, num_cls)
#
#     def forward(self, x):
#         x1 = self.fcrelu1(self.fc1(x))
#         x2 = self.fcrelu2(self.fc2(x1))
#         x3 = self.fc3(x2)
#
#         return x3
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#
#         return num_features


class CRNet(nn.Module):
    """
    definition of CRNet
    """

    def __init__(self):
        super(CRNet, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}

        model_ft = models.resnet18(pretrained=True)

        self.model = model_ft
        self.regressor = Regressor(model_ft)
        self.classifier = Classifier(model_ft, num_cls=5)

    def forward(self, x):
        for name, module in self.model.named_children():
            if name != 'fc':
                x = module(x)

        reg_out = self.regressor.forward(x.view(-1, self.num_flat_features(x)))
        cls_out = self.classifier.forward(x.view(-1, self.num_flat_features(x)))

        return reg_out, cls_out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Regressor(nn.Module):

    def __init__(self, model):
        super(Regressor, self).__init__()

        num_ftrs = model.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.relu(self.fc2(x1))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x3 = self.fc3(x2)

        return x3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Classifier(nn.Module):

    def __init__(self, model, num_cls=5):
        super(Classifier, self).__init__()

        num_ftrs = model.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_cls)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.relu(self.fc2(x1))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x3 = self.fc3(x2)

        return x3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
