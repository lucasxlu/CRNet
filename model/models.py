import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
