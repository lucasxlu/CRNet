import torch.nn as nn


class CRLoss(nn.Module):
    """
    CRLoss definition
    """

    def __init__(self, cls_w=0.4, reg_w=0.6):
        super(CRLoss, self).__init__()

        self.cls_w = cls_w
        self.reg_w = reg_w

        self.class_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()

    def forward(self, cls_pred, cls_gt, score_pred, score_gt):
        class_loss = self.class_criterion(cls_pred, cls_gt)
        regression_loss = self.regression_criterion(score_pred, score_gt)

        cr_loss = self.cls_w * class_loss + self.reg_w * regression_loss

        return cr_loss
