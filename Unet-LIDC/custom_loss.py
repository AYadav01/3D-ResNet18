import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss
    """
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class InvSoftDiceLoss(nn.Module):
    """
    Inverted Soft Dice Loss
    """
    def __init__(self):
        super(InvSoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

