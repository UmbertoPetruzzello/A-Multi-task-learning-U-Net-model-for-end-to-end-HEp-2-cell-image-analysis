import torch.nn as nn
import torchgeometry.losses as losses
import torch.nn.functional as F
import torch
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, inputs, targets):
        targets = targets.type(torch.int64)
        num_classes = 1
        true_1_hot = torch.eye(num_classes + 1)[targets.cpu().squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(inputs)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, inputs.ndimension()))
        n = torch.mul(probas, true_1_hot).sum(dims)
        n =  torch.mul(2.,n)
        d = torch.add(probas, true_1_hot).sum(dims)
        dice_score = torch.divide(n,d)
        return torch.sub(1, dice_score.mean())


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.dice_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')
        self.ce_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')
        self.bce_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')

    def forward(self, preds, mask, intensity):
        crossEntropy = nn.CrossEntropyLoss()
        binaryCrossEntropy = nn.BCEWithLogitsLoss()
        diceLoss = DiceLoss()

        label = label.long()
        intensity = intensity.unsqueeze(1)
        intensity = intensity.float()

        loss0 = diceLoss(preds[0], mask)
        loss1 = crossEntropy(preds[1], label)
        loss2 = binaryCrossEntropy(preds[2], intensity)

        self.dice_loss += loss0
        self.ce_loss += loss1
        self.bce_loss += loss2
        
        loss_0 = (1/3) * loss0
        loss_1 = (1/3) * loss1
        loss_2 = (1/3) * loss2
        
        return loss_0 + loss_1 + loss_2, loss0, loss1, loss2

    def get_losses(self, c):
        return self.dice_loss.item()/c, self.bce_loss.item()/c
    
    def set_losses(self):
        self.dice_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')
        self.ce_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')
        self.bce_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')