import torch

def diceScore(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * intersection + 1e-8) / (union + 1e-8)
    
    return dice.mean()

def iouScore(preds, targets, threshold=0.5):
    
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    
    return iou.mean()

class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds_sigmoid = torch.sigmoid(preds)
        intersection = (preds_sigmoid * targets).sum()
        dice = (2.0 * intersection + 1e-8) / (preds_sigmoid.sum() + targets.sum() + 1e-8)

        dice_loss = 1.0 - dice
        return bce_loss + dice_loss
