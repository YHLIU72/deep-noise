from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
class MyLossFunc(nn.Module):
    def __init__(self):
        super(MyLossFunc, self).__init__()
        pass
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)  # pt 是预测为正确的概率
        
        # 计算 Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 根据 reduction 方式进行归约
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        elif self.reduction == 'none':
            pass  # 不进行归约
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        
        # 如果 alpha 不为 0，则乘以 alpha
        if self.alpha != 0:
            focal_loss = focal_loss * self.alpha
        
        return focal_loss