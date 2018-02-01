import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_heaveside(inputs, eps=0.01):
    return torch.clamp(0.5 * (1. + inputs / eps), 0., 1.)


class ROC_AUC_loss(nn.Module):
    heaveside_funcs = {"sigmoid": F.sigmoid, "linear": linear_heaveside}

    def __init__(self, heaveside="sigmoid"):
        super(ROC_AUC_loss, self).__init__()
        self.heaveside = heaveside
        self.heaveside_func = self.heaveside_funcs[self.heaveside]

    def forward(self, outputs, targets):
        true_mask = targets.ge(0.5)
        false_mask = targets.lt(0.5)
        true_outputs = outputs.masked_select(true_mask)
        false_outputs = outputs.masked_select(false_mask)
        assert true_outputs.ndim == 1
        assert false_outputs.ndim == 1
        return -self.heaveside_func(
            true_outputs.view(1, true_outputs.size(0)) - false_outputs.view(false_outputs.size(0), 1)).mean()

