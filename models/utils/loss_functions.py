import torch
import torch.nn as nn

class LossFunctions():

    def GaussianEntropy(pred, x):
        return None

    def HeatmapMSE(pred, x):
        return 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred, x['target']), dim=(-3,-2,-1))

    def HeatmapFitGaussianEntropy(pred, x):
        return None