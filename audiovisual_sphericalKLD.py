import numpy as np
import torch
import torch.cuda
import torch.nn as nn

class AV_KLWeightedLoss(nn.Module):
    def __init__(self, alpha = 0.75):
        super(AV_KLWeightedLoss, self).__init__()
        self.epsilon = 1e-8  # the parameter to make sure the denominator non-zero
        self.alpha = alpha

    def forward(self, map_pred, map_gtd, map_aem, w_AUC=0):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        # Raise warning if the input maps are not the same size
        if map_pred.shape != map_gtd.shape:
            raise ValueError("The input maps must be the same size")
        
        bs, H, W = map_pred.shape
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        map_aem = map_aem.float()

        map_pred = map_pred.view(bs, -1)  # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(bs, -1)  # change the map_pred into a tensor with n rows and 1 cols
        map_aem = map_aem.view(bs, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1, _ = torch.min(map_pred, dim=1, keepdim=True)
        max1, _ = torch.max(map_pred, dim=1, keepdim=True)

        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

        min2, _ = torch.min(map_gtd, dim=1, keepdim=True)
        max2, _ = torch.max(map_gtd, dim=1, keepdim=True)

        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

        min3, _ = torch.min(map_aem, dim=1, keepdim=True)
        max3, _ = torch.max(map_aem, dim=1, keepdim=True)

        map_aem = (map_aem - min3) / (max3 - min3 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

        map_pred = map_pred / (torch.sum(map_pred, dim=1, keepdim=True) + self.epsilon)  # normalization step to make sure that the map_pred sum to 1
        map_gtd = map_gtd / (torch.sum(map_gtd, dim=1, keepdim=True) + self.epsilon)  # normalization step to make sure that the map_gtd sum to 1
        map_aem = map_aem / (torch.sum(map_aem, dim=1, keepdim=True) + self.epsilon)  # normalization step to make sure that the map_aem sum to 1
        
        # Calculate the weights
        weight = np.zeros((H, W))
        theta_range = np.linspace(0, np.pi, num=H + 1)
        dtheta = np.pi / H
        dphi = 2 * np.pi / W
        for theta_idx in range(H):
            weight[theta_idx, :] = dphi * (
                    np.sin(theta_range[theta_idx]) + np.sin(theta_range[theta_idx + 1])) / 2 * dtheta

        weight = torch.Tensor(weight).unsqueeze(0).repeat(bs, 1, 1).view(bs, -1).to(map_pred.device)

        kl_vis = weight * map_gtd * torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        kl_aud = weight * map_aem * torch.log(map_aem / (map_pred + self.epsilon) + self.epsilon)


        return torch.sum(kl_vis * self.alpha  + (1-self.alpha) * kl_aud, dim=1)
    