import torch
import torch.nn.functional as F

from core.losses import ncc
from core.utils import diff


class SimilarityTerm(torch.nn.Module):
    """ Base class for similarity step """
    def __init__(self, tau_config):
        super(SimilarityTerm, self).__init__()
        self.tau_ = torch.nn.Parameter(torch.tensor(tau_config.init), requires_grad=tau_config.requires_grad)
        self.tau_train_scale = tau_config.train_scale

    def energy(self, tar, warped_src):
        """ Compute the energy """
        return NotImplementedError

    def grad(self, tar, warped_src, disp):
        return NotImplementedError

    def forward(self, tar, warped_src, disp):
        return self.grad(tar, warped_src, disp) * self.tau_ * self.tau_train_scale


class SSDSimilarity(SimilarityTerm):
    def __init__(self, tau_config, autograd=True):
        super().__init__(tau_config)
        self.autograd = autograd

    def energy(self, tar, warped_src):
        """ SSD similarity energy """
        return F.mse_loss(tar, warped_src)

    def grad(self, tar, warped_src, disp):
        """ Gradient step of the SSD similairty loss"""
        if self.autograd:
            e = self.energy(tar, warped_src)
            grad = torch.autograd.grad(e, disp, create_graph=self.training)[0]
            # correct the magnitude by number of points because of the averaging in F.mse_loss()
            grad = grad * torch.tensor(tar.shape).prod()
        else:
            # take spatial derivatives
            wapred_src_deriv = torch.cat([diff(warped_src, dim=d) for d in range(disp.shape[1])], dim=1)
            grad = 2 * (warped_src - tar) * wapred_src_deriv  # gradient
        return grad


class NCCSimilarity(SimilarityTerm):
    def __init__(self, tau_config):
        super().__init__(tau_config)

    def energy(self, tar, warped_src):
        return ncc(tar, warped_src)

    def grad(self, tar, warped_src, disp):
        e = self.energy(tar, warped_src)
        grad = torch.autograd.grad(e, disp, create_graph=self.training)[0]
        grad = grad * torch.tensor(tar.shape).prod()
        return grad
