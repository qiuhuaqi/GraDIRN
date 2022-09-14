import torch
from torch import nn as nn
from core.utils import convNd


class Regulariser(torch.nn.Module):
    """Base regulariser class"""
    def __init__(self):
        super(Regulariser, self).__init__()

    def energy(self, x):
        raise NotImplementedError

    def get_theta(self):
        """ return all parameters of the regularization """
        return self.named_parameters()

    def get_vis(self):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.grad(x)


class CNNRegulariser(Regulariser):
    def __init__(self, config):
        super(CNNRegulariser, self).__init__()
        ndim = config.ndim
        num_layers = config.num_layers
        num_channels = config.num_channels

        self.layers = nn.ModuleList()
        indim = ndim + 2 if config.input_images else ndim
        self.layers.append(nn.Sequential(convNd(ndim, indim, num_channels, a=0.2), nn.LeakyReLU(0.2)))

        for i in range(num_layers-2):
            self.layers.append(
                nn.Sequential(convNd(ndim, num_channels, num_channels, a=0.2), nn.LeakyReLU(0.2)))

        # last layer with no activation
        self.layers.append(convNd(ndim, num_channels, ndim))

    def grad(self, x):
        for layer in self.layers:
            x = layer(x)
        return x