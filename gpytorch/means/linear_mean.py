#!/usr/bin/env python3

import torch

from .mean import Mean


class LinearMean(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

class PolynomialMean(Mean):
    def __init__(self, input_size, degree: int, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, degree)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.register_buffer("degree", torch.Tensor([degree]))
        
    def forward(self, x):
        CENTERING_CONST = 0.5
        x = x - CENTERING_CONST
        degrees = torch.arange(1, self.degree.item() + 1)
        x_poly = torch.pow(x.unsqueeze(-1), degrees)
        res = torch.sum(x_poly * self.weights.unsqueeze(-3), dim=[-1, -2])
        if self.bias is not None:
            res = res + self.bias
        return res
