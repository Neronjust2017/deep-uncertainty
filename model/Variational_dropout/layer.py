import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import math

def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))

class VdLinear(nn.Module):
    """
    Linear Layer variational dropout

    """
    def __init__(self, n_in, n_out, alpha_shape=(1, 1), bias=True):
        super(VdLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.alpha_shape = alpha_shape
        self.bias = bias

        # Learnable parameters -> Initialisation is set empirically.
        self.W = nn.Parameter(torch.Tensor(self.n_out, self.n_in))
        self.log_alpha = nn.Parameter(torch.Tensor(*self.alpha_shape))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.n_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.kl_value = calculate_kl

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, X, sample=False):

        mean = F.linear(X, self.W)
        if self.bias is not None:
            mean = mean + self.bias

        sigma = torch.exp(self.log_alpha) * self.W * self.W

        std = torch.sqrt(1e-16 + F.linear(X * X, sigma))

        if self.training or sample:
            epsilon = std.data.new(std.size()).normal_()
        else:
            epsilon = 0.0

        # Local reparameterization trick
        out = mean + std * epsilon

        kl = self.kl_loss()

        return out, kl

    def kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()

class VdConv1D(nn.Module):
    """
    Conv1D Layer variational dropout

    """
    def __init__(self, in_channels, out_channels, kernel_size, alpha_shape=(1,1), bias=True, stride=1,
                 padding=0, dilation=1):
        super(VdConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # 2D: (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha_shape = alpha_shape
        self.groups = 1

        # Learnable parameters -> Initialisation is set empirically.
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size))
        self.log_alpha = nn.Parameter(torch.Tensor(*alpha_shape))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.out_bias = lambda input, kernel: F.conv1d(input, kernel, self.bias, self.stride, self.padding,
                                                       self.dilation, self.groups)
        self.out_nobias = lambda input, kernel: F.conv1d(input, kernel, None, self.stride, self.padding, self.dilation,
                                                         self.groups)
        self.reset_parameters()
        self.kl_value = calculate_kl

    def reset_parameters(self):
        n = self.in_channels
        # for k in self.kernel_size:
        #     n *= k
        n *= self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x, sample=False):

        mean = self.out_bias(x, self.weight)

        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))

        if self.training or sample:
            epsilon = std.data.new(std.size()).normal_()
        else:
            epsilon = 0.0

        # Local reparameterization trick
        out = mean + std * epsilon

        kl = self.kl_loss()

        return out, kl

    def kl_loss(self):
        return self.weight.nelement() / self.log_alpha.nelement() * self.kl_value(self.log_alpha)