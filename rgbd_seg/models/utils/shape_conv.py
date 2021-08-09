# coding=utf-8

import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
from torch._six import container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class ShapeConv2d(Module):
    """
       ShapeConv2d can be used as an alternative for torch.nn.Conv2d.
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', testing=False):
        super(ShapeConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.testing = testing

        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, M, N))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if M * N > 1:
            self.Shape = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            self.Base = Parameter(torch.Tensor(1))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)

            init_one = np.ones([1], dtype=np.float32)
            self.Shape.data = torch.from_numpy(init_zero)
            self.Base.data = torch.from_numpy(init_one)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([1, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(ShapeConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def compute_shape_w(self, DW_shape):
        # (input_channels, D_mul, M * N)
        Shape = self.Shape + self.D_diag  # (1, M * N, self.D_mul)
        Base = self.Base
        W = torch.reshape(self.weight, (self.out_channels // self.groups, self.in_channels, self.D_mul))
        W_base = torch.mean(W, [2], keepdims=True)  # (self.out_channels // self.groups, self.in_channels)
        W_shape = W - W_base  # (self.out_channels // self.groups, self.in_channels, self.D_mul)

        # einsum outputs (out_channels // groups, in_channels, M * N),
        # which is reshaped to
        # (out_channels, in_channels // groups, M, N)
        D_shape = torch.reshape(torch.einsum('ims,ois->oim', Shape, W_shape), DW_shape)
        D_base = torch.reshape(W_base * Base, (self.out_channels, self.in_channels // self.groups, 1, 1))
        DW = D_shape + D_base
        return DW

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1 and not self.testing:
            DW = self.compute_shape_w(DW_shape)
        else:
            # in this case D_mul == M * N
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N)
            DW = torch.reshape(self.weight, DW_shape)
        return self._conv_forward(input, DW)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        print('load_state_dict')
        self._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                   missing_keys, unexpected_keys, error_msgs)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)
