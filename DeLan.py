import torch
from Net import *

class DeLan_inverseDynamic(torch.nn.Module):
    def __init__(self, LdNet, LoNet, VNet, DOF=2,  device='cpu'):
        self._LdNet = LdNet
        self._LoNet = LoNet
        self._VNet = VNet
        self._dof = DOF
    def forward(self, x):
        q = x[:,0:self._dof]
        qDot = x[:, self._dof:self._dof*2]
        qDDot = x[:, 0:self._dof*2:self._dof*3]

        return x
