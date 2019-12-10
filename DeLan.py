import torch
from Net import *
import numpy as np

class DeLan_inverseDynamic(torch.nn.Module):
    def __init__(self, LdNet, LoNet, VNet, DOF,  device='cpu'):
        self._LdNet = LdNet
        self._LoNet = LoNet
        self._VNet = VNet
        self._dof = DOF
    def forward(self, x):
        q = x[:,0:self._dof]
        qDot = x[:, self._dof:self._dof*2]
        qDDot = x[:, 0:self._dof*2:self._dof*3]
        h_ld = self._LdNet(q)
        h_lo = self._LoNet(q)
        L_mat = torch.zeros(self._dof, self._dof)
        L_mat[np.tril_indices(self._dof, 1)] = h_lo
        L_mat[range(self._dof),range(self._dof)] = h_ld
        H = torch.mm(L_mat.t(),L_mat)
        tau_m = qDDot.matmul(H)
        return tau_m


class DerivativeNet(torch.nn.Module):
    def __init__(self, baseNet, device='cpu'):
        super(DerivativeNet, self).__init__()
        self._baseNet = baseNet
    def forward(self, x):
        x.requires_grad_(True)
        h = self._baseNet(x)
        a = torch.zeros(h.shape,requires_grad=True)
        for i in range(x.shape[0]):
            dx1 = torch.autograd.grad(outputs=h[i][0], inputs=x[i][0],retain_graph=True)
            a = dx1
        return a