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
        h_ld = self._LdNet(q)
        h_lo = self._LoNet(q)
        L_mat = torch.zeros(self._dof, self._dof)
        for i in range(h_ld.shape[1]):
            L_mat[i][i] = h_ld[0][i]

        for i in range(h_ld.shape[1]):
            L_mat[i][i] = h_ld[i]
        cnt = 0
        for i in range(self._dof):
            for j in range(i):
                L_mat[i][j] = h_lo[0][cnt]
                cnt +=1

        return x
