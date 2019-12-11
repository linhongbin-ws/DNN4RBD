import torch
from Net import *
import numpy as np
from Net import ReLuNet

class DeLanNet_inverse(torch.nn.Module):
    def __init__(self, Ld_Net, LoNet, DOF,  device='cpu'):
        super(DeLanNet_inverse, self).__init__()
        self._LdNet = Ld_Net
        self._LoNet = LoNet
        #self._VNet = VNet
        self._dof = DOF
        self.device = device
    def forward(self, x):
        q = x[:,0:self._dof]
        qDot = x[:, self._dof:self._dof*2]
        qDDot = x[:, self._dof*2:self._dof*3]
        h_ld = self._LdNet(q)
        h_lo = self._LoNet(q)
        # for i in range(x.shape[0]):
        #     L_mat = torch.zeros(self._dof, self._dof)
        #     L_mat[np.tril_indices(self._dof, -1)] = h_lo[i][:]
        #     L_mat[range(self._dof),range(self._dof)] = h_ld[i][:]
        #     H = torch.mm(L_mat.t(),L_mat)
        #     tau_m = qDDot[i][:].matmul(H)
        #     tau_m_mat[i][:] = tau_m
        L_mat = torch.zeros(x.shape[0], self._dof,self._dof).to(self.device)
        for i  in range(x.shape[0]):
            L_mat[i][np.tril_indices(self._dof, -1)] = h_lo[i][:]
            L_mat[i][range(self._dof),range(self._dof)] = h_ld[i][:]
        L_mat_P = L_mat.permute(0,2,1)
        H = L_mat.bmm(L_mat_P)
        tau_m_mat = qDDot.unsqueeze(1).bmm(H)
        tau_m_mat = tau_m_mat.squeeze(1)
        return tau_m_mat


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