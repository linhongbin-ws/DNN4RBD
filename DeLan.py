import torch
from Net import *
import numpy as np
from Net import ReLuNet
import time

class DeLanNet_inverse(torch.nn.Module):
    def __init__(self, Ld_Net, LoNet, gNet, DOF,  device='cpu'):
        super(DeLanNet_inverse, self).__init__()
        self._LdNet = Ld_Net
        self._LoNet = LoNet
        self._gNet = gNet
        self._dof = DOF
        self.device = device
    def forward(self, x):

        q = x[:,0:self._dof]
        q.requires_grad_(True)
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
            L_mat[i][np.tril_indices(self._dof, -1)] = h_lo[i,:]
            L_mat[i][range(self._dof),range(self._dof)] = h_ld[i,:]
        L_mat_P = L_mat.permute(0,2,1)
        H = L_mat.bmm(L_mat_P)
        tau_m_mat = qDDot.unsqueeze(1).bmm(H)
        tau_m_mat = tau_m_mat.squeeze(1)



        # Calculate Coriolis and Centrifugal term
        # d(qdT H qd)/dq
        C1 = qDot.unsqueeze(1).bmm(H).bmm(qDot.unsqueeze(2))
        c1 = torch.autograd.grad(outputs=C1, inputs=q, grad_outputs=torch.ones(C1.shape).to(self.device), retain_graph=True, create_graph=True)[0]

        # dH/dt q
        dH = torch.zeros(H.shape[0],H.shape[1],H.shape[2]).to(self.device)
        for i in range(H.shape[1]):
            for j in range(H.shape[2]):
                tmp = torch.autograd.grad(outputs=H[:,i,j], inputs=q, grad_outputs=torch.ones(H[:,i,j].shape).to(self.device), retain_graph=True,create_graph=True)[0]
                tmp = tmp.unsqueeze(1).bmm(qDot.unsqueeze(2))
                dH[:, i, j] = tmp.squeeze(2).squeeze(1)
        c2 = qDot.unsqueeze(1).bmm(dH)
        c2 = c2.squeeze(1)
        g = self._gNet(q)
        tau_mat = tau_m_mat + c1 + c2 + g

        #tau_mat = tau_m_mat
        return tau_mat


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