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
    def cal_func(self, x):

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
        # tau_mat = tau_m_mat + c1 + c2 + g
        #
        # #tau_mat = tau_m_mat
        return tau_m_mat, c1+c2, g
    def forward(self, x):
        m, c, g = self.cal_func(x)
        return m+c+g
    def forward_m(self,x):
        m, _, _ = self.cal_func(x)
        return m
    def forward_c(self,x):
        _, c, _ = self.cal_func(x)
        return c
    def forward_g(self,x):
        _, _, g = self.cal_func(x)
        return g

    def forward_all(self,x):
        m, c, g = self.cal_func(x)
        return m, c, g



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


class Delan_Sin(torch.nn.Module):
    def __init__(self, DOF, device='cpu'):
        super(Delan_Sin, self).__init__()
        L_Dim = 0
        for i in range(DOF):
            L_Dim += i + 1
        self.L_Dim = L_Dim
        self._LdNet = SinNet(DOF, 30, DOF).to(device)
        self._LoNet = SinNet(DOF, 30, L_Dim-DOF).to(device)
        self._gNet = SinNet(DOF, 30, DOF).to(device)
        self._dof = DOF
        self.device = device
        # input l and qdd
        self._mNet = SigmoidNet(L_Dim+DOF, 30, DOF).to(device)
        # input derivative of l and qd
        self._cNet = SigmoidNet(L_Dim*DOF+DOF, 30, DOF).to(device)
        self._gNet = SinNet(DOF, 30, DOF).to(device)

    def cal_func(self, x):
        q = x[:, 0:self._dof]
        q.requires_grad_(True)
        qDot = x[:, self._dof:self._dof * 2]
        qDDot = x[:, self._dof * 2:self._dof * 3]

        h_ld = self._LdNet(q)
        h_lo = self._LoNet(q)
        h_l = torch.cat((h_ld, h_lo), 1)

        # inertia force
        mInput = torch.cat((h_l, qDDot), 1)
        m = self._mNet(mInput)

        # coriolis and centrifigal force
        dl = torch.zeros(q.shape[0], h_l.shape[1]*self._dof).to(self.device)
        for i in range(h_l.shape[1]):
            dl[:, i*self._dof:(i+1)*self._dof] = torch.autograd.grad(outputs=h_l[:, i], inputs=q, grad_outputs=torch.ones(h_l[:, i].shape).to(self.device),
                                retain_graph=True, create_graph=True)[0]
        cInput = torch.cat((dl, qDot), 1)
        c = self._cNet(cInput)

        # gravity force
        g = self._gNet(q)

        return m, c, g

    def forward(self, x):
        m, c, g = self.cal_func(x)
        return m + c + g
    def forward_m(self,x):
        m, _, _ = self.cal_func(x)
        return m
    def forward_c(self,x):
        _, c, _ = self.cal_func(x)
        return c
    def forward_g(self,x):
        _, _, g = self.cal_func(x)
        return g

    def forward_all(self,x):
        m, c, g = self.cal_func(x)
        return m, c, g

class DeLanJacobianNet_inverse(torch.nn.Module):
    def __init__(self, DOF,  device='cpu'):
        super(DeLanJacobianNet_inverse, self).__init__()
        self.cartes_dim = 3
        L_Dim = 0
        for i in range(DOF):
            L_Dim += i + 1
        self.L_Dim = L_Dim
        self._gNet = SinNet(DOF, 40, DOF)
        self._dof = DOF
        self.device = device
        self.m = torch.nn.Parameter(torch.randn(DOF))
        self.episilon = 1e-6

        JoList = []
        JpList = []
        paramList = []
        inertia_dim = 0
        for i in range(self.cartes_dim):
            inertia_dim += i+1

        for i in range(DOF):
            JpList.append(SinNet(i+1, 40, self.cartes_dim * (i+1)))
            JoList.append(SinNet(i+1, 40, self.cartes_dim * (i+1)))
            paramList.append(torch.nn.Parameter(torch.randn(inertia_dim)))
        self.JpNetList = torch.nn.ModuleList(JpList)
        self.JoNetList = torch.nn.ModuleList(JoList)
        self.InertiaParam = torch.nn.ParameterList(paramList)

        self.isSpecialCase = True
    def cal_func(self, x):
        q = x[:,0:self._dof]
        q.requires_grad_(True)
        qDot = x[:, self._dof:self._dof*2]
        qDDot = x[:, self._dof*2:self._dof*3]

        # Jacobian Net
        Hp_mat = torch.zeros(x.shape[0], self._dof, self._dof).to(self.device)
        for i in range(self._dof):
            Jp = self.JpNetList[i](q[:,:i+1]).view(q.shape[0], self.cartes_dim, -1)
            Jpi = torch.cat((Jp,  torch.zeros(x.shape[0], self.cartes_dim, self._dof-i-1).to(self.device)), 2)
            if self.isSpecialCase:
                Jpi[:,2,:] = torch.zeros(Jpi.shape[0], Jpi.shape[2]).to(self.device)
            Hp_mat = Hp_mat + Jpi.permute(0,2,1).bmm(Jpi)*(self.m[i].clamp(min=self.episilon))

        # Orientation Inertia Matrix
        # h_lo =  self._LoNet(q)
        # LO_mat = torch.zeros(x.shape[0], self._dof, self._dof).to(self.device)
        # for i  in range(x.shape[0]):
        #     LO_mat[i][np.tril_indices(self._dof, 0)] = h_lo[i,:]
        # Ho_mat = LO_mat.bmm(LO_mat.permute(0,2,1))
        Ho_mat = torch.zeros(x.shape[0], self._dof, self._dof).to(self.device)
        for i in range(self._dof):
            Il_mat = torch.zeros(x.shape[0], self.cartes_dim, self.cartes_dim).to(self.device)
            for j in range(x.shape[0]):
                Il_mat[j][np.tril_indices(self.cartes_dim, 0)] = self.InertiaParam[i]
            I_mat = Il_mat.bmm(Il_mat.permute(0,2,1))
            if not self.isSpecialCase:
                Jo = self.JoNetList[i](q[:,:i+1]).view(q.shape[0], self.cartes_dim, -1)
                Joi = torch.cat((Jo, torch.zeros(x.shape[0], self.cartes_dim, self._dof - i - 1).to(self.device)), 2)
                Ho_mat = Ho_mat + Joi.permute(0,2,1).bmm(I_mat).bmm(Joi)
            else:
                # Jo = torch.zeros(x.shape[0], self.cartes_dim, self._dof).to(self.device)
                # Jo_3 = torch.cat((torch.ones(x.shape[0], i+1), torch.zeros(x.shape[0], self._dof-i-1)), 1).to(self.device)
                # Jo[:,2,:]  = Jo_3
                Ho_mat = Ho_mat + I_mat[:,:2,:2]

        H = Ho_mat + Hp_mat

        # for i in range(x.shape[0]):
        #     L_mat = torch.zeros(self._dof, self._dof)
        #     L_mat[np.tril_indices(self._dof, -1)] = h_lo[i][:]
        #     L_mat[range(self._dof),range(self._dof)] = h_ld[i][:]
        #     H = torch.mm(L_mat.t(),L_mat)
        #     tau_m = qDDot[i][:].matmul(H)
        #     tau_m_mat[i][:] = tau_m

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
        # tau_mat = tau_m_mat + c1 + c2 + g
        #
        # #tau_mat = tau_m_mat
        return tau_m_mat, c1+c2, g
    def forward(self, x):
        m, c, g = self.cal_func(x)
        return m+c+g
    def forward_m(self,x):
        m, _, _ = self.cal_func(x)
        return m
    def forward_c(self,x):
        _, c, _ = self.cal_func(x)
        return c
    def forward_g(self,x):
        _, _, g = self.cal_func(x)
        return g

    def forward_all(self,x):
        m, c, g = self.cal_func(x)
        return m, c, g

    def forward_Jp_mat(self, q):
        Jp = self.JpNetList[-1](q).view(q.shape[0], self.cartes_dim, -1)
        return Jp

    def forward_cartesVel(self, q, qdot):
        Jp_mat = self.forward_Jp_mat(q)
        x_vel = qdot.unsqueeze(1).bmm(Jp_mat.permute(0,2,1))
        x_vel = x_vel.squeeze(1)
        return x_vel