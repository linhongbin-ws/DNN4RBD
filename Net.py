import torch

class PolNet(torch.nn.Module):
    def __init__(self, D_in, pol_dim,  device='cpu'):
        super(PolNet, self).__init__()
        self.D_in = D_in
        self.pol_dim = pol_dim
        self.device = device
        self.linears = torch.nn.ModuleList([torch.nn.Linear(pol_dim+1, 1) for i in range(D_in)])
    def forward(self, x):
        y = torch.ones([x.shape[0], self.D_in], dtype=torch.float32, device=self.device)
        for j in range(self.D_in):
            feature = torch.ones([x.shape[0], self.pol_dim+1], dtype=torch.float32, device=self.device)
            for i in range(self.pol_dim):
                feature[:,i+1] = x[:,j].pow(i+1).squeeze()
            linear = self.linears[j]
            y_pred = linear(feature)
            y[:,j] = y_pred.squeeze()
        return y

class VanillaNet(torch.nn.Module):
    def __init__(self, base_net, addition_net):
        super(VanillaNet, self).__init__()
        self.base_net = base_net
        self.addition_net = addition_net
    def forward(self, x):
        y1 = self.base_net(x)
        y = self.addition_net(x) + y1
        return y

class LagrangeNet(torch.nn.Module):
    def __init__(self, base_model, delta_q, w_vec, device='cpu'):
        super(LagrangeNet, self).__init__()
        self.base_model = base_model
        self.delta_q = delta_q
        self.w_vec = w_vec
        self.device = device
    def forward(self, feature):
        target_hat = torch.zeros(feature.shape, dtype=torch.float32, device=self.device)
        base_hat = self.base_model(feature)
        base_hat.require_grad = False
        for j in range(feature.shape[1]):
            feature_d = feature.clone()
            feature_d[:, j] = feature_d[:, j].clone() + torch.ones(feature_d[:, j].shape,
                                                                   device=self.device).float() * self.delta_q
            target_hat[:, j] = self.base_model(feature_d).squeeze() - base_hat.squeeze()
        target_hat = target_hat / self.delta_q
        target_hat = torch.mul(target_hat, self.w_vec)
        return target_hat

# def Lagrange_Net(model, feature, delta_q, w_vec, device='cpu'):
#     target_hat = torch.zeros(feature.shape, dtype=torch.float32, device=device)
#     base_hat = model(feature)
#     base_hat.require_grad = False
#     for j in range(feature.shape[1]):
#         feature_d = feature.clone()
#         feature_d[:, j] = feature_d[:, j].clone() + torch.ones(feature_d[:, j].shape, device=device).float() * delta_q
#         target_hat[:, j] = model(feature_d).squeeze()-base_hat.squeeze()
#     target_hat = target_hat / delta_q
#     target_hat = torch.mul(target_hat, w_vec)
#     return target_hat


# create Net architecture
class LogNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LogNet, self).__init__()
        H1 = 100
        H2 = 50
        self._relu = torch.nn.ReLU()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._middle_linear = torch.nn.Linear(H, H)
        self._output_linear = torch.nn.Linear(H, D_out)
        self._epsilon = 1

    def forward(self, x):
        h = x
        h = self._input_linear(h).clamp(self._epsilon)
        h = torch.log(h)
        h = self._middle_linear(h)
        h = torch.exp(h)
        y_pred = self._output_linear(h)
        return y_pred

class SinLogNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SinLogNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)
        self._epsilon = 0.003
        self.bias = torch.nn.Parameter(torch.randn(1, D_in))

    def forward(self, x):
        h = x + self.bias
        h1 = torch.sin(h)
        h = h.clamp(self._epsilon)
        h = torch.log(h)
        h = self._input_linear(h)
        h = torch.exp(h)
        y_pred = self._output_linear(h)
        return y_pred

# create Net architecture
class BPNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(BPNet, self).__init__()
        self._tanh = torch.nn.Tanh()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = self._tanh(h)
        y_pred = self._output_linear(h)
        return y_pred

class ReLuNet(torch.nn.Module):
    def __init__(self, D_in, H_list, D_out):
        super(ReLuNet, self).__init__()
        self.relu = torch.nn.ReLU()
        H_list = [D_in] + H_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(H_list[i], H_list[i+1]) for i in range(len(H_list)-1)])
        self.output_layer = torch.nn.Linear(H_list[-1], D_out)
    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
        x = self.output_layer(x)
        return x

class SinNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SinNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._middle_linear = torch.nn.Linear(H, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = torch.sin(h)
        y_pred = self._output_linear(h)
        return y_pred

class Multi_SinNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Multi_SinNet, self).__init__()
        self.sin_layers = torch.nn.ModuleList([SinNet(i+1, H, D_out) for i in range(D_in)])
        self.D_in = D_in
        self.D_out = D_out


    def forward(self, x):
        y_pred = torch.zeros([x.shape[0], self.D_out], dtype=torch.float32, device='cpu')
        for i in range(self.D_in):
            layer = self.sin_layers[i]
            y_pred = y_pred + layer(x[:,:i+1])
        return y_pred


class SigmoidNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SigmoidNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = torch.sigmoid(h)
        y_pred = self._output_linear(h)
        return y_pred