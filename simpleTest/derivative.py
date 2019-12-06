import torch
from Net import *
from DeLan import DerivativeNet


device =  torch.device('cpu')

N, D_in, D_out = 100, 1, 1

base_model = BPNet(D_in,100,D_out)
model = DerivativeNet(base_model)
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-1
weight_decay = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
for t in range(2000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

