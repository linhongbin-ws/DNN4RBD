import torch


x = torch.tensor([2., 1.], requires_grad=True).view(1, 2)
y = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

z = torch.mm(x, y)
print(f"z:{z}")
z.backward(torch.Tensor([[1., 0]]), retain_graph=True)
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")