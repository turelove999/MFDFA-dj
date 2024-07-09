import torch

A = torch.ones((1, 100), requires_grad=True)

B = None
with torch.no_grad():
    B = A * 2

_A = torch.mean(B)

_A.requires_grad = True

_A.backward()

print(_A.grad)
print(_A.requires_grad)
