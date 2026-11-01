import torch


x = torch.tensor(36, requires_grad=True, dtype=torch.float32)
y = 4 * (x ** 4)

y.retain_grad()

print(x.grad)
print(y.grad)

y.backward()

print('POST')
print(x.grad)
print(y.grad)