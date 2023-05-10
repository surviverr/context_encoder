import torch
import torch.nn as nn
import torch.optim as optim

class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        A = torch.randn((3, 6, 6), requires_grad = True)
        B = torch.randn((3, 2, 3), requires_grad = True)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.f = nn.Dropout(p = 0.5)
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1, 1)
        x = torch.matmul(self.A, x)
        x = x.view(x.shape[0], x.shape[1], 2, 3)
        x = x + self.B
        x = self.f(x)
        return x

a = torch.randn((1, 3, 2, 3))
model = Disc()
b = model(a)
loss = b.sum()

opt = optim.Adam(model.parameters(), lr = 1)
opt.zero_grad()
loss.backward()
opt.step()