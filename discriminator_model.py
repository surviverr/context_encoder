import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # 3 * 68 * 68
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 9, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = 12),
            nn.ReLU(inplace = True),
            # 12 * 62 * 62
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 4, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = 24),
            nn.ReLU(inplace = True),
            # 24 * 30 * 30
            nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size = 4, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = 36),
            nn.ReLU(inplace = True),
            # 36 * 14 * 14
            nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 4, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = 48),
            nn.ReLU(inplace = True),
            # 48 * 6 * 6
            nn.Flatten(start_dim = 1),
            nn.Linear(in_features = 1728, out_features = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).reshape(-1, 1)

def test():
    model = Discriminator()
    x = torch.randn((2, 3, 68, 68))
    print(model(x).shape)

if __name__ == '__main__':
    test()