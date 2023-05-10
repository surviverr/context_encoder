import torch
import torch.nn as nn

class Genenator(nn.Module):
    def __init__(self):
        super(Genenator, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            # 3 * 128 * 128
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 6, stride = 2, padding = 0, bias = False),
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
            nn.Conv2d(in_channels = 36, out_channels = 36, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = 36),
            nn.ReLU(inplace = True),
            # 36 * 14 * 14
            nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size = 4, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = 48),
            nn.ReLU(inplace = True),
            # 48 * 6 * 6
        )

        # channel-wise fc
        A = torch.randn((48, 36, 36))
        B = torch.randn((48, 6, 6))
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.f = nn.ReLU()
        self.dp = nn.Dropout(p = 0.5)
        self.pro_c = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 1)

        # decoder
        self.decoder = nn.Sequential(
            # 48 * 6 * 6
            nn.ConvTranspose2d(in_channels = 48, out_channels = 36, kernel_size = 4, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = 36),
            nn.ReLU(inplace = True),
            # 36 * 14 * 14
            nn.ConvTranspose2d(in_channels = 36, out_channels = 24, kernel_size = 4, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = 24),
            nn.ReLU(inplace = True),
            # 24 * 30 * 30
            nn.ConvTranspose2d(in_channels = 24, out_channels = 12, kernel_size = 4, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(num_features = 12),
            nn.ReLU(inplace = True),
            # 12 * 62 * 62
            nn.ConvTranspose2d(in_channels = 12, out_channels = 3, kernel_size = 9, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = 3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], x.shape[1], -1, 1)
        x = torch.matmul(self.A, x)
        x = x.view(x.shape[0], x.shape[1], 6, 6)
        x = self.f(x + self.B)
        x = self.dp(x)
        x = self.pro_c(x)
        return self.decoder(x)

def test():
    model = Genenator()
    x = torch.randn((2, 3, 129, 129))
    print(model(x).shape)

if __name__ == '__main__':
    test()