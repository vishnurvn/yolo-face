import torch
import torch.nn as nn


class YOLOFaceNet(nn.Module):
    def __init__(self):
        super(YOLOFaceNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, (7, 7))
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.conv_2 = nn.Conv2d(64, 96, (3, 3))
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.conv_3 = nn.Conv2d(96, 128, (1, 1))
        self.conv_4 = nn.Conv2d(128, 256, (3, 3))
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.conv_5 = nn.Conv2d(256, 256, (1, 1))
        self.conv_6 = nn.Conv2d(256, 512, (3, 3))
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.linear_1 = nn.Linear(512 * 25 * 25, 4096)
        self.linear_2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_3(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.pool_4(x)
        x = self.linear_1(x.flatten())
        x = self.linear_2(x)
        return x


net = YOLOFaceNet()
inp = torch.rand((1, 3, 448, 448))
with torch.no_grad():
    op = net(inp)

print(op.flatten().size())