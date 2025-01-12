

import torch.nn as nn
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=16, output_dim=128):
        super(SimpleCNN, self).__init__()

        # 定义卷积层部分
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3),  # 输入：1通道，输出：16通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32

            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),  # 输入：16通道，输出：32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),  # 输入：32通道，输出：64通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # 展平操作，将卷积层输出展平成向量
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(64 * 8 * 8, 16*output_dim)  # 64通道，8x8的图像块

    def forward(self, x):
        # 通过卷积层部分
        x = self.features(x)
        # 展平输出
        x = self.flatten(x)
        # print(x.shape)
        # 通过全连接层
        x = self.fc(x)

        return x






