import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3),
                     stride=(stride, stride, stride), padding=(1,1,1), bias=False)

class ResidualBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channels, stride=1, dim_change=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels*self.expansion, stride=1)
        self.bn2 = nn.BatchNorm3d(out_channels*self.expansion)
        self.dim_change = dim_change

    def forward(self, x):
        residual = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.dim_change:
            residual = self.dim_change(residual)
        x += residual
        return self.relu(x)

class ResNet3D(nn.Module):
    def __init__(self, block, layers, image_channel, num_classes=1000):
        super().__init__()
        self.in_channel = 64
        self.num_class = num_classes
        # In 'Pytoch Documentation', paddinng of 3 is used in the very first layer
        self.conv1 = nn.Conv3d(image_channel, 64, kernel_size=(7,7,7), stride=(1,1,1), padding=(3,3,3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool= nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1,1,1))

        # Defining Resnet layers
        self.layer1 = self._makeLayers(block, layers[0], 64, stride=1)
        self.layer2 = self._makeLayers(block, layers[1], 128, stride=2)
        self.layer3 = self._makeLayers(block, layers[2], 256, stride=2)
        self.layer4 = self._makeLayers(block, layers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # After 'AvgPool', It becomes a 1x1 image
        self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=0.2),
                                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=0.2),
                                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p=0.2),
                                nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                nn.Linear(32, self.num_class)
                                )

    def _makeLayers(self, block, num_residual_blocks, out_channels, stride=1):
        dim_change = None
        layers = []

        # Change dimensions for residuals
        if (stride != 1) or (self.in_channel != out_channels * block.expansion):
            dim_change = nn.Sequential(nn.Conv3d(self.in_channel, out_channels * block.expansion, kernel_size=(1,1,1),
                                                 stride=(stride, stride, stride), bias=False),
                                       nn.BatchNorm3d(out_channels * block.expansion)
                                       )

        layers.append(block(self.in_channel, out_channels, stride, dim_change))
        self.in_channel = out_channels * block.expansion
        for i in range(1, num_residual_blocks):
            layers.append(block(self.in_channel, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Assuming input shape is: 3 x 224 x 224
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # 112x112
        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14
        x = self.layer4(x)  # 7x7
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # 1x1
        x = self.fc(x)
        return x

