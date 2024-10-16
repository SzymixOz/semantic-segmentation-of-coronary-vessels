import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super(CustomNet, self).__init__()
        
        def CBR(in_channels, out_channels, dropout_rate):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            )

        self.enc1 = CBR(3, 64, dropout_rate)
        self.enc2 = CBR(64, 128, dropout_rate)
        self.enc3 = CBR(128, 256, dropout_rate)
        self.enc4 = CBR(256, 512, dropout_rate)
        
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024, dropout_rate)

        self.linear4 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear1 = nn.Linear(16, num_classes)
        

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))

        linear4 = self.linear4(bottleneck.view(bottleneck.size(0), -1))
        linear3 = self.linear3(linear4)
        linear2 = self.linear2(linear3)
        linear1 = self.linear1(linear2)      

        return linear1

    def predict(self, x):
        return F.softmax(self.forward(x), dim=1)
