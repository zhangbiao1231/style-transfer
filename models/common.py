
import torch.nn as nn

class VggBlock(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        layers = []
        for _ in range(n):
            layers.append(nn.Conv2d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1,padding=1))
            layers.append(nn.ReLU())
            c1 = c2
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.m = nn.Sequential(*layers)
    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.m(x)

class Linear(nn.Module):
    def __init__(self,c1,c2,drop=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        self.linear = nn.Linear(c1,c2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
    def forward(self,x):
        return self.drop(self.relu(self.linear(x)))