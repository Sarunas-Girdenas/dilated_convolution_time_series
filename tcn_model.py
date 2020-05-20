import torch
from torch import nn
import numpy as np

class DilatedNet(nn.Module):
    def __init__(self, num_features=5, out_channels=64, 
                 dilation=2,
                 kernel_size=2,
                 depth=3,
                 seq_length=16):
        """
        Dilated Conv Network
        Inputs:
        =======
        num_features (int): number of input features
        out_channels (int): out channels in conv layer
        dilation (int): dilation for the network
        kernel_size (int): kernel size for conv1d
        depth (int): number of Conv1D layers
        seq_length (int): length of sequence
        """
        
        super(DilatedNet, self).__init__()

        self.relu = nn.ReLU()
        
        # First Layer
        self.dilated_conv1 = nn.Conv1d(
            num_features,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation)

        # calculate output size for the final feed forward layer
        temp_out_shape = (seq_length + 2 * 0 - kernel_size - (kernel_size-1) * (dilation-1))/1 + 1
        out_shape = int(temp_out_shape) - (depth * dilation)
        
        # 1D Convolution Blocks
        conv_blocks = []
        for _ in range(depth):
            conv_blocks.append(
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation
                    )
                )
            conv_blocks.append(self.relu)
        
        self.convolution_blocks = nn.Sequential(*conv_blocks)

        self.conv_final = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        # Output Layer
        self.feed_forward = torch.nn.Linear(out_shape, 1)
        
        return None

    @staticmethod
    def init_weights(layer):
        """
        Purpose: initialize weights in each
        LINEAR layer.
        Input: pytorch layer
        """

        if isinstance(layer, torch.nn.Linear):
            np.random.seed(42)
            size = layer.weight.size()
            fan_out = size[0] # number of rows
            fan_in = size[1] # number of columns
            variance = np.sqrt(2.0/(fan_in + fan_out))
            # initialize weights
            layer.weight.data.normal_(0.0, variance)

    def forward(self, x):
        """
        Forward pass. We stack 1D convolution blocks,
        then do maxpool to shape this as classification problem.
        """

        out = self.dilated_conv1(x)
        out = self.relu(out)

        out = self.convolution_blocks(out)
        out = self.conv_final(out)
        
        # max pool
        out = out.max(dim=1)[0]
        
        # linear
        out = self.feed_forward(out)

        return torch.sigmoid(out)