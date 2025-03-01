# common_autoencoder_blocks.py
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Simple convolutional encoder for MNIST that dynamically computes shape,
    then flattens and passes through optional hidden dims -> final latent_dim.
    """
    def __init__(
        self,
        input_shape=(1, 28, 28),
        channels=[32, 64],
        kernel_sizes=[3, 3],
        strides=[2, 2],
        paddings=[1, 1],
        latent_dim=128,
        hidden_dims=None,
        dropout_conv=0.0,
        batch_norm_conv=False
    ):
        super(Encoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = []

        # Build conv layers
        layers = []
        in_channels = input_shape[0]
        for out_channels, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p))
            if batch_norm_conv:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout_conv > 0.0:
                layers.append(nn.Dropout2d(p=dropout_conv))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Compute flattened size after conv
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
        self.conv_output_shape = out.shape[1:]   
        flattened_size = out.numel()  # number of elements

        # Build FC layers
        fc_layers = []
        in_dim = flattened_size
        for hd in hidden_dims:
            fc_layers.append(nn.Linear(in_dim, hd))
            fc_layers.append(nn.ReLU(inplace=True))
            in_dim = hd

        # final linear to latent
        fc_layers.append(nn.Linear(in_dim, latent_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    """
    A generic deconvolutional decoder that mirrors the Encoder’s conv stack.
    It first projects the latent vector to a feature map, then applies a series
    of ConvTranspose2d layers (with ReLU in-between) to reconstruct the image.
    """
    def __init__(
        self, 
        conv_output_shape, 
        channels, 
        kernel_sizes, 
        strides, 
        paddings,
        latent_dim=128, 
        dropout_conv=0.0,
        output_channels=1, # If we are using RGB, this should be 3
        batch_norm_conv=False
    ):
        """
        Args:
            conv_output_shape (tuple): Shape (C_out, H_out, W_out) the encoder produced.
            channels (list[int]): Same channels list used in the encoder (will be reversed).
            kernel_sizes (list[int]): Same kernel sizes used in the encoder (will be reversed).
            strides (list[int]): Same strides used in the encoder (will be reversed).
            paddings (list[int]): Same paddings used in the encoder (will be reversed).
            latent_dim (int): Dimension of the latent vector.
            output_channels (int): Number of channels in the reconstructed output (1 for MNIST, 3 for RGB).
        """
        super(Decoder, self).__init__()
        self.conv_output_shape = conv_output_shape
        flattened_size = int(torch.prod(torch.tensor(conv_output_shape)).item())

        # FC layer to expand latent vector to the flattened conv feature map
        self.fc = nn.Linear(latent_dim, flattened_size)

        # Reverse the architecture to "undo" the encoder
        rev_channels = channels[::-1]
        rev_kernel_sizes = kernel_sizes[::-1]
        rev_strides = strides[::-1]
        rev_paddings = paddings[::-1]

        layers = []
        in_channels = conv_output_shape[0]
        num_layers = len(rev_kernel_sizes)

        for i in range(num_layers):
            # For all but last layer, output_channels come from rev_channels
            if i < num_layers - 1:
                out_ch = rev_channels[i + 1]
            else:
                out_ch = output_channels

            # We assume typical downsampling in the encoder used stride=2
            # so we do "output_padding = stride - 1" to invert that.
            # If you used stride=1, output_padding=0 is correct automatically.
            output_padding = max(rev_strides[i] - 1, 0)

            layers.append(nn.ConvTranspose2d(
                in_channels, out_ch,
                kernel_size=rev_kernel_sizes[i],
                stride=rev_strides[i],
                padding=rev_paddings[i],
                output_padding=output_padding
            ))

            if i < num_layers - 1:
                if batch_norm_conv:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout2d(p=dropout_conv))
            in_channels = out_ch

        self.deconv = nn.Sequential(*layers)
        self.out_activation = nn.Sigmoid()  # Restrict output to [0, 1] range

    def forward(self, latent):
        """
        Maps a latent vector (batch_size, latent_dim) back to an image.
        """
        batch_size = latent.size(0)
        x = self.fc(latent)                                 # (batch_size, flattened_size)
        x = x.view(batch_size, *self.conv_output_shape)      # (batch_size, C_out, H_out, W_out)
        x = self.deconv(x)                                   # (batch_size, output_channels, H, W)
        x = self.out_activation(x)
        return x
