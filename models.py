import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm

class FullyConnectedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoded_dim, width =10, num_layers=10):
        super(FullyConnectedAutoencoder, self).__init__()

        self.encoder = nn.ModuleList()
        # Encoder layers
        for i in range(encoded_dim):
            encoder_layers = []
            layer_input_dim = input_dim
            layer_output_dim = width
            for _ in range(num_layers):
                if _ == num_layers - 1:
                    layer_output_dim = 1
                encoder_layers.append(spectral_norm(nn.Linear(layer_input_dim, layer_output_dim)))
                if _ < num_layers - 1:
                    encoder_layers.append(nn.GELU())
                layer_input_dim = layer_output_dim
            self.encoder.append(nn.Sequential(*encoder_layers))
        
        # Decoder layers
        decoder_layers = []
        layer_input_dim = encoded_dim
        layer_output_dim = width
        for _ in range(num_layers):
            if _ == num_layers - 1:
                layer_output_dim = input_dim
            decoder_layers.append(nn.Linear(layer_input_dim, layer_output_dim))
            if _ < num_layers - 1:
                decoder_layers.append(nn.GELU())
            layer_input_dim = layer_output_dim
        self.decoder = nn.Sequential(*decoder_layers)
                

    def forward(self, x):
        encoded = torch.hstack([encoder(x) for encoder in self.encoder])
        #encoded = self.encoder(x) 
        decoded = self.decoder(encoded)
        return encoded, decoded


class ElementwiseMultiplicationLayer(nn.Module):
    def __init__(self, dim):
        super(ElementwiseMultiplicationLayer, self).__init__()
        # Initialize learnable weights
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Element-wise multiplication with learnable weights
        return x * self.weights
    

class UnitNormalizationLayer(nn.Module):
    def __init__(self):
        super(UnitNormalizationLayer, self).__init__()
       
    def forward(self, x):
        # Normalize the input tensor
        norm = torch.norm(x, dim=1, keepdim=True)
        return x / (norm + 1e-8) 



class PositiveDefiniteMatrix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Raw parameters for lower-triangular part
        self.lower_entries = nn.Parameter(torch.randn(dim, dim))
    
    def forward(self):
        # Make lower-triangular matrix
        L = torch.tril(self.lower_entries)
        # Force positive diagonals by applying softplus
        diag_indices = torch.arange(L.size(0))
        L[diag_indices, diag_indices] = torch.nn.functional.softplus(L[diag_indices, diag_indices])
        # Create PSD matrix
        M = L @ L.t()
        return M
