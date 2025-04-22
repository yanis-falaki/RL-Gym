import torch
import numpy as np

class NumberEncoder():
    def __init__(self, dimensionality, device='cpu'):
        self.dimensionality = dimensionality
        self.odd_indices_tensor_size = (dimensionality // 2)*2
        self.device = device

    def encode(self, number):
        if isinstance(number, np.ndarray):
            number = torch.from_numpy(number)

        pe = torch.empty(self.dimensionality, device=self.device)

        # Create div_term for even and odd positions separately with correct shapes
        div_term_even = torch.exp(torch.arange(0, self.dimensionality, 2, device=self.device) * 
                        (-torch.log(torch.tensor(10000.0, device=self.device)) / self.dimensionality))
                        
        # For odd indices (if dimensionality is odd, this will have one fewer element)
        div_term_odd = torch.exp(torch.arange(0, self.odd_indices_tensor_size, 2, device=self.device) * 
                       (-torch.log(torch.tensor(10000.0, device=self.device)) / self.dimensionality))
        
        # Apply sin to even indices
        pe[0::2] = torch.sin(number * div_term_even)
        
        # Apply cos to odd indices
        pe[1::2] = torch.cos(number * div_term_odd)
        
        return pe