import torch
from torch import nn
from src.utils.corruptions import create_corruption_nmatrix  

class Dense_Corrupted(nn.Module):
    """ Dense network with corruption matrices. """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        output_size: int,
        corruption,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()

        corruption_type = corruption['corruption_type']
        alpha = corruption['alpha']
        block_size = corruption['block_size']
        C1 = create_corruption_nmatrix(hidden_size1, corruption_type, alpha, block_size)
        C2 = create_corruption_nmatrix(hidden_size2, corruption_type, alpha, block_size)
        C3 = create_corruption_nmatrix(hidden_size3, corruption_type, alpha, block_size)

        # Register buffers to ensure they are part of the model state
        self.register_buffer('C1', C1)
        self.register_buffer('C2', C2)
        self.register_buffer('C3', C3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Expects flat input """
        x = self.relu(self.fc1(x))
        x = x @ self.C1.T  
        x = self.relu(self.fc2(x))
        x = x @ self.C2.T  
        x = self.relu(self.fc3(x))
        x = x @ self.C3.T  #
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    _ = SimpleDenseNet()
