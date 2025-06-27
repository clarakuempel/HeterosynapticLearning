import torch 

def create_corruption_nmatrix(d, corruption_type='identity', alpha=0.1, block_size=4):
    """Create correlation matrix C
    Args:
        d (int): Dimensionality of the activation vector.
        corruption_type (str): One of ['identity', 'block_diagonal', 'full_dense'].
        alpha (float): Strength of off-diagonal correlation.
        block_size (int): Block size for block-diagonal structure.
    
    Returns:
        torch.Tensor: Correlation matrix C of shape (d, d).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if corruption_type == 'identity':
        return torch.eye(d, device=device)
    
    elif corruption_type == 'block_diagonal':
        C = torch.eye(d, device=device)
        
        # Handle case where d is not perfectly divisible by block_size
        for start in range(0, d, block_size):
            end = min(start + block_size, d)
            block_d = end - start
            
            # Create block with alpha off-diagonal, 1 on diagonal
            if block_d > 1:  # Only apply correlation if block has more than 1 element
                block = torch.full((block_d, block_d), alpha, device=device)
                block.fill_diagonal_(1.0)
                C[start:end, start:end] = block
            
        return C
    
    elif corruption_type == 'full_dense':
        C = torch.full((d, d), alpha, device=device)
        C.fill_diagonal_(1.0)
        return C

    else:
        # Default to identity for unknown types
        print(f"Warning: Unknown matrix corruption type '{corruption_type}', using identity")
        return torch.eye(d, device=device)

