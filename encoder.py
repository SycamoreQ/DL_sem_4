import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat

class LaplacianPositionalEncoding(nn.Module):
    """
    Laplacian eigenvector-based positional encoding for image patches
    """
    def __init__(self, patch_size, img_size, dim, normalized=True):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.dim = dim
        self.normalized = normalized
        
        # Calculate grid size based on image size and patch size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        # Pre-compute the Laplacian eigenvalues and eigenvectors
        self.compute_laplacian_eigenvectors()
        
        # Trainable projection layer
        self.projection = nn.Linear(self.num_patches, dim)
        
    def compute_laplacian_eigenvectors(self):
        """
        Compute the Laplacian eigenvalues and eigenvectors for a 2D grid graph
        representing the patch structure of the image
        """
        # Create adjacency matrix for a 2D grid graph
        adj_matrix = torch.zeros(self.num_patches, self.num_patches)
        
        # Connect adjacent patches (4-neighborhood)
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                node = i * self.grid_w + j
                
                # Up connection
                if i > 0:
                    adj_matrix[node, (i-1) * self.grid_w + j] = 1
                
                # Down connection
                if i < self.grid_h - 1:
                    adj_matrix[node, (i+1) * self.grid_w + j] = 1
                
                # Left connection
                if j > 0:
                    adj_matrix[node, i * self.grid_w + (j-1)] = 1
                
                # Right connection
                if j < self.grid_w - 1:
                    adj_matrix[node, i * self.grid_w + (j+1)] = 1
        
        # Compute degree matrix
        degree_matrix = torch.diag(adj_matrix.sum(dim=1))
        
        # Compute Laplacian matrix
        laplacian = degree_matrix - adj_matrix
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        
        # Store the eigenvectors and edge indices
        self.register_buffer('eigenvectors', eigenvectors)
        self.register_buffer('eigenvalues', eigenvalues)
        
        # Create and store edge_index for GCN
        edge_index = []
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                if adj_matrix[i, j] > 0:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index).t()
        self.register_buffer('edge_index', edge_index)
    
    def forward(self, batch_size):
        """
        Generate Laplacian positional embeddings for each position in the grid
        """
        # Use top-k eigenvectors as the positional encoding basis
        k = min(self.num_patches, self.dim)
        laplacian_basis = self.eigenvectors[:, :k]
        
        if self.normalized:
            # Weight eigenvectors by the inverse square root of eigenvalues
            # (adding small epsilon to avoid division by zero)
            epsilon = 1e-6
            weights = 1.0 / torch.sqrt(self.eigenvalues[:k] + epsilon)
            
            # Zero out the weight for the first eigenvector (constant)
            weights[0] = 0
            
            # Apply weights to the eigenvectors
            scaled_basis = laplacian_basis * weights.unsqueeze(0)
        else:
            scaled_basis = laplacian_basis
        
        # Project to the desired embedding dimension
        if k < self.dim:
            positional_embedding = self.projection(scaled_basis)
        else:
            positional_embedding = scaled_basis
        
        # Repeat for each item in the batch
        positional_embedding = repeat(positional_embedding, 'n d -> b n d', b=batch_size)
        
        return positional_embedding, self.edge_index

    

class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x
    

class ConditionalPositionEncoding(nn.Module):
    """
    Implementation of conditional positional encoding. For more details refer to paper: 
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x
