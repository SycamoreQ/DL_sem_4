import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats

class MutualInformationGraph:
    def __init__(self, method='ksg', bins=10):
        """
        Construct a graph based on mutual information between patches
        
        Args:
            method (str): Mutual information estimation method
            bins (int): Number of bins for discretization
        """
        self.method = method
        self.bins = bins
    
    def _discretize(self, X):
        """
        Discretize continuous data into bins
        
        Args:
            X (np.ndarray): Input data
        
        Returns:
            np.ndarray: Discretized data
        """
        return np.digitize(X, np.linspace(X.min(), X.max(), self.bins))
    
    def _estimate_mi_ksg(self, x, y):
        """
        Estimate mutual information using Kraskov-Stögbauer-Grassberger (KSG) estimator
        
        Args:
            x (np.ndarray): First variable
            y (np.ndarray): Second variable
        
        Returns:
            float: Mutual information
        """
        x = self._discretize(x)
        y = self._discretize(y)
        
        # Compute joint and marginal probabilities
        p_xy = np.zeros((self.bins, self.bins))
        for i in range(len(x)):
            p_xy[x[i], y[i]] += 1
        p_xy /= len(x)
        
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        # Compute mutual information
        mi = 0
        for i in range(self.bins):
            for j in range(self.bins):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    def construct_graph(self, patches):
        """
        Construct graph based on mutual information between patches
        
        Args:
            patches (np.ndarray): Input patches of shape (num_patches, patch_dim)
        
        Returns:
            np.ndarray: Adjacency matrix representing patch relationships
        """
        num_patches, patch_dim = patches.shape
        adj_matrix = np.zeros((num_patches, num_patches))
        
        # Compute mutual information between all patch pairs
        for i in range(num_patches):
            for j in range(num_patches):
                # Compute mutual information for each feature dimension
                mi_values = [
                    self._estimate_mi_ksg(patches[i, :, k], patches[j, :, k]) 
                    for k in range(patch_dim)
                ]
                
                # Average mutual information across dimensions
                adj_matrix[i, j] = np.mean(mi_values)
        
        # Normalize adjacency matrix
        row_sums = adj_matrix.sum(axis=1)
        adj_matrix /= row_sums[:, np.newaxis]
        
        return adj_matrix
