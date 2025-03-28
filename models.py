import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import ResNet50_Weights , resnet18
import numpy as np 
import seaborn as sns
from torch_geometric.nn import SAGEConv , GATConv , GCNConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from encoder import PatchEmbed , LaplacianPositionalEncoding , ConditionalPositionEncoding
from Grapher import Grapher


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Model_1(nn.Module):
    def __init__(self , num_classes):
        super(Model_1 , self).__init__()
        self.num_classes = num_classes
        self.grouped_conv = nn.Conv2d(in_channels = 16 , out_channels = 32 , kernel_size = 3 , stride = 1 , padding = 1)
        self.transformer_block = nn.TransformerEncoderLayer(d_model = 32 , nhead = 4)
        self.gcn_block = GCNConv(in_channels = 32 , out_channels = 32)
        self.fc = nn.FeedForward(32 , 16)
        
    def forward(self , x , edge_index):
        x = self.grouped_conv(x)
        x = self.transformer_block(x)
        x = self.gcn_block(x , edge_index)
        x = F.relu(x)
        x = F.dropout(x , 0.5)
        x = self.gcn_block(x , edge_index)
        x = F.log_softmax(x , dim = 1)
        return self.fc(x)
    

class Model_2(nn.Module):
    def __init__(self , num_classes):
        super(Model_2 , self).__init__()
        self.num_classes = num_classes
        self.laplacian = AddLaplacianEigenvectorPE()
        self.transformer_block = nn.TransformerEncoderLayer(d_model = 32 , nhead = 4) 
        self.patch_embed = PatchEmbed()
        self.fc = nn.FeedForward(32 , 16)
        
    def forward(self , x):
        x = self.patch_embed(x)
        x = self.laplacian(x)
        x = self.transformer_block(x)
        return self.fc(x)


class Combined(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=2):
        super(Combined, self).__init__()
        
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.grouped_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        feature_size = img_size // 4  
        # Laplacian Positional Encoding
        self.laplacian_pe = LaplacianPositionalEncoding(
            patch_size=patch_size,
            img_size=img_size,
            dim=32
        )
        
        self.to_sequence = nn.Sequential(
            nn.Flatten(2, 3),  
            nn.Linear(feature_size * feature_size, 32), 
        )
        self.transformer_block = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.gcn_block = GCNConv(in_channels=32, out_channels=32)
        self.final_gcn = GCNConv(in_channels=32, out_channels=num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.init_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.grouped_conv(x)
        
        b, c, h, w = x.shape
        
        pos_emb, edge_index = self.laplacian_pe(batch_size)
        
        x = x.permute(0, 2, 3, 1)  
        x = x.reshape(b, h*w, c)   
        
        if pos_emb.shape[1] == x.shape[1]:
            x = x + pos_emb
        else:
            if pos_emb.shape[1]>x.shape[1]:
                pass
            else:
                pos_emb = pos_emb[:, :x.shape[1], :]
            x = x + pos_emb
        
        x = self.transformer_block(x)
        
        x = x.reshape(-1, x.size(-1)) 
        batch_edge_index = []
        num_nodes_per_sample = h * w
        
        for i in range(batch_size):
            offset = i * num_nodes_per_sample
            batch_edges = edge_index.clone()
            batch_edges = batch_edges + offset
            batch_edge_index.append(batch_edges)
        
        if batch_size > 1:
            batch_edge_index = torch.cat(batch_edge_index, dim=1)
        else:
            batch_edge_index = batch_edge_index[0]
        
        x = self.gcn_block(x, batch_edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.final_gcn(x, batch_edge_index)
        
        x = x.reshape(batch_size, -1, self.num_classes)
        
        x = x.mean(dim=1)
        return F.log_softmax(x, dim=1)
    
class Combined_2(nn.Module):
    def __init__(self , num_classes = 2):
        super(Combined_2 , self).__init__()
        self.cpe = ConditionalPositionEncoding(in_channels = 4 , kernel_size = 3)
        self.gcn = Grapher(in_channels = 4)
        self.one_block = Combined()

    def forward(self , x):
        x = self.cpe(x)
        x = self.gcn(x)
        return x.log_softmax(x)
    

class final(nn.Module):
    def __init__(self , num_classes):
        super(final , self).__init__()
        self.combined_1 = Combined()
        self.combined_2 = Combined_2()
        self.fc = nn.Linear(2*num_classes , num_classes)

    def forward(self , x):
        x1 = self.combined_1(x)
        x2 = self.combined_2(x)

        x = self.fc(torch.cat((x1, x2), dim=1)) 

        return x.log_softmax(x)
    
    



        





        