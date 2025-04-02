import torch 
import torch_geometric 
import torch.nn as nn 
import torch.functional as F
from ViHGNN.assets.gcn_lib.torch_edge import fuzzy_c_means , construct_hyperedges
from Ricci_curv_hyp.OllivierRicci_hypergraphs_edges import OllivierRicciHypergraphEdges
from torch_geometric.nn import GPTConv 
from encoder import PatchEmbed


class create_hypergraph:
    def __init__(self , x , num_clusters , threshold = 0.5 , m = 2 , method = 'OTDSinkhornMix' , nbr_topk = 3000 , verbose = "ERROR" ):
        self.x = x 
        self.num_clusters = num_clusters 
        self.threshold = threshold
        self.m = m 
        self.method = method 
        self.nbr_topk  = nbr_topk
        self.verbose = verbose
        
        hyp_edges = construct_hyperedges(x , num_clusters , threshold , m)
        patch_embed = PatchEmbed()
        
        
        
        

