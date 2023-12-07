import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from manifolds import *

class HCentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise hyperbolic distances between node representations
    and centroids
    """
    def __init__(self, args, logger, manifold, dim):
        super(HCentroidDistance, self).__init__()
        self.args = args
        self.logger = logger
        self.manifold = manifold
        self.debug = False
        self.dim = dim
    #    print('dim',dim)
        # centroid embedding
        self.centroid_embedding = nn.Embedding(
            args.num_centroid, dim,
            sparse=False,
            scale_grad_by_freq=False,
        )
        nn_init(self.centroid_embedding, self.args.proj_init)
        args.variables.append(self.centroid_embedding)

    def forward(self, node_repr, c):
        """
        Args:
            node_repr: [node_num, dim] #B, T, N, C 
            mask: [node_num, 1] 1 denote real node, 0 padded node
        return:
            graph_centroid_dist: [1, num_centroid]
            node_centroid_dist: [1, node_num, num_centroid]
        """
    #    print(c)
        B, T, node_num, C  = node_repr.shape
    #    print(node_repr.size())
        # broadcast and reshape node_repr to [node_num * num_centroid, dim]
        node_repr =  node_repr.unsqueeze(-2).expand(-1,-1,
                                                -1,
                                                self.args.num_centroid,
                                                -1)
       # print(self.centroid_embedding.weight)
       # print("node_repr",node_repr.shape)
        # broadcast and reshape centroid embeddings to [node_num * num_centroid, dim]
        centroid_repr = self.manifold.exp_map_zero(self.centroid_embedding(th.arange(self.args.num_centroid).cuda().to(self.args.device)),c)
      #  print("centroid_repr",centroid_repr.shape)
        centroid_repr = centroid_repr.unsqueeze(0).expand(B,T,
                                                node_num,
                                                -1,
                                                -1)
       # print("centroid_repr",centroid_repr.shape)
        # get distance
      #  node_centroid_dist = self.manifold.distance(node_repr, centroid_repr, c)
      #  print(node_centroid_dist)
        node_centroid_dist = self.manifold.sqdist(node_repr, centroid_repr, c).squeeze()
       # print(node_centroid_dist.shape)
       # print('loool')
       # node_centroid_dist = node_centroid_dist.view(1, node_num, self.args.num_centroid) 
        # average pooling over nodes
        graph_centroid_dist = th.sum(node_centroid_dist, dim=1) / node_num
        return graph_centroid_dist, node_centroid_dist.view(B, T, node_num, -1)
        
class SCentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise hyperbolic distances between node representations
    and centroids
    """
    def __init__(self, args, logger, manifold, dim):
        super(SCentroidDistance, self).__init__()
        self.args = args
        self.logger = logger
        self.manifold = manifold
        self.debug = False
        self.dim = dim
        # centroid embedding
        self.centroid_embedding = nn.Embedding(
            args.num_centroid, dim,
            sparse=False,
            scale_grad_by_freq=False,
        )
        nn_init(self.centroid_embedding, self.args.proj_init)
        args.variables.append(self.centroid_embedding)

    def forward(self, node_repr, mask, c):
        """
        Args:
            node_repr: [node_num, dim] 
            mask: [node_num, 1] 1 denote real node, 0 padded node
        return:
            graph_centroid_dist: [1, num_centroid]
            node_centroid_dist: [1, node_num, num_centroid]
        """
    #    print(c)
        node_num = node_repr.size(0)
    #    print(node_repr.size())
        # broadcast and reshape node_repr to [node_num * num_centroid, dim]
        node_repr =  node_repr.unsqueeze(1).expand(
                                                -1,
                                                self.args.num_centroid,
                                                -1).contiguous().view(-1, self.dim)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, dim]
        centroid_repr = self.manifold.proj(self.centroid_embedding(th.arange(self.args.num_centroid).cuda().to(self.args.device)),c)
        centroid_repr = centroid_repr.unsqueeze(0).expand(
                                                node_num,
                                                -1,
                                                -1).contiguous().view(-1, self.dim) 
        # get distance
      #  node_centroid_dist = self.manifold.distance(node_repr, centroid_repr, c)
      #  print(node_centroid_dist)
        node_centroid_dist = self.manifold.sqdist(node_repr, centroid_repr, c)
      #  print(node_centroid_dist)
      #  print('loool')
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.args.num_centroid) 
        # average pooling over nodes
        graph_centroid_dist = th.sum(node_centroid_dist, dim=1) / th.sum(mask)
        return graph_centroid_dist, node_centroid_dist

class ECentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise Euclidean distances between node representations
    and centroids
    """
    def __init__(self, args, logger, dim):
        super(ECentroidDistance, self).__init__()
        self.args = args
        self.logger = logger
        self.debug = False
        self.dim = dim
        # centroid embedding
        self.centroid_embedding = nn.Embedding(
            args.num_centroid, dim,
            sparse=False,
            scale_grad_by_freq=False,
        )
        nn_init(self.centroid_embedding, self.args.proj_init)
        args.variables.append(self.centroid_embedding)

    def forward(self, node_repr, mask):
        """
        Args:
            node_repr: [node_num, dim] 
            mask: [node_num, 1] 1 denote real node, 0 padded node
        return:
            graph_centroid_dist: [1, num_centroid]
            node_centroid_dist: [1, node_num, num_centroid]
        """
        node_num = node_repr.size(0)
    #    print(node_repr.size())
        # broadcast and reshape node_repr to [node_num * num_centroid, dim]
        node_repr =  node_repr.unsqueeze(1).expand(
                                                -1,
                                                self.args.num_centroid,
                                                -1).contiguous().view(-1, self.dim)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, dim]
        centroid_repr = self.centroid_embedding(th.arange(self.args.num_centroid).cuda().to(self.args.device))
        centroid_repr = centroid_repr.unsqueeze(0).expand(
                                                node_num,
                                                -1,
                                                -1).contiguous().view(-1, self.dim) 
        # get distance
        node_centroid_dist = th.sum(th.square(node_repr - centroid_repr) ,dim = 1)
      #  print(th.sum(th.square(node_repr[0]-centroid_repr[0])))
      #  print(centroid_repr[0])
      #  print(node_centroid_dist[0])
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.args.num_centroid) 
        # average pooling over nodes
        graph_centroid_dist = th.sum(node_centroid_dist, dim=1) / th.sum(mask)
        return graph_centroid_dist, node_centroid_dist