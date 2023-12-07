import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import manifolds
import utils.math_utils as pmath
import torch as th
import geotorch as geoth
from utils import *
from utils import pre_utils
from utils.pre_utils import *
from manifolds import *
#from manifolds import LorentzManifold
from layers.CentroidDistance import *
        
class TBGCN(nn.Module):

    def __init__(self, args, logger, manifold, t_kernel, ch_in, ch_out, First_layer):
        super(TBGCN, self).__init__()
        self.debug = False
        self.args = args
        self.logger = logger
        self.manifold = manifold
        self.t_kernel = t_kernel
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.First_layer = First_layer
        self.eps = 1e-14
        self.padding = True
        self.c = torch.Tensor([1.]).cuda().to(args.device)
        self.max_norm = 10000

        H = th.ones((ch_out, t_kernel,ch_in), requires_grad=True)/np.sqrt(ch_in-1)
        nn.init.kaiming_uniform_(H)
        H = nn.Parameter(H)

        self.args.variables.append(H)

        self.activation = nn.SELU()

        if self.First_layer:
            self.linear = nn.Linear(
                    int(args.feature_dim), int(self.ch_in),
            )
            nn_init(self.linear, 'kaiming')
            self.args.variables.append(self.linear)


        setattr(self, "msg_temporal_kernel", H)
        
    def create_params_h(self):
        """
        create the GNN params for hyperbolic rotation transformation using basic hyperbolic rotations
        """
        h_weight = []
        #for iii in range(layer):
        H = th.randn(self.ch_dim-1, requires_grad=True)*0.01
        nn.init.uniform_(H, -0.001, 0.001)
        H = nn.Parameter(H)
        self.args.variables.append(H)
        h_weight.append(H)
        return nn.ParameterList(h_weight)
        
    def test_lor(self, A):
        tmp1 = (A[:,0] * A[:,0]).view(-1)
        tmp2 = A[:,1:]
        tmp2 = th.diag(tmp2.mm(tmp2.transpose(0,1)))
        return (tmp1 - tmp2)

    def lorentz_mean(self, y, dim=0, c=1.0, ):

        num_nodes = y.size(1)
        nu_sum = torch.sum(y, dim = 1)/num_nodes
        l_dot = self.manifold.minkowski_dot(nu_sum,nu_sum,keepdim=False)
        coef = torch.sqrt(c / torch.abs(l_dot))
        mean = torch.mul(coef.unsqueeze(-2), nu_sum.transpose(-2, -1)).transpose(-2, -1)
        return mean

    def get_h_t_params(self):
        """
        retrieve the GNN parameters for hyperbolic rotation using an axis and hyperbolic angle
        Args:
            weight_h: a list of weights
            step: a certain layer
        """
        tmp = getattr(self, "msg_temporal_kernel")
        v_d = tmp[...,1:].view(-1,self.ch_in-1)
        n_d = v_d/th.sqrt(v_d.pow(2).sum(-1, keepdim=True)+self.eps)
        C_I = ((1-pmath.cosh(tmp[...,0].view(-1)))*th.einsum("bi,bj->bij",n_d,n_d).T).T
        C = th.eye(self.ch_in-1).cuda().to(self.args.device)-C_I
        layer_weight = th.cat(((pmath.sinh(tmp[...,0].view(-1))*n_d.T).T.unsqueeze(-1), C), dim=-1)
        aB = th.cat([pmath.cosh(tmp[...,0].view(-1)).unsqueeze(0),(pmath.sinh(tmp[...,0].view(-1))*n_d.T)]).T
        layer_weight = th.cat((aB.unsqueeze(1), layer_weight), dim=-2)
        layer_weight = layer_weight.view(self.ch_out, self.t_kernel,self.ch_in,self.ch_in)
        return layer_weight

    def apply_activation(self, node_repr, c):
        """
        apply non-linearity for different manifolds
        """
        if self.args.select_manifold in {"poincare", "euclidean"}:
            return self.activation(node_repr)
        elif self.args.select_manifold == "lorentz":
            return self.manifold.from_poincare_to_lorentz(
                self.activation(self.manifold.from_lorentz_to_poincare(node_repr, c)),c
            )

    def encode(self, node_repr):
        weight = self.get_h_t_params()
        if self.First_layer:
            node_repr = self.activation(self.linear(node_repr))
            node_repr = self.manifold.exp_map_zero(node_repr, 1)                   

        B, T, N, C = node_repr.shape

        if self.padding and self.t_kernel!=1:
            out = node_repr[:,:int((self.t_kernel-1)/2),:,:]
        else:
            out = None


        for j in range(node_repr.size(1)-self.t_kernel+1):
            
            com_msg = None
            for k in range(self.t_kernel):
                msg = th.matmul(node_repr[:,j+k,:,:].unsqueeze(1),weight[:,k,:,:])   
                if com_msg == None:
                    com_msg = msg.unsqueeze(2)
                else:
                    com_msg = th.cat((com_msg,msg.unsqueeze(2)),dim=2)
            
            if out==None:
                out = self.lorentz_mean(com_msg.view(B,-1,N,C)).unsqueeze(1)
            else:
                out = th.cat((out,self.lorentz_mean(com_msg.view(B,-1,N,C)).unsqueeze(1)),dim=1)
             
        if self.padding and self.t_kernel!=1:
            out = th.cat((out,node_repr[:,-int((self.t_kernel-1)/2):,:,:]),dim=1)

        out = self.apply_activation(out, self.c)
        out = self.manifold.normalize(out, self.c, self.max_norm).view(B, T, N, C)
        return out

class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

class SRBGCN(nn.Module):

    def __init__(self, args, logger, manifold, dim, First_layer, Last_layer):
        super(SRBGCN, self).__init__()
        self.debug = False
        self.args = args
        self.logger = logger
        self.manifold = manifold
        self.First_layer = First_layer
        self.Last_layer = Last_layer
        self.max_norm = 10000
        self.dim = [dim] * args.num_layers
        tie_list = [0]*len(self.dim)
        counter = 0;
        for i in range(1,len(self.dim)):
            if args.dim[i] == args.dim[i-1]:
                tie_list[i] = counter
            else:
                tie_list[i] = counter + 1
                counter = counter + 2
        self.tie_list = tie_list            
        self.eps = 1e-14
        
        if not self.args.tie_weight:
            self.tie_list = [i for i in range(len(args.dim))]
        
        self.set_up_boost_params()
        self.activation = nn.SELU()
        
        if self.First_layer:
            self.linear = nn.Linear(
                    int(args.feature_dim), int(self.dim[0]),
            )
            nn_init(self.linear, 'kaiming')
            self.args.variables.append(self.linear)

        self.msg_weight = []
        layer = self.args.num_layers if not self.args.tie_weight else self.tie_list[-1]+1

        for iii in range(layer):
            if iii == 0:
                M =  nn.Linear(self.dim[self.tie_list.index(iii)]-1, self.dim[self.tie_list.index(iii)]-1,bias=0)
            else:
                M = nn.Linear(self.dim[self.tie_list.index(iii)]-1, self.dim[self.tie_list.index(iii-1)]-1,bias=0)
         
            geoth.orthogonal(M,"weight","cayley")

            self.args.variables.append(M)
            self.msg_weight.append(M)
        self.msg_weight = nn.ModuleList(self.msg_weight)

        self.use_att = args.use_att

        if self.use_att:
            self.key_linear = LorentzLinear(self.manifold, self.dim[-1], self.dim[-1])
            self.query_linear = LorentzLinear(self.manifold, self.dim[-1], self.dim[-1])
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(self.dim[-1],))
            args.variables.append(self.key_linear)
            args.variables.append(self.query_linear)
            args.variables.append(self.bias)
            args.variables.append(self.scale)

    def create_boost_params(self):
        """
        create the GNN params for hyperbolic rotation transformation using axis and hyperbolic angle 
        """
        h_weight = []
        layer = self.args.num_layers if not self.args.tie_weight else self.tie_list[-1]+1
        for iii in range(layer):
            H = th.randn(self.dim[self.tie_list.index(iii)], requires_grad=True)*0.01
            H = th.ones(self.dim[self.tie_list.index(iii)], requires_grad=True)/np.sqrt(self.dim[self.tie_list.index(iii)]-1)
            H[0] = 0
            H = nn.Parameter(H)
            self.args.variables.append(H)
            h_weight.append(H)
        return nn.ParameterList(h_weight)
        

    def set_up_boost_params(self):
        """
        set up the params for all message types
        """
        self.type_of_msg = 1
        
        for i in range(0, self.type_of_msg):
            if self.args.hyp_rotation:
                setattr(self, "msg_%d_weight_h" % i, self.create_boost_params())

    def apply_activation(self, node_repr, c):
        """
        apply non-linearity for different manifolds
        """
        if self.args.select_manifold in {"poincare", "euclidean"}:
            return self.activation(node_repr)
        elif self.args.select_manifold == "lorentz":
            return self.manifold.from_poincare_to_lorentz(
                self.activation(self.manifold.from_lorentz_to_poincare(node_repr, c)),c
            )
    
    def split_input(self, adj_mat, weight):
        return [adj_mat], [weight]

    def p2k(self, x, c):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom


    def lorenz_factor(self, x, *, c=1.0, dim=-1, keepdim=False):
        """
            Calculate Lorenz factors
        """
        x_norm = x.pow(2).sum(dim=dim, keepdim=keepdim)
        x_norm = torch.clamp(x_norm, 0, 0.9)
        tmp = 1 / torch.sqrt(1 - c * x_norm)
        return tmp
     
    def from_lorentz_to_poincare(self, x):
        """
        Args:
            u: [batch_size, d + 1]
        """
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def h2p(self, x):
        return self.from_lorentz_to_poincare(x)

    def from_poincare_to_lorentz(self, x, eps=1e-3):
        """
        Args:
            u: [batch_size, d]
        """
        x_norm_square = x.pow(2).sum(-1, keepdim=True)
        tmp = th.cat((1 + x_norm_square, 2 * x), dim=1)
        tmp = tmp / (1 - x_norm_square)
        return  tmp

    def p2h(self, x):
        return  self.from_poincare_to_lorentz(x)

    def p2k(self, x, c=1.0):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    
    def test_lor(self, A):
        tmp1 = (A[:,0] * A[:,0]).view(-1)
        tmp2 = A[:,1:]
        tmp2 = th.diag(tmp2.mm(tmp2.transpose(0,1)))
        return (tmp1 - tmp2)
    
    def lorentz_mean(self, y, node_num, max_neighbor, weight, dim=0, c=1.0, ):

        B, T, N, C = y.shape
        nu_sum = torch.mul(y.transpose(-2,-1).reshape(-1, node_num, max_neighbor),weight).sum(-1).view(B,T,C,node_num).transpose(-2,-1)
        l_dot = self.manifold.minkowski_dot(nu_sum,nu_sum,keepdim=False)
        coef = torch.sqrt(c / torch.abs(l_dot))
        mean = torch.mul(coef.unsqueeze(-2), nu_sum.transpose(-2, -1)).transpose(-2, -1)

        return mean

    def retrieve_params(self, weight, step):
        """
        Args:
            weight: a list of weights
            step: a certain layer
        """
        weight = weight[step].weight
       # print(weight)
        layer_weight = th.cat((th.zeros((weight.size(0), 1)).cuda().to(self.args.device), weight), dim=1)
        tmp = th.zeros((1, weight.size(1)+1)).cuda().to(self.args.device)
        tmp[0,0] = 1
        layer_weight = th.cat((tmp, layer_weight), dim=0)
        return layer_weight
    
    def retrieve_params_h(self, weight_h, step):
        """
        retrieve the GNN parameters for hyperbolic rotation using an axis and hyperbolic angle
        Args:
            weight_h: a list of weights
            step: a certain layer
        """
        tmp = weight_h[step]
        v_d = tmp[1:]
        n_d = v_d/th.sqrt(v_d.pow(2).sum(-1, keepdim=True)+self.eps)
        C = th.eye(tmp.size(0)-1).cuda().to(self.args.device)-(1-pmath.cosh(tmp[0]))*th.outer(n_d,n_d)
        layer_weight = th.cat((pmath.sinh(tmp[0])*n_d.reshape((-1, 1)), C), dim=1)
        aB = th.cat([pmath.cosh(tmp[0]).reshape(1),pmath.sinh(tmp[0])*n_d])
        layer_weight = th.cat((aB.reshape((1, -1)), layer_weight), dim=0)
        return layer_weight
    
    def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, layer_weight_h, adj_mx, c):
        """
        message passing for a specific message type.
        """
        node_num, max_neighbor = adj_mat.shape[0], adj_mat.shape[1] 
        combined_msg = node_repr.clone()
        msg = th.matmul(node_repr, layer_weight.T)
      
        if(self.args.hyp_rotation):
            msg = th.matmul(msg, layer_weight_h)
        
        if self.use_att:
            query = self.query_linear(node_repr)
            key = self.key_linear(node_repr)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            support_t = torch.matmul(att_adj, msg)

            
        else:
            support_t = torch.matmul(adj_mx, msg)

        denom = (self.manifold.minkowski_dot(support_t, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        combined_msg = support_t / denom

        return combined_msg 

    def get_combined_msg(self, step, node_repr, adj_mat, weight, adj_mx, c):
        """
        perform message passing in the tangent space of x'
        """
        gnn_layer = self.tie_list[step] if self.args.tie_weight else step
        combined_msg = None
        for relation in range(0, self.type_of_msg):
            layer_weight = self.retrieve_params(self.msg_weight, gnn_layer)
            layer_weight_h = self.retrieve_params_h(getattr(self, "msg_%d_weight_h" % relation), gnn_layer) if self.args.hyp_rotation else None
            aggregated_msg = self.aggregate_msg(node_repr,
                                                adj_mat[relation],
                                                weight[relation],
                                                layer_weight,layer_weight_h, adj_mx, c)
            combined_msg = aggregated_msg if combined_msg is None else (combined_msg + aggregated_msg)
        return combined_msg


    def encode(self, node_repr, adj_list, weight, adj_mx, c):
        
        adj_list, weight = self.split_input(adj_list, weight)

        if self.First_layer:
            node_repr = self.activation(self.linear(node_repr))
            node_repr = self.manifold.exp_map_zero(node_repr, c)

        B, T, N, C = node_repr.shape
        for step in range(self.args.num_layers):
            node_repr = self.get_combined_msg(step, node_repr, adj_list, weight, adj_mx, c)
            node_repr = self.apply_activation(node_repr, c)
            node_repr = self.manifold.normalize(node_repr, c, self.max_norm)

        if self.args.task == 'nc' or self.args.task == 'nr' and self.Last_layer:
            _, node_centroid_sim = self.distance(node_repr, c) 
            return node_centroid_sim.squeeze()

        return node_repr.view(B, T, N, C)
