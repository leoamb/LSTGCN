import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import FermiDiracDecoder
import manifolds
import models.encoders as encoders
from models.encoders import SRBGCN, TBGCN
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from manifolds import LorentzManifold, Sphere
from utils import *
from utils import pre_utils
from utils.pre_utils import *
from layers.CentroidDistance import *
 
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args, First_layer = True, skip_layer = False, Last_layer = False):
        super(BaseModel, self).__init__()
        self.Lmanifold = LorentzManifold(args)
        self.Smanifold = Sphere(args)
        args.feat_dim = args.feat_dim + 1
        self.args = args
        self.First_layer = First_layer
        self.skip_layer = skip_layer
        self.nnodes = args.n_nodes
        self.encoder = []
        self.c = []

        if args.c is not None:
            self.c = torch.Tensor([1.]*(len(args.models_dim))).cuda().to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]*(len(args.models_dim))))
            args.variables.append(self.c)

        if self.skip_layer:
            self.linear = nn.Linear(
                    int(args.feature_dim), int(args.dim[0]+1),
            )
            nn_init(self.linear, 'kaiming')
            self.args.variables.append(self.linear)
            self.activation = nn.SELU()

        for i in range(len(args.models_dim)): 
            self.encoder.append(SRBGCN(args, 1, self.Lmanifold, args.models_dim[i]+1, First_layer, Last_layer))
        self.encoder = nn.ModuleList(self.encoder)

        
    def encode(self, x, hgnn_adj, hgnn_weight, adj_mx):
        for i in range(len(self.args.models_dim)):         
            h = self.encoder[i].encode(x, hgnn_adj, hgnn_weight, adj_mx, self.c[i])
        return h

    def skip_con(self, x, y):

        if self.skip_layer:
            x = self.activation(self.linear(x))
            x = self.Lmanifold.exp_map_zero(x, self.c[0])

        mean = 0.5 * x + 0.5 * y
        l_dot = self.Lmanifold.minkowski_dot(mean, mean, keepdim=False)
        coef = torch.sqrt(self.c[0] / torch.abs(l_dot))
        mean = torch.mul(coef.unsqueeze(-2), mean.transpose(-2, -1)).transpose(-2, -1)

        return mean

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class NRModel(nn.Module):
    """
    Base model for node regression task.
    """

    def __init__(self, args):
        super(NRModel, self).__init__()
        self.spatial_encoders = []
        self.temporal_encoders = []
        self.temporal_attention = []
        self.distance = []
        self.temporal_encoders1 = []
        self.num_layers = args.num_layers
        args.num_layers = 1
        self.Lmanifold = LorentzManifold(args)
        t_kernel = args.t_kernel
        t_ch = args.models_dim[0]*2
        for i in range(self.num_layers):
            self.temporal_encoders1.append(TBGCN(args, 1, self.Lmanifold, t_kernel, args.models_dim[0]+1, t_ch, True if i==0 else False)) 
            self.spatial_encoders.append(BaseModel(args, False, True if i==0 else False, False))
            self.temporal_encoders.append(TBGCN(args, 1, self.Lmanifold, t_kernel, args.models_dim[0]+1, t_ch, False)) 
            self.distance.append(HCentroidDistance(args, 1, self.Lmanifold, args.dim[0]+1))
        self.spatial_encoders = nn.ModuleList(self.spatial_encoders)
        self.temporal_encoders = nn.ModuleList(self.temporal_encoders)
        self.distance = nn.ModuleList(self.distance)
        self.temporal_encoders1 = nn.ModuleList(self.temporal_encoders1)

        fin_ch = args.num_centroid*3

        self.final_conv = nn.Conv2d(int(args.num_centroid*self.num_layers), fin_ch, kernel_size=(1, 12))

        args.variables.append(self.final_conv)

        self.linear = nn.Linear(
                fin_ch, 12,
        )
        args.variables.append(self.linear)
        
    def encode(self, x, hgnn_adj, hgnn_weight, adj_mx):
        x = x.permute(0, 3, 1, 2)
        out_cat = None
        for i in range(self.num_layers):
            y = self.temporal_encoders1[i].encode(x)
            y = self.spatial_encoders[i].encode(y, hgnn_adj, hgnn_weight, adj_mx)
            x = self.spatial_encoders[i].skip_con(x,self.temporal_encoders[i].encode(y))
            if out_cat == None: 
                _,out_cat = self.distance[i](x, 1)
            else:
                out_cat = torch.cat((out_cat,self.distance[i](x, 1)[1]),dim=-1)
        B, T, N, C = out_cat.shape #B C N T
        x = self.final_conv(out_cat.permute(0,3,2,1))[:, :, :, -1].permute(0, 2, 1)
        x = self.linear(x)
        
        return x
    
    def compute_metrics(self, model, val_loader, criterion, epoch, hgnn_adj, hgnn_weight, adj_mx, limit=None):
        '''
        for rnn, compute mean loss on validation set
        :param net: model
        :param val_loader: torch.utils.data.utils.DataLoader
        :param criterion: torch.nn.MSELoss
        :param sw: tensorboardX.SummaryWriter
        :param global_step: int, current global_step
        :param limit: int,
        :return: val_loss
        '''

        model.train(False)  # ensure dropout layers are in evaluation mode

        with torch.no_grad():

            val_loader_length = len(val_loader)  # nb of batch

            tmp = []  

            for batch_index, batch_data in enumerate(val_loader):
                encoder_inputs, labels = batch_data
                outputs = model.encode(encoder_inputs, hgnn_adj, hgnn_weight, adj_mx)
                loss = criterion(outputs, labels)  
                tmp.append(loss.item())
                if batch_index % 100 == 0:
                    print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
                if (limit is not None) and batch_index >= limit:
                    break

            validation_loss = sum(tmp) / len(tmp)
        return validation_loss
   
    def decode(self, h, idx, split):
        print("decode")
    
    def init_metric_dict(self):
        return np.inf

class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        self.margin = args.margin
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        sq_dist = h[0]
        for i in range(1,len(self.args.models_dim)):
            sq_dist += h[i]
        output = self.decoder.decode(torch.sqrt(sq_dist), adj)
        return F.log_softmax(output[idx], dim=1)


    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}'] 
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        correct = output.gather(1, data['labels'][idx].unsqueeze(-1))
        loss = F.relu(self.margin - correct + output).mean()
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx, split):
       # print(h)
        for i in range(len(self.args.models_dim)):
            manifold_emb = h[i]
         #   print(manifold_emb)
            emb_in = manifold_emb[idx[:, 0], :]
            emb_out = manifold_emb[idx[:, 1], :]
            if i ==0:
                sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            #    print(0)
            else:
                sqdist += self.manifold.sqdist(emb_in, emb_out, self.c)
             #   print(1)
          #  print(sqdist)
        probs = self.dc.forward(sqdist, split)
        return probs



    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)] 
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'], split)
        neg_scores = self.decode(embeddings, edges_false, split)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

