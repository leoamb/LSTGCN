"""Graph decoders."""
import manifolds
import math
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import Linear
import torch as th

import geoopt
from geoopt import ManifoldParameter

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class LorentzDecoder(Decoder):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c, args):
        super(LorentzDecoder, self).__init__(c)
        self.input_dim = args.num_centroid
        self.output_dim = args.n_classes
        act = lambda x: x
        self.cls = Linear(args, self.input_dim, self.output_dim, args.dropout, act, args.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = x
        return super(LorentzDecoder, self).decode(h, adj)
        
class LDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LorentzDecoder, self).__init__(c)
        self.input_dim = args.dim[-1]
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.manifold = args.manifold
        self.device = args.device
        
        std = 1./math.sqrt(args.dim[-1])
        tens = th.randn([args.n_classes, args.dim[-1]], requires_grad=True)*std
        tens /= tens.norm(dim=-1, keepdim=True)
        self.cls = nn.Parameter(tens)
        args.variables.append(self.cls)
        if args.bias:
            self.bias = nn.Parameter(th.zeros(args.n_classes))
            args.variables.append(self.bias)
        self.decode_adj = False

   # def decode(self, x, adj):
       # self.manifold.exp_map_zero(self.cls)
       # return (2 + 2 * self.manifold.cinner(x, self.manifold.exp_map_zero(self.cls))) + self.bias
        
    
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        r"""
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.
        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device
        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid
        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
      #  self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
        tens = th.randn(*size) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
    #    self.args.variables.append(nn.Parameter(tens))
        return geoopt.ManifoldTensor(self.args.manifold.exp_map_zero(tens), manifold=self.args.manifold)

model2decoder = {
    'SRBGCN': LorentzDecoder,
}

