"""Lorentz and spherical manifolds."""
import torch
import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable
import torch
from utils import *
from utils.pre_utils import *
from manifolds import *
from utils.math_utils import arcosh, cosh, sinh 

_eps = 1e-10

class LorentzManifold:

    def __init__(self, args, eps=1e-3, norm_clip=1, max_norm=7.5e2):
        self.args = args
        self.eps = eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm

    def minkowski_dot(self, x, y, keepdim=True):
       # z = x * y
       # print("z",z.shape)

        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        #print("res",res.shape)
        if keepdim:
            res = res.view(res.shape + (1,))
        return res


    def sqdist(self, x, y, c):
        
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        theta = torch.clamp(-prod / K, min=1.0 + eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)
    
       
    
    @staticmethod
    def dist(x, y, c, keepdim=False):
        
        K = 1. / c
        prod = np.sum(x * y, axis=-1,keepdims=keepdim) - 2 * x[..., 0] * y[..., 0]
        eps = {'float32': 1e-7, 'float64': 1e-15}
     #   print(x.dtype)
      #  print(eps[x.dtype])
        theta = np.clip(-prod / K, a_min=1.0 + 1e-7, a_max=None)
        dist = np.sqrt(K) * np.arccosh(theta)
        return dist
        
    def distance(self, u, v, c):
        K = 1. / c
        d = -LorentzDot.apply(u, v)
        dis = torch.sqrt(K) * Acosh.apply(d, self.eps)
        return dis

    @staticmethod
    def ldot(u, v, keepdim=False):
        """
        Lorentzian Scalar Product
        Args:
            u: [batch_size, d + 1]
            v: [batch_size, d + 1]
        Return:
            keepdim: False [batch_size]
            keepdim: True  [batch_size, 1]
        """
        d = u.size(-1) - 1
        uv = u * v
        uv = th.cat((-uv.narrow(-1, 0, 1), uv.narrow(-1, 1, d)), dim=-1) 
        return th.sum(uv, dim=-1, keepdim=keepdim)
        
    @staticmethod    
    def from_L_to_P(x,c):
        """
        Args:
            u: [batch_size, d + 1] numpy array
        """
      #  d = x.shape[-1] - 1
    #    print('hena1')
        K = 1. / c
        sqrtK = K ** 0.5
        return x[..., 1:] / (x[..., 0:1] + sqrtK)    

    def from_lorentz_to_poincare(self, x, c):
        """
        Args:
            u: [batch_size, d + 1]
        """
     #   print('hena2')
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + sqrtK)

    def from_poincare_to_lorentz(self, x, c):
        """
        Args:
            u: [batch_size, d]
        """
     #   print('hena3')
        K = 1./ c
        sqrtK = K ** 0.5
        x_norm_square = th_dot(x, x)
      #  print(x_norm_square.shape)
        return sqrtK * th.cat((K + x_norm_square, 2 * sqrtK * x), dim=-1) / (K - x_norm_square + self.eps)
    '''
    def distance(self, u, v):
        d = -LorentzDot.apply(u, v)
        dis = Acosh.apply(d, self.eps)
        return dis
    '''
    def normalize(self, w, c, max_norm):
        """
        Normalize vector such that it is located on the Lorentz
        Args:
            w: [batch_size, d + 1]
        """
        K = 1./ c
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
      #  print(narrowed.view(-1, d).shape)
        if max_norm:
          #  print('henaa')
            narrowed_renorm = th.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
            narrowed = narrowed_renorm.view(narrowed.shape)
        first = K + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
        first = th.sqrt(first)
        tmp = th.cat((first, narrowed), dim=-1)
        return tmp

    def init_embed(self, embed, c, irange=1e-2):
        print('gah f3lan')
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data,c))
    '''
    def rgrad(self, p, d_p):
        """Riemannian gradient for Lorentz"""
        u = d_p
        x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p
    '''
    def exp_map_zero(self, v, c):
        zeros = th.zeros_like(v)
        zeros[..., 0] = 1
      #  print(zeros.shape)
      #  print(zeros[0:3,0,0,:])
        return self.exp_map_x(zeros, v, c)

    def exp_map_x(self, p, d_p, c, d_p_normalize=True, p_normalize=True):
        if d_p_normalize:
            d_p = self.normalize_tan(p, d_p,c)
        
        K = 1. / c
        sqrtK = K ** 0.5
        
      #  print("d_p", d_p.shape)
        ldv = self.ldot(d_p, d_p, keepdim=True)
       # print("ldv", ldv.shape)
        nd_p = th.sqrt(th.clamp(ldv + self.eps, _eps))/ sqrtK

        t = th.clamp(nd_p, max=self.norm_clip)
        newp = (th.cosh(t) * p) + (th.sinh(t) * d_p / nd_p)

        if p_normalize:
            newp = self.normalize(newp,c, 1000)
       # print("newp", newp.shape)
        return newp

    def normalize_tan(self, x_all, v_all, c):
      #  print(c)
        K = 1./ c
        d = v_all.size(-1) - 1
       # print(d)
        x = x_all.narrow(-1, 1, d)
      #  print(x.shape)
        xv = th.sum(x * v_all.narrow(-1, 1, d), dim=-1, keepdim=True)
        tmp = K + th.sum(th.pow(x_all.narrow(-1, 1, d), 2), dim=-1, keepdim=True)
        tmp = th.sqrt(tmp)
        return th.cat((xv / tmp, v_all.narrow(-1, 1, d)), dim=-1)
    
    '''
    def log_map_zero(self, y, i=-1):
        zeros = th.zeros_like(y)
        zeros[:, 0] = 1
        return self.log_map_x(zeros, y)
    
    def log_map_x(self, x, y, normalize=False):
        """Logarithmic map on the Lorentz Manifold"""
        xy = self.ldot(x, y).unsqueeze(-1)
        tmp = th.sqrt(th.clamp(xy * xy - 1 + self.eps, _eps))
        v = Acosh.apply(-xy, self.eps) / (
            tmp
        ) * th.addcmul(y, xy, x)
        if normalize:
            result = self.normalize_tan(x, v,c)
        else:
            result = v
        return result

    def parallel_transport(self, x, y, v):
        """Parallel transport for Lorentz"""
        v_ = v
        x_ = x
        y_ = y

        xy = self.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        return vnew
    '''
    def metric_tensor(self, x, u, v):
        return self.ldot(u, v, keepdim=True)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u

class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps): 
        z = th.sqrt(th.clamp(x * x - 1 + eps, _eps))
        ctx.save_for_backward(z)
        ctx.eps = eps
        xz = x + z
        tmp = th.log(xz)
        return tmp

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None

class Sphere:

    def __init__(self, args, eps=1e-3, norm_clip=1, max_norm=1e3):
        self.args = args
        self.eps = eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
    
    def e_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1)
    #    print(x,y,res)
        if keepdim:
            res = res.view(res.shape + (1,))
        return res
        
    def normalize(self, w, c):
        """
        Normalize vector such that it is located on the sphere
        Args:
            w: [batch_size, d + 1]
        """
        K = 1.0/c
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
       # if self.max_norm:
       #     narrowed = th.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        first = torch.clamp(K - th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True), min = eps[narrowed.dtype])
     #   print(first)
        first = th.sqrt(first)
        tmp = th.cat((first, narrowed), dim=-1)
        return tmp
        
    def proj(self,x,c):
        K = 1.0/c
        eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        return torch.sqrt(K)*x/(torch.norm(x,2,-1,True)+eps[x.dtype])
    
    def sqdist(self, x, y, c):    
        theta = self.e_dot(x,y) * c
        K = 1.0/c
    #    print(theta)
        sqdist = K * (torch.arccos(torch.clamp(theta, min = -1.0, max=1.0))) ** 2
        return torch.clamp(sqdist, max=50.0)