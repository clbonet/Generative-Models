from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform


device = "cuda" if torch.cuda.is_available() else "cpu"

class BaseNormalizingFlow(ABC,nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,x):
        pass

    @abstractmethod
    def backward(self,z):
        pass


class AdditiveCoupling(BaseNormalizingFlow):
    """
        NICE
    """
    def __init__(self, coupling, parity):
        super().__init__()
        self.parity = parity
        self.coupling = coupling
        
    def forward(self, x):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            m = self.coupling(x1)
            z0 = x0+m
            z1 = x1
        else:
            m = self.coupling(x0)
            z0 = x0
            z1 = x1+m
            
        z = torch.cat([z0,z1], dim=1)
        return z,torch.zeros(x.shape[0],device=device)
    
    def backward(self, z):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            m = self.coupling(z1)
            x0 = z0-m
            x1 = z1
        else:
            m = self.coupling(z0)
            x0 = z0
            x1 = z1-m
            
        x = torch.cat([x0,x1], dim=1)
        return x, torch.zeros(z.shape[0],device=device)


class Scale(BaseNormalizingFlow):
    """
        NICE
    """
    def __init__(self, d):
        super().__init__()
        self.log_s = nn.Parameter(torch.randn(1, d, requires_grad=True))
        
    def forward(self, x):
        return torch.exp(self.log_s)*x, torch.sum(self.log_s, dim=1)
    
    def backward(self, z):
        return torch.exp(-self.log_s)*z, -torch.sum(self.log_s, dim=1)


class AffineCoupling(BaseNormalizingFlow):
    """
        RealNVP
    """
    def __init__(self, scaling, shifting, parity):
        super().__init__()
        self.scaling = scaling
        self.shifting = shifting
        self.parity = parity

    def forward(self, x):
        x0, x1 = x[:,::2], x[:,1::2]

        if self.parity:
            s = self.scaling(x1)
            t = self.shifting(x1)
            z0 = torch.exp(s)*x0+t
            z1 = x1
        else:
            s = self.scaling(x0)
            t = self.shifting(x0)
            z0 = x0
            z1 = torch.exp(s)*x1+t

        z = torch.cat([z0,z1], dim=1)
        return z, torch.sum(s, dim=1)


    def backward(self, z):
        z0, z1 = z[:,::2], z[:,1::2]

        if self.parity:
            s = self.scaling(z1)
            t = self.shifting(z1)
            x0 = torch.exp(-s)*(z0-t)
            x1 = z1
        else:
            s = self.scaling(z0)
            t = self.shifting(z0)
            x0 = z0
            x1 = torch.exp(-s)*(z1-t)
        
        x = torch.cat([x0,x1], dim=1)
        return x, -torch.sum(s, dim=1)
        

class BatchNorm(BaseNormalizingFlow):
    """
        Ref: https://github.com/acids-ircam/pytorch_flows/blob/master/flows_04.ipynb
        and Masked Autoregressive Flows for Density Estimation / Density Estimation Using Real NVP
    """
    def __init__(self, d, eps=1e-5, momentum=0.95):
        super().__init__()
        self.eps = eps
        self.momentum = momentum ## To compute train set mean
        self.train_mean = torch.zeros(d, device=device)
        self.train_var = torch.ones(d, device=device)

        self.gamma = nn.Parameter(torch.ones(d, requires_grad=True))
        self.beta = nn.Parameter(torch.ones(d, requires_grad=True))

    def forward(self, x):
        """
            mean=batch_mean in training time, mean of the entire dataset in test
        """
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = (x-x.mean(0)).pow(2).mean(0)+self.eps

            self.train_mean = self.momentum*self.train_mean+(1-self.momentum)*self.batch_mean
            self.train_var = self.momentum*self.train_var+(1-self.momentum)*self.batch_var

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        z = torch.exp(self.gamma)*(x-mean)/var.sqrt()+self.beta
        log_det = torch.sum(self.gamma-0.5*torch.log(var))
        return z, log_det

    def backward(self, z):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        x = (z-self.beta)*torch.exp(-self.gamma)*var.sqrt()+mean
        log_det = torch.sum(-self.gamma+torch.log(var))
        return x, log_det
        
        
class LUInvertible(BaseNormalizingFlow):
    """
        https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
        https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.W = torch.Tensor(dim, dim)
        nn.init.orthogonal_(self.W)

        # P, L, U = torch.lu_unpack(*self.W.lu())
        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.S = nn.Parameter(torch.from_numpy(U).diag())
        self.U = nn.Parameter(torch.triu(torch.from_numpy(U),1))

    def forward(self, x):
        P = self.P.to(device)
        L = torch.tril(self.L,-1)+torch.diag(torch.ones(self.dim,device=device))
        U = torch.triu(self.U,1)+torch.diag(self.S)
        W = P @ L @ U
        return x@W, torch.sum(torch.log(torch.abs(self.S)))

    def backward(self, z):
        P = self.P.to(device)
        L = torch.tril(self.L,-1)+torch.diag(torch.ones(self.dim, device=device))
        U = torch.triu(self.U,1)+torch.diag(self.S)
        W = P @ L @ U
        return z@torch.inverse(W), -torch.sum(torch.log(torch.abs(self.S)))


class PlanarFlow(BaseNormalizingFlow):
    """
        Variational Inference with NF, https://arxiv.org/pdf/1505.05770.pdf
    """
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim, requires_grad=True))
        self.w = nn.Parameter(torch.randn(1, dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, 1, requires_grad=True))

    def forward(self, x):
        ## enforce invertibility
        wu = self.w@self.u.t()
        m_wu = -1+torch.log(1+torch.exp(wu))
        u_hat = self.u+(m_wu-wu)*self.w/torch.sum(self.w**2)

        z = x+u_hat*torch.tanh(x@self.w.t()+self.b)
        psi = (1-torch.pow(torch.tanh(x@self.w.t()+self.b),2))*self.w
        log_det = torch.log(1+psi@u_hat.t())
        return z, log_det[:,0]

    def backward(self, z):
        ## can't compute it analytically
        return NotImplementedError
        

class NormalizingFlows(BaseNormalizingFlow):
    """
        Composition of flows
        
        (ref: https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py)
    """
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        
    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=device)
        zs = [x]
        for flow in self.flows:
            x, log_det_i = flow(x)
            log_det += log_det_i
            zs.append(x)
        return zs, log_det
    
    def backward(self, z):
        log_det = torch.zeros(z.shape[0], device=device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, log_det_i = flow.backward(z)
            log_det += log_det_i
            xs.append(z)
        return xs, log_det
        