import torch
import numpy as np


def minkowski_ip(x, y, keepdim=True):
    if len(x.shape)==1:
        x = x.reshape(1,-1)
    if len(y.shape)==1:
        y = y.reshape(1,-1)
    
    if x.shape[0] != y.shape[0]:
        return -x[...,0][None]*y[...,0][:,None] + torch.sum(x[...,1:][None]*y[...,1:][:,None], axis=-1)
    else:
        return (-x[...,0]*y[...,0])[:,None] + torch.sum(x[...,1:]*y[...,1:], axis=-1, keepdim=True)
    
def minkowski_ip2(x, y):
    """
        Return a n x m matrix where n and m are the number of batchs of x and y.
    """
    return -x[:,0][None]*y[:,0][:,None] + torch.sum(x[:,1:][None]*y[:,1:][:,None], axis=-1)


def lorentz_to_poincare(y, r=1):
    return r*y[...,1:]/(r+y[...,0][:,None])

# def poincare_to_lorentz(x):
#     norm_x = np.linalg.norm(x, axis=-1)[:,None]
#     return np.concatenate([1+norm_x**2, 2*x], axis=-1)/(1-norm_x**2)

def poincare_to_lorentz(x):
    norm_x = torch.linalg.norm(x, axis=-1)[:,None]
    return torch.cat([1+norm_x**2, 2*x], axis=-1)/(1-norm_x**2)


def sum_mobius(z, y, r=1):
    ip = torch.sum(z*y, axis=-1)
    y_norm2 = torch.sum(y**2, axis=-1)
    z_norm2 = torch.sum(z**2, axis=-1)
    num = (1+2*r*ip+r*y_norm2)[:,None]*z + (1-r*z_norm2)[:,None]*y
    denom = 1+2*r*ip+r**2*z_norm2*y_norm2
    return num/denom[:,None]

def prod_mobius(r, x):
    norm_x = torch.sum(x**2, axis=-1)**(1/2)
    return torch.tanh(r[:,None]*torch.arctanh(norm_x)) * x/norm_x

def dist_poincare(x, y, r=1):
    num = torch.linalg.norm(x-y, axis=-1)**2
    denom = (1-r*torch.linalg.norm(y, axis=-1)**2) * (1-r*torch.linalg.norm(x, axis=-1)**2)
    frac = num/denom
    return torch.arccosh(1+2*r*frac)/np.sqrt(r)


def projection(x, x0, v):
    ip_x0_x = minkowski_ip(x0, x)
    ip_v_x = minkowski_ip(v, x)
        
    if v.shape[0] != x.shape[0]:
        num = -(ip_x0_x[:,None]*x0) + ip_v_x[:,:,None]*v[None]
        denom = torch.sqrt((ip_x0_x)**2 - ip_v_x**2)[:,:,None]
    else:
        num = -ip_x0_x*x0 + ip_v_x*v
        denom = torch.sqrt((ip_x0_x)**2 - ip_v_x**2)
    proj = num/denom
    return proj


def parallelTransport(v, x0, x1):
    """
        Transport v\in T_x0 H to u\in T_x1 H by following the geodesics by parallel transport
    """
    n, d = v.shape
    if len(x0.shape)==1:
        x0 = x0.reshape(-1, d)
    if len(x1.shape)==1:
        x1 = x1.reshape(-1, d)
        
    u = v + minkowski_ip(x1, v)*(x0+x1)/(1-minkowski_ip(x1, x0))
    return u

def expMap(u, x):
    """
        Project u\in T_x H to the surface
    """
    
    if len(x.shape)==1:
        x = x.reshape(1, -1)
    
    norm_u = minkowski_ip(u,u)**(1/2)
    y = torch.cosh(norm_u)*x + torch.sinh(norm_u) * u/norm_u
    return y