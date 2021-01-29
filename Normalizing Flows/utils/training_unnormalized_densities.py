import torch

import numpy as np

from tqdm.auto import trange


### Definitions of unnormalized densities (in 2d)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidp(z):
    return 1/(1+torch.exp(-z))

def w1(z):
    return np.sin(2*np.pi*z[:,0]/4)

def w1p(z):
    return torch.sin(2*np.pi*z[:,0]/4)

def w2(z):
    cpt = ((z[:,0]-1)/0.6)**2
    return 3*np.exp(-0.5*cpt)

def w2p(z):
    cpt = ((z[:,0]-1)/0.6)**2
    return 3*torch.exp(-0.5*cpt)

def w3(z):
    return 3*sigmoid((z[:,0]-1)/3)

def w3p(z):
    return 3*sigmoidp((z[:,0]-1)/3)

def U1(z):
    cpt1 = ((np.linalg.norm(z,axis=1)-2)/0.4)**2
    cpt2 = -0.5*((z[:,0]-2)/0.6)**2
    cpt3 = -0.5*((z[:,0]+2)/0.6)**2
    return cpt1-np.log(np.exp(cpt2)+np.exp(cpt3))

def U1p(z):
    cpt1 = ((torch.sqrt(z[:,0]**2+z[:,1]**2)-2)/0.4)**2
    cpt1 = ((torch.sqrt(torch.sum(z**2,dim=1))-2)/0.4)**2
    cpt2 = -0.5*((z[:,0]-2)/0.6)**2
    cpt3 = -0.5*((z[:,0]+2)/0.6)**2
    return cpt1-torch.log(torch.clamp(torch.exp(cpt2)+torch.exp(cpt3),min=1e-6))

def U2(z):
    return 0.5*((z[:,1]-w1(z))/0.4)**2

def U2p(z):
    return 0.5*((z[:,1]-w1p(z))/0.4)**2

def U3(z):
    cpt1 = ((z[:,1]-w1(z))/0.35)**2
    cpt2 = ((z[:,1]-w1(z)+w2(z))/0.35)**2
    return -np.log(np.exp(-0.5*cpt1)+np.exp(-0.5*cpt2))

def U3p(z):
    cpt1 = ((z[:,1]-w1p(z))/0.35)**2
    cpt2 = ((z[:,1]-w1p(z)+w2p(z))/0.35)**2
    return -torch.log(torch.clamp(torch.exp(-0.5*cpt1)+torch.exp(-0.5*cpt2),min=1e-6))

def U4(z):
    cpt1 = ((z[:,1]-w1(z))/0.4)**2
    cpt2 = ((z[:,1]-w1(z)+w3(z))/0.35)**2
    return -np.log(np.exp(-0.5*cpt1)+np.exp(-0.5*cpt2))

def U4p(z):
    cpt1 = ((z[:,1]-w1p(z))/0.4)**2
    cpt2 = ((z[:,1]-w1p(z)+w3p(z))/0.35)**2
    return -torch.log(torch.clamp(torch.exp(-0.5*cpt1)+torch.exp(-0.5*cpt2),min=1e-6))


def kl_reverse(z, x, log_det, true_density, base_distr):
    """
        Inputs:
        - z: variable in the latent space
        - x: variable in the target space
        - log_det: log_det
        - true_density: function of the unnormalized density
        - base_distr

        Output:
        - "reversed" KL divergence
    """
    log_pz = base_distr.log_prob(z)
    log_pX = torch.log(torch.clamp(true_density(x),min=1e-20)) ## only need unnormalized density
    return torch.mean(log_pz-log_det-log_pX,dim=0)


def train_shapes(model,optimizer,n_batch=500,n_epochs=10001,U=U1p,device=None):
    true_density = lambda x : torch.exp(-U(x))

    d = 2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_distr = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(d,device=device),torch.eye(d,device=device))

    train_loss = []

    pbar = trange(n_epochs)

    for i in pbar:
        model.train()

        z0 = torch.randn(n_batch, d, device=device)
        xs,log_det = model(z0)

        loss = kl_reverse(z0, xs[-1], log_det, true_density, base_distr)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.append(loss.item())
        
        if i % 10==0:
            pbar.set_postfix_str(f"loss = {train_loss[-1]:.3f}")

    return train_loss