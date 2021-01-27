import torch

import matplotlib.pyplot as plt
import numpy as np

from tqdm.auto import trange


def loss(h,log_det,base_distr):
    prior = base_distr.log_prob(h).mean()
    return -(prior+log_det.mean())

def log_likelihood(h,log_det,base_distr):
    prior = base_distr.log_prob(h)
    return prior+log_det

    
def uniform_quantization(img):
    return (img*255+torch.rand(img.size()))/256

def rescale_logit(img,lambd=1e-6):
    ## logit space
    return torch.logit(lambd+(1-2*lambd)*img)

def inverse_logit(img,lambd=1e-6):
    return (torch.sigmoid(img)-lambd)/(1-2*lambd)
    

def val_mnist(model, device):
    model.eval()

    d = 28*28
    torch.manual_seed(42)
    r,c = 5,5
    z_random = torch.randn(r,c,d,device=device)
    model.eval()
    zs,log_det = model.backward(z_random.reshape(-1,28*28))
    gen_imgs = inverse_logit(zs[-1].view(-1,28,28).detach().cpu())

    cpt = 0
    fig,ax = plt.subplots(r,c)
    for i in range(r):
        for j in range(c):
            ax[i,j].imshow(gen_imgs[cpt],"gray")
            cpt += 1
    plt.show()


def train_mnist(model, optimizer, train_loader, n_epochs=101, device=None,
                plot_val=True, plot_interval=50):
    d = 784

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_distr = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(d,device=device),torch.eye(d,device=device))

    train_losses = []
    test_losses = []

    pbar = trange(n_epochs)

    for epoch in pbar:
        model.train()

        train_batch = []

        for n_batch, (data,_) in enumerate(train_loader):
            data = data.to(device)
            data = data.view(-1,28*28)
            z, log_det = model(data)

            l = loss(z[-1], log_det, base_distr)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_batch.append(l.item())

        train_losses.append(np.mean(train_batch))

        if epoch % 1 == 0:
            pbar.set_postfix_str(f"loss = {train_losses[-1]:.3f}")

        if plot_val and epoch % plot_interval==0:
            print(epoch,train_losses[-1])
            val_mnist(model, device)
    
    return train_losses
