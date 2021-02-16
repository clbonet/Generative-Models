import torch

import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from tqdm.auto import trange


def loss(h,log_det,distr,base_distr="normal"):
    if base_distr == "logistic":
        prior = distr.log_prob(h).sum(1).mean(0)
    else:
        prior = distr.log_prob(h).mean()
    return -(prior+log_det.mean())

def log_likelihood(h,log_det,distr,base_distr="normal"):
    if base_distr == "logistic":
        prior = distr.log_prob(h).sum(1)
    else:
        prior = distr.log_prob(h)
    return prior+log_det
    

def val_likelihood(model, distr, i, device, base_distr="normal"):
    model.eval()

    xline = torch.linspace(-4, 4, 100)
    yline = torch.linspace(-4, 4, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        xy, log_s = model(xyinput.to(device))
        zz = (log_likelihood(xy[-1],log_s,distr,base_distr)).exp().cpu()
        zgrid = zz.reshape(100,100)

        z = distr.sample((100,))
        xs = model.backward(z)
        x = xs[-1].detach()
        x = x.cpu().numpy()
        z = z.cpu().numpy()

    plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
    plt.colorbar()
    plt.scatter(x[:,0],x[:,1],c="red")
    plt.scatter(z[:,0],z[:,1],c="green")
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.title('iteration {}'.format(i + 1))
    plt.show()


def test_dataloader(model, test_loader, base_distr, device):
    ## Question: mean of means of batchs or mean on all sample?
    val_loss = []
    for n_batch, data in enumerate(test_loader):
        data = data.to(device)

        z, log_det = model(data)
        l = loss(z[-1], log_det, base_distr)
        val_loss.append(l.item())

    return np.mean(val_loss)



def train_dataloader(model, optimizer, train_loader, test_loader,
                    n_epochs=101, device=None, plot_val=True, plot_interval=5):
    d = 2

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

        for n_batch, data in enumerate(train_loader):
            data = data.to(device)

            z, log_det = model(data)

            l = loss(z[-1], log_det, base_distr)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_batch.append(l.item())

        train_losses.append(np.mean(train_batch))

        if epoch % 1 == 0:
            pbar.set_postfix_str(f"loss = {train_losses[-1]:.3f}")

        if plot_val and epoch % plot_interval == 0:
            print(epoch, test_dataloader(model,test_loader,base_distr,device)) # train_loss[-1])
            val_likelihood(model, base_distr, epoch, device)
    
    return train_losses

