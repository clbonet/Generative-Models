import torch

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from tqdm.auto import trange


def loss(h,log_det,base_distr):
    prior = base_distr.log_prob(h).mean()
    return -(prior+log_det.mean())

def log_likelihood(h,log_det,base_distr):
    prior = base_distr.log_prob(h)
    return prior+log_det
    

def val_moons(model, base_distr, i, device):
    model.eval()

    xline = torch.linspace(-1.5, 2.5, 100)
    yline = torch.linspace(-.75, 1.25, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        xy, log_s = model(xyinput.to(device))
        zz = (log_likelihood(xy[-1],log_s,base_distr)).exp().cpu()
        zgrid = zz.reshape(100,100)

        z = base_distr.sample((100,))
        xs, _ = model.backward(z)
        x = xs[-1].detach()
        x = x.cpu().numpy()
        z = z.cpu().numpy()

    plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
    plt.colorbar()
    plt.scatter(x[:,0],x[:,1],c="red")
    plt.scatter(z[:,0],z[:,1],c="green")
    plt.xlim(-1.5,2.5)
    plt.ylim(-0.75,1.25)
    plt.title('iteration {}'.format(i + 1))
    plt.show()    


def train_moons(model, optimizer, n_epochs=10001, base_distr="normal", 
				device=None, plot_val=True, plot_interval=1000):
    d = 2
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if base_distr == "normal":
        base_distr = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(d,device=device),torch.eye(d,device=device))
    elif base_distr == "logistic":
	    base_distr = TransformedDistribution(Uniform(torch.zeros(d, device=device),
           torch.ones(d, device=device)), SigmoidTransform().inv)
    else:
        raise ValueError("wrong base distribution")

    train_loss = []
    
    pbar = trange(n_epochs)

    for i in pbar: #range(n_epochs):        
        x, y = datasets.make_moons(128, noise=.1)
        x = torch.tensor(x, dtype=torch.float32).to(device)

        model.train()

        z, log_det = model(x)
        l = loss(z[-1],log_det,base_distr)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.append(l.item())
        
        if i%100==0:
            pbar.set_postfix_str(f"loss = {train_loss[-1]:.3f}")

        if plot_val and i % plot_interval == 0:
            print(i,train_loss[-1])
            val_moons(model, base_distr, i, device)
            
    return train_loss
