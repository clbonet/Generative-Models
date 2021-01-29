import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm.auto import trange



criterion = nn.BCELoss(reduction='sum')

def vae_loss(x, y, mu, log_sigma, log_det):    
    reconstruction_loss = criterion(y, x)
    kl_loss = -0.5*torch.sum(mu.pow(2)+log_sigma.exp()-log_sigma-1)
    
    return reconstruction_loss-kl_loss-log_det.sum()


def plot_val(model, test_loader, device):
    for x_val, _ in test_loader:
        fig,ax = plt.subplots(1,2,figsize=(10,10))

        ax[0].imshow(x_val[0][0],"gray")

        x_val = x_val.to(device)

        model.eval()
        yhat, mu, sigma, log_det = model(x_val[0][0].reshape(-1,28,28))
        yhat = yhat.reshape(-1,1,28,28)
        ax[1].imshow(yhat[0][0].cpu().detach().numpy(),"gray")
        plt.show()
        break
        

def loss_val(model, test_loader, device):
    loss_val_epoch = 0
    for cpt_batch, (x_val, _) in enumerate(test_loader):
        x_val = x_val.to(device)

        model.eval()
        yhat, mu, log_sigma, log_det = model(x_val.reshape(-1,28,28))
        yhat = yhat.reshape(-1,1,28,28)

        val_l = vae_loss(x_val,yhat,mu,log_sigma,log_det)
        loss_val_epoch += val_l.item()/x_val.size(0)
    
    return loss_val_epoch/(cpt_batch+1)


def train_vae(model, optimizer, train_loader, test_loader=None, n_epochs = 201, device = None):
    train_losses = []
    val_losses = []
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = trange(n_epochs)

    for epoch in pbar:   
        loss_epoch = 0

        for cpt_batch, (x_batch, _) in enumerate(train_loader):
            x_batch = x_batch.to(device)

            model.train()

            yhat, mu, log_sigma, log_det = model(x_batch.reshape(-1,28,28))
            yhat = yhat.reshape(-1,1,28,28)

            l = vae_loss(x_batch, yhat, mu, log_sigma, log_det)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_epoch += l.item()/x_batch.size(0) ## Mean

        train_losses.append(loss_epoch/(cpt_batch+1))

        with torch.no_grad():
            loss_val_epoch = loss_val(model, test_loader, device)
            val_losses.append(loss_val_epoch)

        pbar.set_postfix_str(f"train_loss = {train_losses[-1]:.3f}, test_loss = {val_losses[-1]:.3f}")

        if test_loader is not None and epoch % 50 == 0:
            print(epoch, "train_loss:", train_losses[-1], "val_loss:", val_losses[-1])
            plot_val(model, test_loader, device)
    
    return train_losses, val_losses
        