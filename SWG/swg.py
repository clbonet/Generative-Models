## Code inspired from Taken from https://github.com/maremun/swg/blob/master/swg.ipynb

import argparse
import torch
import torchvision
import math

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from tqdm import trange
from PIL import Image

from utils_swg.nn_cifar import *


parser = argparse.ArgumentParser()
parser.add_argument("--nprojs", type=int, default=500, help="Number of projections")
parser.add_argument("--nepochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--d_latent", type=int, default=100, help="Dimension latent space")
parser.add_argument("--method", type=str, default="rsot", help="Method, rsot or suot or sw")
parser.add_argument("--rho1", type=float, default=1, help="rho1")
parser.add_argument("--rho2", type=float, default=1, help="rho2")
parser.add_argument("--inner_iter", type=int, default=20, help="Number of inner iter of suot or rsot")
#parser.add_argument("--fraction_corrupt", type=float, default=0, help="Corrupted data")
parser.add_argument("--pbar", help="Plot pbar", action="store_true")
parser.add_argument("--use_disc", help="Use discriminator", action="store_true")
args = parser.parse_args()


imsize = 64
c = 3

transform=transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()
])


def resize_and_crop(img, size):
    img_w, img_h = img.size
    scale_factor = max((float(size[0]) / img_w), (float(size[1]) / img_h))
    re_w, re_h = int(math.ceil(img_w * scale_factor)), int(math.ceil(img_h * scale_factor))

    # Resize
    img = img.resize((re_w, re_h))

    # Crop
    if re_h == size[1]:
        start_w = (re_w - re_h) / 2
        end_w = start_w + size[0]
        start_h = 0
        end_h = start_h + size[1]
    else:
        start_w = 0
        end_w = start_w + size[0]
        start_h = (re_h - re_w) / 2
        end_h = start_h + size[1]

    img = img.crop((start_w, start_h, end_w, end_h))
    return img


def get_cifar10():
    
    train_dataset = torchvision.datasets.CIFAR10(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )
    
    #if args.fraction_corrupt > 0:
        #print("Check", args.fraction_corrupt, flush=True)
        
        #train_dataset_mnist = torchvision.datasets.MNIST(
            #root="~/torch_datasets", train=True, transform=transform, download=True
        #)
        
        #n_mnist = train_dataset_mnist.data.shape[0]
        #n_cifar = train_dataset.data.shape[0]

        #prop_outlier = args.fraction_corrupt

        #n_outliers = int(prop_outlier * n_cifar)
        #index_outliers = np.random.randint(0, n_mnist, n_outliers)

        #outliers = train_dataset_mnist.data[index_outliers].numpy()

        #outliers_3 = np.zeros((outliers.shape[0], 32, 32, 3), dtype=int)

        #for j, img in enumerate(outliers):
            #img_pil = Image.fromarray(img)
            #img_pil = resize_and_crop(img_pil, (32,32))
            #img = np.asarray(img_pil)

            #for i in range(3):
                #outliers_3[j,:,:,i] = img
                

        #indices = np.random.randint(0, n_cifar, n_outliers)
        #train_dataset.data[indices] = outliers_3

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    return train_loader, test_loader


def train_swg(device, train_loader, test_loader):
    x_dim = imsize*imsize*c
    d_latent = args.d_latent
    
    G = Generator(d_latent).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4)

    D = Discriminator(c).to(device)
    d_criterion = nn.BCEWithLogitsLoss()
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4)
    
    pbar = trange(args.nepochs)
    
    for epoch in pbar:
        for i, (batch_x, _) in enumerate(train_loader):
            G.train()

            # Generator step
            x = batch_x.to(device)
            z = torch.randn((batch_x.size(0), d_latent), device=device)
            
            xpred = G(z)
            
            if args.use_disc:
                _, fake_features = D(xpred) # get image features from Discriminator
                _, true_features = D(x)
                
            else:
                fake_features = xpred.reshape(batch_x.size(0), -1)
                true_features = x.reshape(batch_x.size(0), -1)
            
            
            if args.method == "sw":
                gloss = ot.sliced_wasserstein_distance(fake_features, true_features, n_projections=args.nprojs)
                
            g_opt.zero_grad()
            gloss.backward()
            g_opt.step()

            
            if args.use_disc:
                # Discriminator step
                z = torch.randn((batch_x.size(0), d_latent), device=device)
                xpred = G(z).detach()
                fake_score, _ = D(xpred)
                true_score, _ = D(x)

                dloss_fake = d_criterion(fake_score, torch.zeros_like(fake_score))
                dloss_true = d_criterion(true_score, torch.ones_like(true_score))
                dloss = dloss_fake.mean() + dloss_true.mean()

                d_opt.zero_grad()
                dloss.backward()
                d_opt.step()
                
                
    torch.save(G.state_dict(), "./results/swg_cifar_"+args.method+"_rho1"+str(args.rho1)+"_rho2"+str(args.rho2)+"_corrupt"+str(args.fraction_corrupt)+".model")




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, test_loader = get_cifar10()
    
    train_swg(device, train_loader, test_loader)
    
    
