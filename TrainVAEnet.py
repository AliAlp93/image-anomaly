# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:38:53 2022

@author: alial
"""

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from Vae150by150 import VAE
from torch.utils.data import Dataset, DataLoader

# Root directory for dataset
# dataroot = "data/celeba"
# Number of workers for dataloader
workers = 0

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 400
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 200
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lrG= 0.01
lrD= 0.00002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.8
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
batchSize=15
niter=50
outf='DCVAEoutput'

transform = transforms.Compose([ 
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])
    
    # define dataset and dataloader
data_set_train = torchvision.datasets.ImageFolder(root='bottle/train',transform=transform) #
traingood_dataloader = DataLoader(data_set_train, batch_size=batchSize, shuffle=True)



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# ngpu = int(opt.ngpu)
# nz = int(opt.nz)
# ngf = int(opt.ngf)
# ndf = int(opt.ndf)

latent_dim = 100 # please do not change latent dimension
lr = 0.00002     # learning rate
num_epochs = 600
beta1=0.75

KL_Loss=[]
Sim_Loss=[]
TotalLoss=[]
   


# build the model
vae = VAE( latent_dim=latent_dim).to(device)
print("VAE model:\n", vae)

# define your loss function here
criterion = nn.MSELoss()   #BCELoss(reduction='sum') MSELoss
# loss = ?
fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

# define optimizer for discriminator and generator separately
optim = optim.Adam(vae.parameters(), lr=lr,betas=(beta1, 0.999))

for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(traingood_dataloader):
            y_real = local_batch.to(device)
            
            z_dist_toDecode, mu, log_var = vae.forward(y_real)
            trainigConstructX=vae.decode(z_dist_toDecode)
            #KL-Divergence.
            #KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())#/y_real.size(0) # 0.5 is incorporated as alpha in front of KL divergence
            similarityLoss = criterion(trainigConstructX.to(device), y_real)
            alpha=0.0001
            loss = (similarityLoss + alpha* KLD) ##/y_real.size(0)
            # loss=similarityLoss
    
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if (n_batch ) % 13 == 0:
                print("Epoch: [{}/{}], Batch: {}, loss: {}, fromKL: {}, fromSimilarity: {},".format(
                    epoch, num_epochs, n_batch, loss.item(),KLD.item(),similarityLoss.item()))
                KL_Loss.append(KLD.item()*alpha)
                Sim_Loss.append(similarityLoss.item())
                TotalLoss.append(loss.item())
            
                if (epoch) % 33 == 0:
                    plt.figure(6)
                    plt.plot(range(len(TotalLoss)), TotalLoss, color='red', marker='o')
                    plt.plot(range(len(Sim_Loss)), Sim_Loss, color='green', marker='o')
                    plt.plot(range(len(KL_Loss)), KL_Loss, color='black', marker='o')
                
                    # plt.plot(range(len(Disc_FakeLoss)), Disc_FakeLoss, color='blue', marker='o')
                
                    plt.title('Different Losses', fontsize=14)
                    plt.xlabel('Epoch', fontsize=14)
                    plt.ylabel('Loss', fontsize=14)
                    plt.grid(True)
                    plt.legend(["Total", "Similarity" ,"KL Div."], loc ="lower right")
                    plt.show()                    
                    plt.close()
                    
                    num_samples=8
                    noise = torch.randn((num_samples, latent_dim)).to(device)   # create random noise 
                    generated = vae.decode(noise)
                    vutils.save_image(generated.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)
                    torch.save(vae, '%s/netEn_epoch_%d.pth' % (outf, epoch))
                    # torch.save(vae.decode.state_dict(), '%s/netDe_epoch_%d.pth' % (outf, epoch))
                    
PATH = './Vae.pt'
    
torch.save(VAE, PATH)