# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:14:46 2022

@author: alial
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from Vae150by150 import VAE
# from utils import *
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
latent_dim=100;
noise_fn = lambda x: torch.rand((x, latent_dim)) # for random latent vector production , device='cuda:0'


modelVAE = torch.load('Vae.pt',map_location=torch.device('cpu'))
M_VAE = torch.load('DCVAEoutput/netEn_epoch_594.pth')


transform = transforms.Compose([ 
transforms.Resize((150, 150)),
transforms.ToTensor(),
])

data_set_test = torchvision.datasets.ImageFolder(root='bottle/test',transform=transform) #,transform=transform

# dataset = AirfoilDataset()
# airfoil_x = dataset.get_x()
batch_sizeInput=8
test_dataloader = DataLoader(data_set_test, batch_size=batch_sizeInput, shuffle=True)
img_dim =150;
    
# test trained GAN model
# num_samples = 100

for n_batch, (local_batch, label) in enumerate(test_dataloader):
    
    y_faulty = local_batch.to(device)
    
    z_dist_toDecode, mu, log_var = M_VAE.forward(y_faulty)
    trainigConstructX=M_VAE.decode(z_dist_toDecode)

    # z_dist_toDecode, mu, log_var = modelVAE.forward(y_faulty)
    # trainigConstructX=modelVAE.decode(z_dist_toDecode)
        


    
    gridDis = make_grid(y_faulty,nrow = 4)
        # grid = make_grid([y_faulty[0], y_faulty[1], y_faulty[2], y_faulty[3]],nrow = 2)
    
    imgD = torchvision.transforms.ToPILImage()(gridDis)
    imgD.show()
    plt.pause (20)
    print("Label Numbers: ",label.numpy())
    # print("Label Numbers: ",faultyORnot.detach().numpy())
    plt.close()

    GeneratedFromFaulty = make_grid(trainigConstructX,nrow = 4)

    imgG = torchvision.transforms.ToPILImage()(GeneratedFromFaulty)
    imgG.show()
    plt.pause (20)
    # print("Label Numbers: ",label.numpy())
    # print("Label Numbers: ",faultyORnot.detach().numpy())
    plt.close()
