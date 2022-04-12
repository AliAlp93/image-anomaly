# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:47:10 2022

@author: Alialp
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

from GanNetFor300by300 import Discriminator, Generator
# from utils import *
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
latent_dim=100;
noise_fn = lambda x: torch.rand((x, latent_dim)) # for random latent vector production , device='cuda:0'



# modelG = torch.load('Generator1.pt',map_location=torch.device('cpu'))
# modelD = torch.load('Discriminator1.pt',map_location=torch.device('cpu'))


transform = transforms.Compose([ 
transforms.Resize((300, 300)),
transforms.ToTensor(),
])

data_set_test = torchvision.datasets.ImageFolder(root='bottle/test',transform=transform) #,transform=transform

# dataset = AirfoilDataset()
# airfoil_x = dataset.get_x()
batch_sizeInput=8
test_dataloader = DataLoader(data_set_test, batch_size=batch_sizeInput, shuffle=True)
img_dim =300;
    
# test trained GAN model
# num_samples = 100

for n_batch, (local_batch, label) in enumerate(test_dataloader):
    
    
    latent_vec = noise_fn(batch_sizeInput)

    y_faulty = local_batch.to(device)
    faultyORnot=modelD.forward(y_faulty)
    
    fake_samples = modelG.forward(latent_vec)

    
    gridDis = make_grid(y_faulty,nrow = 4)
        # grid = make_grid([y_faulty[0], y_faulty[1], y_faulty[2], y_faulty[3]],nrow = 2)
    
    imgD = torchvision.transforms.ToPILImage()(gridDis)
    imgD.show()
    # plt.pause (20)
    print("Label Numbers: ",label.numpy())
    print("Label Numbers: ",faultyORnot.detach().numpy())
    plt.close()

    gridGen = make_grid(fake_samples,nrow = 4)

    imgG = torchvision.transforms.ToPILImage()(gridGen)
    imgG.show()
    # plt.pause (20)
    # print("Label Numbers: ",label.numpy())
    # print("Label Numbers: ",faultyORnot.detach().numpy())
    plt.close()



