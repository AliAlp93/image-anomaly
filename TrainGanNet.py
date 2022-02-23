# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:56:31 2022

@author: alialp
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, utils

# from dataset import AirfoilDataset #can do a similar dataset object...
from GanNet import Discriminator, Generator
from utils import *
from matplotlib import pyplot as plt

#%%
def main():
    #%%
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
      # hyperparameters
    latent_dim = 50 # please do not change latent dimension
    lr_dis = 0.0002 # discriminator learning rate
    lr_gen = 0.0003 # generator learning rate
    num_epochs = 100
    beta1 = 0.5 # beta1 value for Adam optimizer
    batch_sizeInput= 15 # Determine Batch Size
    
    
    noise_fn = lambda x: torch.rand((x, latent_dim), device='cuda:0') # for random latent vector production
   

    
    ##DO WE WANT TO TRANSFORM THE DATA??
    transform = transforms.Compose([ 
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])
    
    # define dataset and dataloader
    data_set_train = torchvision.datasets.ImageFolder(root='bottle/train',transform=transform) #
        
# =============================================================================
#     normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
# 
#     transform = transforms.Compose([ 
#     transforms.Resize((128, 128)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
# normalize,
# ])
    
# define custom transform function
#     transform = transforms.Compose([
#     transforms.ToTensor()
#     ])
#         # Python code to calculate mean and std
#     # of image
#       
#     # get tensor image
#     img_tr = transform(data_set_train[40][0])
#       
#     # calculate mean and std
#     mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
#       
#     # print mean and std
#     print("mean and std before normalize:")
#     print("Mean of the image:", mean)
#     print("Std of the image:", std)
# =============================================================================
    
    
    data_set_test = torchvision.datasets.ImageFolder(root='bottle/test',transform=transform) #,transform=transform

    # dataset = AirfoilDataset()
    # airfoil_x = dataset.get_x()
    
    traingood_dataloader = DataLoader(data_set_train, batch_size=batch_sizeInput, shuffle=True)
    img_dim =300;
    
    target_ones = torch.ones((batch_sizeInput, 1)).to(device)
    target_zeros = torch.zeros((batch_sizeInput, 1)).to(device)

    Disc_RealLoss=[]
    Disc_FakeLoss=[]
    TotalDiscriminatorLoss=[]
    GeneratorLoss=[]
    
    # build the model
    dis = Discriminator(input_dim=img_dim).to(device)
    gen = Generator(latent_dim=latent_dim, img_dim=img_dim).to(device)
    print("Distrminator model:\n", dis)
    print("Generator model:\n", gen)

    # define your GAN loss function here
    # you may need to define your own GAN loss function/class
    criterion = nn.BCELoss()

    # loss = ?

    # define optimizer for discriminator and generator separately
    optim_dis = Adam(dis.parameters(), lr=lr_dis,betas=(beta1, 0.999))
    optim_gen = Adam(gen.parameters(), lr=lr_gen,betas=(beta1, 0.999))
    
    # train the GAN model
    for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(traingood_dataloader):
            y_real = local_batch.to(device)

            # train discriminator

            # calculate customized GAN loss for discriminator
            # enc_loss = loss(...)
            
            #Start with Real Samples
            Pred_real= dis.forward(y_real)
            #loss_dis= criterion(Pred_real, target_ones)
                      
            loss_real = criterion(Pred_real, target_ones) #The overarching objective is to fool discriminator into labeling all the fake ones as real...
             #Discriminator should be able to recognize real samples as real in the beginning. 
            
            #Continue with Fake Generated Samples
            latent_vec = noise_fn(batch_sizeInput)
            
            with torch.no_grad():
                 fake_samples = gen.forward(latent_vec)
            pred_fake = dis.forward(fake_samples)
            loss_fake = criterion(pred_fake, target_zeros) ## A good working Discriminator should distinguish fake as a zero in the very beginning, After full training it is going to be %50 confidence, flip a coin as fake is similar to real

             # combine
            loss_dis = (loss_real + loss_fake) /2
            
            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()

            # train generator
            if(n_batch+1) % 2 == 1:
                latent_vec = noise_fn(batch_sizeInput)
                generated=gen.forward(latent_vec)
                classify=dis.forward(generated)
                loss_gen=criterion(classify,target_ones) #Aim of the generator is to get 1s from Discriminator for generated results.
    
                
                # calculate customized GAN loss for generator
                # enc_loss = loss(...)
                
                
    
                optim_gen.zero_grad()
                loss_gen.backward()
                optim_gen.step()

            # print loss while training
            if (n_batch ) % 91 == 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}, Disc Fake loss:{} , Disc real Loss:{}".format(
                    epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item(),loss_real.item(),loss_fake.item()))
                Disc_RealLoss.append(loss_real.item())
                Disc_FakeLoss.append(loss_fake.item())
                TotalDiscriminatorLoss.append(loss_dis.item())
                GeneratorLoss.append(loss_gen.item())
                
#%%
# =============================================================================
#     # test trained GAN model
#     num_samples = 100
#     
#     real_airfoils = dataset.get_y()[num_samples:num_samples*2]
#     
#     # create random noise 
#     noise = torch.randn((num_samples, latent_dim)).to(device)
#     # generate airfoils
#     gen_airfoils = gen(noise)
#     if 'cuda' in device:
#         gen_airfoils = gen_airfoils.detach().cpu().numpy()
#     else:
#         gen_airfoils = gen_airfoils.detach().numpy()
# 
#     # plot generated airfoils
#     plot_airfoils(airfoil_x, gen_airfoils)
#     plot_airfoils(airfoil_x, real_airfoils)
#     
#     import matplotlib.pyplot as plt
#     
#     plt.figure(6)
# 
#     plt.plot(range(len(TotalDiscriminatorLoss)), TotalDiscriminatorLoss, color='red', marker='o')
#     plt.plot(range(len(GeneratorLoss)), GeneratorLoss, color='green', marker='o')
#     plt.plot(range(len(Disc_RealLoss)), Disc_RealLoss, color='black', marker='o')
# 
#     plt.plot(range(len(Disc_FakeLoss)), Disc_FakeLoss, color='blue', marker='o')
# 
#     plt.title('Different Losses', fontsize=14)
#     plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Loss', fontsize=14)
#     plt.grid(True)
#     plt.legend(["Total_Discriminator", "Generator" ,"Real Discri","Fake Discri"], loc ="lower right")
#     plt.show()
# =============================================================================

#%%

    PATHD = './Discriminator.pt'
    PATHG = './Generator.pt'
    
    torch.save(dis, PATHD)
    
    torch.save(gen, PATHG)
#%%
if __name__ == "__main__":
    main()
