# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:43:53 2022

@author: alial
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
    
        self.Conv1 = nn.Conv2d(3, 8, 6, stride=3) # 150-6}/3=48  // 49 torch.Size([15, 8, 49, 49])
        self.relu1=nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(8)
        
        self.Conv2 = nn.Conv2d(8, 15, 3 , stride=2) # 48-4}/2= 22  //24 torch.Size([15, 15, 24, 24])
        self.relu2=nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(12)
       
               
        self.Conv3 = nn.Conv2d(15, 30, 2 , stride=2) # 22-2}/2=10  //torch.Size([15, 30, 12, 12])
        self.relu3=nn.ReLU() 
        
        self.Conv4 = nn.Conv2d(30, 50, 2 , stride=2) # 10-2}/2=4  //torch.Size([15, 50, 6, 6])
        self.relu4=nn.ReLU() 
       
        self.Dense1=nn.Linear(50*6*6,1200)
        self.Dense1Act=nn.ReLU()
        
        self.Dense2=nn.Linear(1200,600)
        self.Dense2Act=nn.ReLU() ##nn.LeakyReLU(0.01)
        
        self.Dense3=nn.Linear(600,300)
        self.Dense3Act=nn.ReLU()
        
        self.Dense4=nn.Linear(300,200)
        self.Dense4Act=nn.ReLU()
        
        self.linear_means = nn.Linear(200, latent_dim)
        self.linear_log_var = nn.Linear(200, latent_dim)
    
        # self.Dense1=nn.Linear(input_dim,120)
        # self.Dense1Act=nn.ReLU()
        
        # self.Dense2=nn.Linear(120,80)
        # self.Dense2Act=nn.Tanh()
        
        # self.Dense3=nn.Linear(80,40)
        # self.Dense3Act=nn.Tanh()
        
        # self.Dense4=nn.Linear(40,20)
        # self.Dense4Act=nn.Tanh()
        
        # self.linear_means = nn.Linear(20, latent_dim)
        # self.linear_log_var = nn.Linear(20, latent_dim)
    
    def forward(self, x):
        # define your feedforward pass
    
    
        # output=self.Dense1Act(self.Dense1(x))
        # output=self.Dense2Act(self.Dense2(output))
        # output=self.Dense3Act(self.Dense3(output))
        # output=self.Dense4Act(self.Dense4(output))
        
        
        output=self.Conv1(x)
        output=self.relu1(output)
        # output=self.bn1(output)
      
        output=self.Conv2(output)
        output=self.relu2(output)
        # output=self.bn2(output)
        
        output=self.Conv3(output)
        output=self.relu3(output)
        
        output=self.Conv4(output)
        output=self.relu4(output)
        
        output=output.view(-1,50*6*6) #out = out.view(out.size(0), -1)
  
        output= self.Dense1(output)
        output=self.Dense1Act(output)
        output=self.Dense2Act(self.Dense2(output))
        output=self.Dense3Act(self.Dense3(output))
        output=self.Dense4Act(self.Dense4(output))
        

        #% Reduce latent dimension to mu and std by a linear fully connected
        means = self.linear_means(output)
        log_vars = self.linear_log_var(output)

        
        return means, log_vars, output


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        
        #self.Dense=nn.Linear(latent_dim,24)
        
        self.Dense1=nn.Linear(latent_dim,400) #// torch.Size([15, 50])
        self.Dense1Act=nn.ReLU()
        
        self.Dense2=nn.Linear(400,1200)
        self.Dense2Act=nn.ReLU()

        
        self.Dense3=nn.Linear(1200,50*4*4) #// torch.Size([15, 50, 4, 4])
        self.Dense3Act=nn.ReLU()
        

# =============================================================================
#   ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0      
#   H out=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1 
#   H out=(Hin−1)×stride[0]+(kernel_size[0])     
# =============================================================================

        self.Conv4 = nn.ConvTranspose2d(50, 30, 2 , stride=2) # 4-1}*2+2 = 8  torch.Size([15, 30, 8, 8])
        self.relu4=nn.ReLU() 

        self.Conv3 = nn.ConvTranspose2d(30, 15, 3, stride=3) # 8-1)*3+3 = 24 torch.Size([15, 15, 24, 24])
        self.relu3=nn.ReLU() 

        self.Conv2 = nn.ConvTranspose2d(15, 8, 3 , stride=2) # 24-1)*2+ 3 = 49 torch.Size([15, 8, 55, 55])
        self.relu2=nn.Sigmoid()
        #self.bn2 = nn.BatchNorm1d(12)

        
        self.Conv1 = nn.ConvTranspose2d(8, 3, 6, stride=3) # 49-1)*3+ 6 = 150
        self.relu1=nn.Sigmoid()
        #self.bn1 = nn.BatchNorm1d(6)
        
        # self.Dense1=nn.Linear(latent_dim,20)
        # #%self.Dense1Act=nn.Tanh()
        
        # self.Dense2=nn.Linear(20,40)
        # self.Dense2Act=nn.Tanh()
        
        # self.Dense3=nn.Linear(40,80)
        # self.Dense3Act=nn.Tanh()
        
        # self.Dense4=nn.Linear(80,120)
        # self.Dense4Act=nn.ReLU()
        
        
        # self.Dense6=nn.Linear(120,output_dim)
        
        
    
    def forward(self, x):
        # define your feedforward pass
        
        # output=self.Dense1(x)
        # output=self.Dense2Act(self.Dense2(output))
        # output=self.Dense3Act(self.Dense3(output))
        # output=self.Dense4Act(self.Dense4(output))
        # # output=self.Dense5Act(self.Dense5(output))

        # output=self.Dense6(output)
        
        output=self.Dense1Act(self.Dense1(x))
        output=self.Dense2Act(self.Dense2(output))
        output=self.Dense3Act(self.Dense3(output))
  
        # output=self.Dense3(output)
        
        output=output.view(-1,50,4,4) #out = out.view(out.size(0), -1)
        
        output=self.Conv4(output)
        output=self.relu4(output)
        
        output=self.Conv3(output)
        output=self.relu3(output)
        # output=self.bn1(output)
      
        output=self.Conv2(output)
        output=self.relu2(output)
        # output=self.bn2(output)
        
        output=self.Conv1(output)
        output=self.relu1(output)
    
        return output


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)
    
    def forward(self, x):
        # define your feedforward pass
    
    
        mu, log_var, output =self.enc(x)
        
        #%%%  The data is reduced to latent vectors!!!
        
        # probability=EncodedData.view(-1, 2) #,latent_dim
        # ## get `mu` and `log_var`
        # mu = probability[:, 0, :] # the first feature values as mean
        # log_var = probability[:, 1, :] # the other feature values as variance
        
        #z  self.reparameterize(mu, log_var)
        
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        #end reparameterize to construct x data from z. 
        
        return sample , mu, log_var  # 16by16 , batch size and latent number
        
    
    def decode(self, z):
        # given random noise z, generate airfoils
        

        return self.dec(z)

