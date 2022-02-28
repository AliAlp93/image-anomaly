# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:55:57 2022

@author: alialp
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, 1)
        # since discriminator is a binary classifier
        
        # self.Dense1=nn.Linear(input_dim,150)
        # self.Dense1Act=nn.LeakyReLU()
# =============================================================================
#         Dim_out=(Dim_in+2*pad-dil*(ker-1)-1)/str
#  or     Dim_out=(Dim_in+2*pad-ker)/str
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# =============================================================================
        
        self.Conv1 = nn.Conv2d(3, 6, 5, stride=5) # 300+2*0-1*(5-1)-1}/5 = 59
        self.relu1=nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(6)
        
        self.Conv2 = nn.Conv2d(6, 12, 3 , stride=2) # 59-3}/2= 28 real60
        self.relu2=nn.ReLU()
        #self.bn2 = nn.BatchNorm1d(12)
       
               
        self.Conv3 = nn.Conv2d(12, 24, 2 , stride=2) # 28-2}/2=13 real14
        self.relu3=nn.ReLU() 
        
        self.Conv4 = nn.Conv2d(24, 32, 2 , stride=2) # 14-2}/2=6  real7
        self.relu4=nn.ReLU() 
       
        self.Dense1=nn.Linear(32*7*7,120)
        self.Dense1Act=nn.ReLU()
        
        self.Dense2=nn.Linear(120,64)
        self.Dense2Act=nn.ReLU() ##nn.LeakyReLU(0.01)
        
        self.Dense3=nn.Linear(64,16)
        self.Dense3Act=nn.ReLU()
        
        self.Dense4=nn.Linear(16,1)
        self.Dense4Act=nn.Sigmoid()
    
    def forward(self, x):
        # define your feedforward pass
      
        
      # inputforConv= x.unsqueeze(1) # Add new dimension at position 0

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
      
      output=output.view(-1,32*7*7) #out = out.view(out.size(0), -1)

      output= self.Dense1(output)
      output=self.Dense1Act(output)
      output=self.Dense2Act(self.Dense2(output))
      output=self.Dense3Act(self.Dense3(output))
      output=self.Dense4Act(self.Dense4(output))

      return output


class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super(Generator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, airfoil_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        
        self.Dense1=nn.Linear(latent_dim,64)
        self.Dense1Act=nn.ReLU()
        
        self.Dense2=nn.Linear(64,120)
        self.Dense2Act=nn.ReLU()

        
        self.Dense3=nn.Linear(120,32*7*7)
        self.Dense3Act=nn.ReLU()
        
        self.Conv4 = nn.ConvTranspose2d(32, 24, 2 , stride=2) # 7-1}*2+2=14 
        self.relu4=nn.ReLU() 
# =============================================================================
#   ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0      
#   H out=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1 
#   H out=(Hin−1)×stride[0]+(kernel_size[0])     
# =============================================================================
        self.Conv3 = nn.ConvTranspose2d(24, 12, 2 , stride=2) # 14-1)*2+2= 28
        self.relu3=nn.ReLU() 

        self.Conv2 = nn.ConvTranspose2d(12, 6, 6 , stride=2) # 28-1)*2+ 6 =60
        self.relu2=nn.ReLU()
        #self.bn2 = nn.BatchNorm1d(12)

        
        self.Conv1 = nn.ConvTranspose2d(6, 3, 5, stride=5) # 60-1)*5+ 5 = 300
        self.relu1=nn.ReLU()
        #self.bn1 = nn.BatchNorm1d(6)
        


    
    def forward(self, x):
        # define your feedforward pass
    
      output=self.Dense1Act(self.Dense1(x))
      output=self.Dense2Act(self.Dense2(output))
      output=self.Dense3Act(self.Dense3(output))

      # output=self.Dense3(output)
      
      output=output.view(-1,32,7,7) #out = out.view(out.size(0), -1)
      
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
