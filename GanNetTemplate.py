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
       
        self.Dense1=nn.Linear(50*6*6,120)
        self.Dense1Act=nn.ReLU()
        
        self.Dense2=nn.Linear(120,64)
        self.Dense2Act=nn.ReLU() ##nn.LeakyReLU(0.01)
        
        self.Dense3=nn.Linear(64,16)
        self.Dense3Act=nn.ReLU()
        
        self.Dense4=nn.Linear(16,1)
        self.Dense4Act=nn.ReLU()
    
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
      
      output=output.view(-1,50*6*6) #out = out.view(out.size(0), -1)

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
        
        self.Dense1=nn.Linear(latent_dim,120) #// torch.Size([15, 50])
        self.Dense1Act=nn.ReLU()
        
        self.Dense2=nn.Linear(120,200)
        self.Dense2Act=nn.ReLU()

        
        self.Dense3=nn.Linear(200,50*4*4) #// torch.Size([15, 50, 4, 4])
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
        self.relu2=nn.ReLU()
        #self.bn2 = nn.BatchNorm1d(12)

        
        self.Conv1 = nn.ConvTranspose2d(8, 3, 6, stride=3) # 49-1)*3+ 6 = 150
        self.relu1=nn.Sigmoid()
        #self.bn1 = nn.BatchNorm1d(6)
        


    
    def forward(self, x):
        # define your feedforward pass
    
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
