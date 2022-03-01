# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 18:10:14 2022

@author: https://github.com/pytorch/examples/blob/master/dcgan/main.py
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
outf='PytorchDCGANoutput'

transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
    
    # define dataset and dataloader
data_set_train = torchvision.datasets.ImageFolder(root='bottle/train',transform=transform) #

dataloader = torch.utils.data.DataLoader(data_set_train, batch_size=batchSize,
                                         shuffle=True, num_workers=int(workers))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# ngpu = int(opt.ngpu)
# nz = int(opt.nz)
# ngf = int(opt.ngf)
# ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.03)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.03)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
# if netG != '':
#     netG.load_state_dict(torch.load(netG))
# print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
# if netD != '':
#     netD.load_state_dict(torch.load(netD))
# print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
# optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.8)
schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[10, 30], gamma=0.2) #LR will decay by a factor of 0.1 at 150 and 200 epoch
schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[10, 20,30], gamma=0.1) #LR will decay by a factor of 0.1 at 150 and 200 epoch

# if dry_run:
#     niter = 1
totalGlossinAnEpoch=0
totalDforRealAnEpoch=0
totalDforFakeAnEpoch=0
GlossEpochs=[]
D_RealEpochs=[]
D_FakeEpochs=[]

for epoch in range(niter):
    schedulerD.step()
    schedulerG.step()
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # if(i+1) % 2 == 1:

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        totalGlossinAnEpoch=totalGlossinAnEpoch+errG.item()
        totalDforRealAnEpoch=totalDforRealAnEpoch+errD_real.item() 
        totalDforFakeAnEpoch=totalDforFakeAnEpoch+errD_fake.item()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i == 13:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)
            
            GlossEpochs.append(totalGlossinAnEpoch)
            totalGlossinAnEpoch=0
            D_RealEpochs.append(totalDforRealAnEpoch)
            totalDforRealAnEpoch=0
            D_FakeEpochs.append(totalDforFakeAnEpoch)
            totalDforFakeAnEpoch=0
            
            if (epoch) % 10 == 1:
                plt.figure(6)
                # plt.plot(range(len(TotalDiscriminatorLoss)), TotalDiscriminatorLoss, color='red', marker='o')
                plt.plot(range(len(GlossEpochs)), GlossEpochs, color='green', marker='o')
                plt.plot(range(len(D_RealEpochs)), D_RealEpochs, color='red', marker='o')       
                plt.plot(range(len(D_FakeEpochs)), D_FakeEpochs, color='blue', marker='o')
            
                plt.title('Different Losses', fontsize=14)
                plt.xlabel('Epoch', fontsize=14)
                plt.ylabel('Loss', fontsize=14)
                plt.grid(True)
                plt.legend(["Generator" ,"Real Discri","Fake Discri"], loc ="lower right")
                plt.show() 
                plt.savefig("Total Loss at epochs.png")                   
                plt.close()

        # if dry_run:
        #     break
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
    
