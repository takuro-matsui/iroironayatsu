import argparse
import os
import numpy as np
import math
import itertools
import sys
import shutil

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from Util.util import *
from temp_models import GeneratorResNet, Discriminator


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="VOC", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=384, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=384, help="high res. image width")
parser.add_argument("--channels", type=int, default=6, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--train_name", type=str, default="resnet8", help="training name")
parser.add_argument("--pretrain", type=int, default=0, help="pretrained data")
parser.add_argument("--loss_log_interval", type=int, default=100)

parser.add_argument("--num_of_layers", type=int, default=20)
parser.add_argument("--features", type=int, default=64)
parser.add_argument("--rain_type", type=int, default=1)

opt = parser.parse_args()
print(opt)

os.makedirs("images/"+opt.train_name , exist_ok=True)
os.makedirs("saved_models/"+opt.train_name , exist_ok=True)

log_name= "saved_models/"+opt.train_name
writer = SummaryWriter(log_dir=log_name)

# GPU???CPU???
if torch.cuda.is_available(): 
  device = 'cuda'
  torch.backends.cudnn.benchmark=True
else:
  device = 'cpu'

hr_shape = (opt.hr_height, opt.hr_width)

# ????????????????????????????????????
generator = GeneratorResNet(num_of_layers=opt.num_of_layers, features = opt.features)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))

# ?????????????????????????????????
criterion_GAN = torch.nn.MSELoss()
criterion_mse = torch.nn.MSELoss()

# ?????????????????????????????????
generator = generator.to(device)
discriminator = discriminator.to(device)

criterion_GAN = criterion_GAN.to(device)
criterion_mse = criterion_mse.to(device)


# Load pretrained models
if opt.pretrain > 0:
    generator.load_state_dict(torch.load( "saved_models/"+opt.train_name +"/generator_n"+str(opt.num_of_layers)+"_f"+str(opt.features)+"_"+str(opt.pretrain)+".pth"))

# ??????????????????????????????????????????
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# ????????????????????????????????????
from datasets import ImageDataset

# Pytorch?????????????????????????????????
dataloader = DataLoader(
    ImageDataset(hr_shape=hr_shape, rain_type=opt.rain_type),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8,
)

# Load pretrained models
if opt.epoch > 0:
    generator.load_state_dict(torch.load("saved_models/"+opt.train_name +"/generator_n"+str(opt.num_of_layers)+"_f"+str(opt.features)+"_"+str(opt.epoch)+".pth"))

#Rmap_threshold = torch.tensor(0.6, device=device, requires_grad=False)

# ----------
#  Training
# ----------
for epoch in range(opt.epoch, opt.n_epochs): #????????????
    for i, imgs in enumerate(dataloader):    #??????????????????????????????????????????
        
        batches_done = epoch * len(dataloader) + i

        # ????????????????????????????????????
        # rainy_image ---> B_hat
        rainy_image = imgs["rainy_image"].to(device) #??????
        ground_truth = imgs["ground_truth"].to(device)   #??????

        B = rainy_image - ground_truth

        

        # discriminator?????????????????????
        valid = torch.randn((rainy_image.size(0), *discriminator.output_shape), device = device, requires_grad = False) * 0.001 + 0.5 #??????
        fake = torch.randn((rainy_image.size(0), *discriminator.output_shape), device = device, requires_grad = False) * 0.001 - 0.5  #??????

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # generator??????????????????
        B_hat = generator(rainy_image)
        

        # ??????????????? (MSE)
        loss_l2 = criterion_mse(B_hat, B)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(torch.cat([rainy_image, rainy_image-B_hat], 1 )), valid)

        

        # ????????????????????? ?????????????????????
        loss_G = 1e-5 * loss_GAN + loss_l2
        
        # ??????????????????????????????Generator?????????????????????
        loss_G.backward()
        # ?????????????????????????????????
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # ??????????????? detatch?????????
        loss_real = criterion_GAN(discriminator(torch.cat([rainy_image, ground_truth], 1 )), valid)
        loss_fake = criterion_GAN(discriminator(torch.cat([rainy_image, rainy_image-B_hat], 1 ).detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        # ??????????????????????????????Discriminator?????????????????????
        loss_D.backward()
        # ?????????????????????????????????
        optimizer_D.step()


        # --------------
        #  Log Progress
        # --------------
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]  [G loss: %f] [l2: %f] [adv: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), loss_l2.item(), loss_GAN.item())
        )
        if batches_done % opt.loss_log_interval == 0:
            writer.add_scalar('Discriminator/loss_D', loss_D.item(), epoch*len(dataloader)+i)
            writer.add_scalar('Generator/loss_G', loss_G.item(), epoch*len(dataloader)+i)
            writer.add_scalar('Generator/loss_l2', loss_l2.item(), epoch*len(dataloader)+i)
            writer.add_scalar('Generator/loss_GAN', loss_GAN.item(), epoch*len(dataloader)+i)

        # ???????????????????????????
        if batches_done % opt.sample_interval == 0:
            img_grid = gridImage((rainy_image, B_hat, B, rainy_image-B_hat, rainy_image-B))
            save_image(img_grid, "images/" + opt.train_name + "/n"+str(opt.num_of_layers)+"_f"+str(opt.features)+"_" + str(batches_done) + ".png", normalize=False)

    # ???????????????
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        #print('saving')
        torch.save(generator.state_dict(), "saved_models/"+opt.train_name +"/generator_n"+str(opt.num_of_layers)+"_f"+str(opt.features)+"_"+str(epoch)+".pth")
        #torch.save(generator.state_dict(), "saved_models/"+opt.train_name +"/generator_n"+str(opt.num_of_layers)+"_f"+str(opt.features)+"_"+str(opt.epoch)+".pth")
    
    #writer.add_scalar('Discriminator_epoch/loss_D', loss_D.item(), epoch)
    #writer.add_scalar('Generator_epoch/loss_G', loss_G.item(), epoch)
    #writer.add_scalar('Generator_epoch/loss_l2', loss_l2.item(), epoch)
    #writer.add_scalar('Generator_epoch/loss_GAN', loss_GAN.item(), epoch)
writer.close()