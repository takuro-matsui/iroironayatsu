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
from temp_models import ResDerainNet, Discriminator


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="VOC", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=384, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=384, help="high res. image width")
parser.add_argument("--channels", type=int, default=6, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
parser.add_argument("--train_name", type=str, default="resnet", help="training name")
parser.add_argument("--dataset_type", type=int, default=0, help="dataset style 0:synthetic 1:real")
parser.add_argument("--pretrain", type=int, default=0, help="pretrained data")

parser.add_argument("--rain_type", type=int, default=1)
opt = parser.parse_args()
print(opt)

os.makedirs("images/"+opt.train_name , exist_ok=True)
os.makedirs("saved_models/"+opt.train_name , exist_ok=True)
log_name= "saved_models/"+opt.train_name
writer = SummaryWriter(log_dir=log_name)

# GPUかCPUか
if torch.cuda.is_available(): 
  device = 'cuda'
  torch.backends.cudnn.benchmark=True
else:
  device = 'cpu'

hr_shape = (opt.hr_height, opt.hr_width)

# ネットを呼び出して、作成
generator = ResDerainNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))

# ロスを呼び出して、作成
criterion_GAN = torch.nn.MSELoss()
criterion_mse = torch.nn.MSELoss()

# 使用するデバイスに転送
generator = generator.to(device)
discriminator = discriminator.to(device)

criterion_GAN = criterion_GAN.to(device)
criterion_mse = criterion_mse.to(device)


# Load pretrained models
if opt.pretrain > 0:
    generator.load_state_dict(torch.load( "saved_models/"+opt.train_name +"/generator_"+str(opt.pretrain)+".pth"))

# 最適化手法を呼び出して、作成
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# ここからデータセット作成
from datasets import ImageDataset

# Pytorchのデータセット作成関数
dataloader = DataLoader(
    ImageDataset(hr_shape=hr_shape, rain_type=opt.rain_type),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8,
)

# Load pretrained models
if opt.epoch > 0:
    generator.load_state_dict(torch.load( "saved_models/"+opt.train_name +"/generator_"+str(opt.epoch)+".pth"))

#Rmap_threshold = torch.tensor(0.6, device=device, requires_grad=False)

# ----------
#  Training
# ----------
for epoch in range(opt.epoch, opt.n_epochs): #エポック
    for i, imgs in enumerate(dataloader):    #エポック内でどの画像を使うか
        
        batches_done = epoch * len(dataloader) + i

        # 学習に使う画像の取り出し
        # rainy_image ---> B_hat
        rainy_image = imgs["rainy_image"].to(device) #入力
        B = imgs["ground_truth"].to(device)   #正解

        

        # discriminator用の正解ラベル
        valid = torch.randn((rainy_image.size(0), *discriminator.output_shape), device = device, requires_grad = False) * 0.1 + 0.5 #本物
        fake = torch.randn((rainy_image.size(0), *discriminator.output_shape), device = device, requires_grad = False) * 0.1 - 0.5  #偽物

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # generatorに画像を通す
        B_hat = generator(rainy_image)
        

        # ロスの計算 (MSE)
        loss_l2 = criterion_mse(B_hat, B)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(torch.cat([rainy_image, B_hat], 1 )), valid)

        

        # ロスを合計する 重み付けが大事
        loss_G = 1e-3 * loss_GAN + loss_l2
        
        # 計算したロスを使ってGeneratorの重みを再計算
        loss_G.backward()
        # 再計算された重みを適用
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # ロスを計算 detatchが重要
        loss_real = criterion_GAN(discriminator(torch.cat([rainy_image, rainy_image-B], 1 )), valid)
        loss_fake = criterion_GAN(discriminator(torch.cat([rainy_image, rainy_image-B_hat], 1 ).detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        # 計算したロスを使ってDiscriminatorの重みを再計算
        loss_D.backward()
        # 再計算された重みを適用
        optimizer_D.step()


        # --------------
        #  Log Progress
        # --------------
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]  [G loss: %f] [l2: %f] [adv: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), loss_l2.item(), loss_GAN.item())
        )
    
        writer.add_scalar('Discriminator/loss_D', loss_D.item(), epoch*len(dataloader)+i)
        writer.add_scalar('Generator/loss_G', loss_G.item(), epoch*len(dataloader)+i)
        writer.add_scalar('Generator/loss_l2', loss_l2.item(), epoch*len(dataloader)+i)
        writer.add_scalar('Generator/loss_GAN', loss_GAN.item(), epoch*len(dataloader)+i)    
        # 学習中の画像を保存
        if batches_done % opt.sample_interval == 0:
            img_grid = gridImage((rainy_image, B_hat, B, rainy_image-B_hat, rainy_image-B))
            save_image(img_grid, "images/" + opt.train_name + "/" + str(batches_done) + ".png", normalize=False)

    # 重みを保存
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/"+opt.train_name +"/generator_"+str(epoch)+".pth")

writer.close()
