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
from models import *



parser = argparse.ArgumentParser()
#parser.add_argument("--dataset_type", type=int, default=0, help="dataset style 0:synthetic 1:real")
#parser.add_argument("--pretrain", type=int, default=0, help="pretrained data")
# 学習パラメータ
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=384, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=384, help="high res. image width")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
parser.add_argument("--loss_log_interval", type=int, default=100)
# 学習モデル
parser.add_argument("--train_name", type=str, default="test_train", help="training name")
parser.add_argument("--rain_type", type=int, default=1)
parser.add_argument("--num_of_layers", type=int, default=17)
parser.add_argument("--features", type=int, default=64)
opt = parser.parse_args()
print(opt)

# 保存済み（重み、loss、画像）の保存場所
os.makedirs("saved_models/"+opt.train_name, exist_ok=True)

weights_dir = "saved_models/"+opt.train_name+"/weights"
loss_dir = "saved_models/"+opt.train_name+"/loss"
images_dir = "saved_models/"+opt.train_name+"/images"

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

args_save("saved_models/"+opt.train_name+"opt_data.csv", opt)
# logをTensorboardに書き出す
writer = SummaryWriter(log_dir=loss_dir)


# GPUかCPUか
if torch.cuda.is_available():
  device = 'cuda'
  torch.backends.cudnn.benchmark=True
else:
  device = 'cpu'

hr_shape = (opt.hr_height, opt.hr_width)

# ネットを呼び出して、作成
generator = ResNet(channels=3,num_of_layers=opt.num_of_layers,features=opt.features)
#generator = UNet(residual=0)


# ロスを呼び出して、作成

criterion_mse = torch.nn.MSELoss()

# 使用するデバイスに転送
generator = generator.to(device)



criterion_mse = criterion_mse.to(device)


# Load pretrained models
#if opt.pretrain > 0:
#    generator.load_state_dict(torch.load( "saved_models/"+opt.train_name +"/#generator_"+str(opt.pretrain)+".pth"))

# 最適化手法を呼び出して、作成
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



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
    generator.load_state_dict(torch.load( weights_dir +"/generator_"+str(opt.epoch)+".pth"))

#Rmap_threshold = torch.tensor(0.6, device=device, requires_grad=False)

# ----------
#  Training
# ----------
for epoch in range(opt.epoch, opt.n_epochs): #エポック
    for i, imgs in enumerate(dataloader):    #エポック内でどの画像を使うか
        
        batches_done = epoch * len(dataloader) + i

        # 学習に使う画像の取り出し
        rainy_image = imgs["rainy_image"].to(device) #入力
        B = imgs["ground_truth"].to(device)   #正解


        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # generatorに画像を通す
        B_hat = generator(rainy_image)
        

        # ロスの計算 (MSE)
        loss_l2 = criterion_mse(B_hat, B)




        # ロスを合計する 重み付けが大事
        loss_G = loss_l2
        
        # 計算したロスを使ってGeneratorの重みを再計算
        loss_G.backward()
        # 再計算された重みを適用
        optimizer_G.step()


      

        # --------------
        #  Log Progress
        # --------------
        print(
            "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader),  loss_G.item())
        )
        # lossを保存
        if batches_done % opt.loss_log_interval == 0:
            writer.add_scalar('Generator/loss_G(l2)', loss_G.item(), epoch*len(dataloader)+i)

        # 学習中の画像を保存
        if batches_done % opt.sample_interval == 0:
            img_grid = gridImage((rainy_image, B_hat, B, rainy_image-B_hat, rainy_image-B))
            save_image(img_grid, images_dir + "/" + str(batches_done) + ".png", normalize=False)

    # 重みを保存
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), weights_dir +"/generator_"+str(epoch)+".pth")
        

# writer
writer.close()

