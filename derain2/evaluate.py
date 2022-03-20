
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import shutil
from skimage.measure import compare_ssim, compare_psnr
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Util.util import *
from Util.functions import *

def convert_to_numpy(input,H,W):
    if input.size(1) == 1:
      return  input[:,:,:H,:W].clone().cpu().numpy().reshape(H,W)#.transpose(1,2,0)
    else:
      return  input[:,:,:H,:W].clone().cpu().numpy().reshape(3,H,W).transpose(1,2,0)

parser = argparse.ArgumentParser()
parser.add_argument("--test_image", type=int, default=-1, help="test image number")
parser.add_argument("--epoch", type=int, default=-1, help="model epochs")
parser.add_argument("--train_name", type=str, default="test31", help="training name")
parser.add_argument("--file_id", type=int, default=1, help="0: create synthe, 1: real, 2: synthe")
parser.add_argument("--imshow", type=int, default=1, help="show result")
parser.add_argument("--synthetic", type=int, default=1, help="show result")

parser.add_argument("--save_image", type=int, default=0, help="save result image")

parser.add_argument("--save_result", type=int, default=0, help="save psnr/ssim/args")


#parser.add_argument('--input_channels', default=3, type=int, help='input channels')
#parser.add_argument('--deepsupervision', default=1, type=int)


parser.add_argument("--num_of_layers", type=int, default=20)
parser.add_argument("--features", type=int, default=64)
parser.add_argument("--rain_type", type=int, default=1)

opt = parser.parse_args()


print(opt)
from models import *
from datasets import *


# デバイス指定
if torch.cuda.is_available():
  device = 'cuda'
  torch.backends.cudnn.benchmark=True
else:
  device = 'cpu'


# 作成して転送

generator = ResNet().to(device)

# 検証用モード、重み計算なし
generator.eval()


# 検証に使う画像の読み込み

file_path = [["real_world/","dehazed/", "Practical/","test_nature/"],
              ["BSD100/","Urban100/","Rain12/", "synthetic/","matsui_noise/","test_syn/"]
            ]



print("test_data: "+file_path[opt.synthetic][opt.file_id])

image_dataset = TestImageDatasetSimple(rain_number=opt.rain_type, synthetic=opt.synthetic, file_path=file_path[opt.synthetic][opt.file_id])


# Generatorに重みを読み込み
generator.load_state_dict(torch.load("saved_models/"+opt.train_name+"/weights/generator_"+str(opt.epoch)+".pth"))

# 画像の読み込み
imgs = image_dataset[opt.test_image]
rainy_image = imgs["rainy_image"].to(device)

if opt.synthetic:
  B = imgs["ground_truth"].to(device)


_,first_h,first_w = rainy_image.size()
#rainy_image = torch.nn.functional.pad(rainy_image,(0,(first_w//16)*16+16-first_w,0,(first_h//16)*16+16-first_h),"constant")



#if opt.synthetic:
#  B = torch.nn.functional.pad(B,(0,(B.size(2)//16)*16+16-B.size(2),0,(B.size(1)//16)*16+16-B.size(1)),"constant")

# 画像の次元変換
rainy_image = rainy_image.view(1,3,rainy_image.size(1),rainy_image.size(2))
if opt.synthetic:
  B = B.view(1,3,B.size(1),B.size(2))


with torch.no_grad(): #おまじない
  output  = generator(rainy_image)
output_for_save = output.clone()

save_dir = "results/"+file_path[opt.synthetic][opt.file_id]+opt.train_name+"/"
if opt.save_image:
  save_image(output_for_save, save_dir+str(opt.test_image)+".png")


#print(output.size())
# 出力画像の形式を整える
output = convert_to_numpy(output,first_h,first_w)
rainy_image = convert_to_numpy(rainy_image,first_h,first_w)

#print(output.shape)


if opt.synthetic:
  B = convert_to_numpy(B,first_h,first_w)



if opt.synthetic:
  psnr_rain = compare_psnr(B, rainy_image)
  ssim_rain =  compare_ssim(B, rainy_image, multichannel=True)
  psnr = compare_psnr(B, output)
  ssim = compare_ssim(B, output, multichannel=True)

  print('PSNR: %f  -----> %f' % (psnr_rain, psnr))
  print('SSIM: %f  -----> %f' % (ssim_rain, ssim))
#print(output*255)

# 画像の表示
if opt.test_image > -1 and opt.imshow > 0:
    plt.figure(1).clear()
    if opt.synthetic:
      plt.imshow(np.concatenate( [rainy_image, output, B, rainy_image-output],1 ))
      im_title = "/[rainy]/[derained]/[groundtruth]/[residual]/"
    else:
      plt.imshow(np.concatenate( [rainy_image, output, rainy_image-output],1 ))
      im_title = "/[rainy]/[derained]/[residual]/"

    plt.title(opt.train_name + " / " + str(opt.epoch) + "epochs" + im_title)
    plt.show()


#if opt.save_image:
    #pil_img = Image.fromarray((output*255).astype(np.uint8))
    #os.makedirs(save_dir, exist_ok=True)
    #pil_img.save(save_dir+str(opt.test_image)+".png")

if opt.save_result:
  if opt.synthetic:
    save_results(save_dir+"log.csv", opt, result=[psnr_rain, psnr, ssim_rain, ssim], name=["psnr_rain", "psnr", "ssim_rain", "ssim"])
  else:
    save_results(save_dir+"result_log.csv", opt)