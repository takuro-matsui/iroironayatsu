
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


#parser.add_argument('--input_channels', default=3, type=int, help='input channels')
#parser.add_argument('--deepsupervision', default=1, type=int)


parser.add_argument("--num_of_layers", type=int, default=20)
parser.add_argument("--features", type=int, default=64)
parser.add_argument("--rain_type", type=int, default=1)

opt = parser.parse_args()


print(opt)
from temp_models import *
from datasets import *


# デバイス指定
if torch.cuda.is_available():
  device = 'cuda'
  torch.backends.cudnn.benchmark=True
else:
  device = 'cpu'


# 作成して転送
#if 'unet' in opt.train_name:
if opt.train_name == 'unet':
    generator = GeneratorUNet().to(device)
elif opt.train_name == 'unet2' or opt.train_name == 'unet3':
    generator = GeneratorUNet2().to(device)
elif opt.train_name == 'unet4':
    generator = GeneratorUNet3().to(device)
elif opt.train_name == 'nesunet':
    generator = NestedUNet(args=opt).to(device)
#elif 'resnet' in opt.train_name:
elif opt.train_name == 'resnet2' or opt.train_name == "res_row2":
      generator = ResDerainNet2().to(device)
elif 'resnet' == opt.train_name: 
      generator = ResDerainNet().to(device)
elif opt.train_name == 'resnet3':
      generator = ResDerainNet2(channels=5).to(device)
elif 'resnet4' == opt.train_name:
      generator = ResDerainNet(channels=8).to(device)
elif 'resnet5' == opt.train_name:
      generator = ResDerainNet2(channels=8).to(device)
elif opt.train_name == 'resnet6' or opt.train_name =='resnet7':
      generator = ResDerainNet3(channels=5).to(device)
    #elif opt.train_name == 'resnet8':
    #  generator = GeneratorResNet(num_of_layers=opt.num_of_layers, features = opt.features).to(device)
#elif opt.train_name == 'res_row'
#  generator = GeneratorResNet(num_of_layers=opt.num_of_layers, features = opt.features).to(device)
else: 
  generator = GeneratorResNet(num_of_layers=opt.num_of_layers, features = opt.features).to(device)

# 検証用モード、重み計算なし
generator.eval()


# 検証に使う画像の読み込み

file_path = [["real_world/","dehazed/", "Practical/","test_nature/"],
              ["BSD100/","Urban100/","Rain12/", "synthetic/","matsui_noise/","test_syn/"]
            ]



print("test_data: "+file_path[opt.synthetic][opt.file_id])

image_dataset = TestImageDatasetSimple(rain_number=opt.rain_type, synthetic=opt.synthetic, file_path=file_path[opt.synthetic][opt.file_id])


# Generatorに重みを読み込み
if opt.train_name == 'resnet8' or opt.train_name == 'res_row' or opt.train_name == 'resnet81' or opt.train_name == 'res_row2':
  "saved_models/"+opt.train_name +"/generator_n"+str(opt.num_of_layers)+"_f"+str(opt.features)+"_"+str(opt.epoch)+".pth"
else:
  generator.load_state_dict(torch.load( "saved_models/"+opt.train_name+"/generator_"+str(opt.epoch)+".pth"))

# 画像の読み込み
imgs = image_dataset[opt.test_image]
rainy_image = imgs["rainy_image"].to(device)
#sobel_x = imgs["sobel_x"].to(device)
#sobel_y = imgs["sobel_y"].to(device)
#ycbcr = imgs["ycbcr"].to(device)
if opt.synthetic:
  B = imgs["ground_truth"].to(device)


_,first_h,first_w = rainy_image.size()
rainy_image = torch.nn.functional.pad(rainy_image,(0,(first_w//16)*16+16-first_w,0,(first_h//16)*16+16-first_h),"constant")
#sobel_x = torch.nn.functional.pad(sobel_x,(0,(first_w//16)*16+16-first_w,0,(first_h//16)*16+16-first_h),"constant")
#sobel_y = torch.nn.functional.pad(sobel_y,(0,(first_w//16)*16+16-first_w,0,(first_h//16)*16+16-first_h),"constant")
#ycbcr = torch.nn.functional.pad(ycbcr,(0,(first_w//16)*16+16-first_w,0,(first_h//16)*16+16-first_h),"constant")


if opt.synthetic:
  B = torch.nn.functional.pad(B,(0,(B.size(2)//16)*16+16-B.size(2),0,(B.size(1)//16)*16+16-B.size(1)),"constant")

# 画像の次元変換            
rainy_image = rainy_image.view(1,3,rainy_image.size(1),rainy_image.size(2))
#sobel_x = sobel_x.view(1,1,rainy_image.size(2),rainy_image.size(3))
#sobel_y = sobel_y.view(1,1,rainy_image.size(2),rainy_image.size(3))
#ycbcr = ycbcr.view(1,3,rainy_image.size(2),rainy_image.size(3))
if opt.synthetic:
  B = B.view(1,3,B.size(1),B.size(2))


with torch.no_grad(): #おまじない
  if opt.train_name == 'resnet'  or opt.train_name=='unet' or opt.train_name=='unet2' or opt.train_name=="res_row2":
    output  = generator(rainy_image)
  #elif opt.train_name == 'unet3' or opt.train_name == 'resnet2' or opt.train_name=='resnet8' or opt.train_name=='res_row':
  #  output  = rainy_image - generator(rainy_image)
  elif opt.train_name == 'resnet3' or opt.train_name == 'resnet6' or opt.train_name == 'resnet7':
    output = rainy_image - generator(torch.cat([rainy_image, sobel_x, sobel_y], 1))
  elif opt.train_name == 'resnet4':
    output = generator(torch.cat([rainy_image, sobel_x, sobel_y, ycbcr], 1))
  elif opt.train_name == 'resnet5':
    output = rainy_image - generator(torch.cat([rainy_image, sobel_x, sobel_y, ycbcr], 1))
  elif opt.train_name == 'nesunet':
    B_hat1, B_hat2, B_hat3, B_hat4 = generator(rainy_image) 
    output = (1*B_hat1 + 2*B_hat2 + 3*B_hat3 + 4*B_hat4)/10
  else:
    print('*')
    output  = rainy_image - generator(rainy_image)



# 出力画像の形式を整える
#output = np.clip(convert_to_numpy(output,first_h,first_w),0,1)
output = convert_to_numpy(output,first_h,first_w)
rainy_image = convert_to_numpy(rainy_image,first_h,first_w)
#ycbcr = convert_to_numpy(ycbcr,first_h,first_w)

#sobel_x = convert_to_numpy(sobel_x,first_h,first_w)
#sobel_y = convert_to_numpy(sobel_y,first_h,first_w)

if opt.synthetic:
  B = convert_to_numpy(B,first_h,first_w)



if opt.synthetic:
  psnr_rain = compare_psnr(B, rainy_image)
  ssim_rain =  compare_ssim(B, rainy_image, multichannel=True)
  psnr = compare_psnr(B, output)
  ssim = compare_ssim(B, output, multichannel=True)

  print('PSNR: %f  -----> %f' % (psnr_rain, psnr))
  print('SSIM: %f  -----> %f' % (ssim_rain, ssim))


# 画像の表示
if opt.test_image > -1 and opt.imshow > 0:
    plt.figure(1).clear()
    if opt.synthetic:
      plt.imshow(np.concatenate( [rainy_image, output, B, rainy_image-output],1 ))
    else:
      plt.imshow(np.concatenate( [rainy_image, output, rainy_image-output],1 ))
    
      

    plt.title(opt.train_name + " / " + str(opt.epoch) + "epochs")
    plt.show()



    #plt.figure(2).clear()
    #plt.imshow(np.concatenate([ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2], sobel_x, sobel_y],  1))
    #plt.show()