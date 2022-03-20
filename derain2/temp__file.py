import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
opt = parser.parse_args()



def args_save(filename, opt):
    dir = vars(opt)
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        if ~os.path.isfile(filename):
            writer.writerow(dir.keys())
        writer.writerow(dir.values())

def save_results(filename, opt, result=[-1,-1],name=["PSNR", "SSIM"]):
    dir = vars(opt)
    title = name+dir.keys()
    values = result+dir.values()
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        if os.path.isfile(filename):
            writer.writerow(values)
        else:
            writer.writerow(title)
            writer.writerow(values)

#args_save("matsui.csv", opt)
save_results("results.csv", opt, [25.2,0.92])
