import argparse
from pathlib import Path


ROOTPATH = "/home/joycenerd/Lane_Detection"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=Path(ROOTPATH).joinpath('Data'))
parser.add_argument('--cuda_devices', type=int, default=0, help='gpu device')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--grid_x', type=int, default=64, help='size of grid x')
parser.add_argument('--grid_y', type=int, default=32, help='size of grid y')
parser.add_argument('--resize_ratio', type=int, default=8, help='resize ratio')
opt=parser.parse_args()

