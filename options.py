import argparse
from pathlib import Path


ROOTPATH = "/mnt/md0/new-home/joycenerd/Lane_Detection"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=Path(ROOTPATH).joinpath('Data'))
parser.add_argument('--cuda_devices', type=int, default=1, help='gpu device')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--mode', type=str, default='train', help='[train, test]')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--grid_x', type=int, default=64, help='size of grid x')
parser.add_argument('--grid_y', type=int, default=32, help='size of grid y')
parser.add_argument('--resize_ratio', type=int, default=8, help='resize ratio')
parser.add_argument('--save_path', type=str, default=Path(ROOTPATH).joinpath('results'), help='save path for model.pth, test_resutl.json, and test.png')
opt=parser.parse_args()

