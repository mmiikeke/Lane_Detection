import argparse
from pathlib import Path


ROOTPATH = "/mnt/md0/new-home/joycenerd/Lane_Detection"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=Path(ROOTPATH).joinpath('Data'))
parser.add_argument('--cuda_devices', type=int, default=0, help="gpu device")
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
opt=parser.parse_args()

