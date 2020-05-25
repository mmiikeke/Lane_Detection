import argparse
from pathlib import Path


ROOTPATH = "/mnt/md0/new-home/joycenerd/Lane_Detection"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=Path(ROOTPATH).joinpath('Data'))
opt=parser.parse_args()

