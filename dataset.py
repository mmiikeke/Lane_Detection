from options import opt
from pathlib import Path
import json
import cv2
import random
import copy
import numpy as np
import math
from torch.utils.data import Dataset


class LaneDataset(Dataset):
    def __init__(self, mode):
        self.x_size=512
        self.y_size=256
        self.mode = mode

        if self.mode == 'train':
            self.train_data_five = []
            self.train_data_four = []
            self.train_data_three = []
            self.train_data_two = []
            self.train_path = Path(opt.data_root).joinpath('train_set')

            with open(Path(self.train_path).joinpath('five.json')) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    jsonString = json.loads(line)
                    self.train_data_five.append(jsonString)
            
            with open(Path(self.train_path).joinpath('four.json')) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    jsonString = json.loads(line)
                    self.train_data_four.append(jsonString)

            with open(Path(self.train_path).joinpath('three.json')) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    jsonString = json.loads(line)
                    self.train_data_three.append(jsonString)
            
            with open(Path(self.train_path).joinpath('two.json')) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    jsonString = json.loads(line)
                    self.train_data_two.append(jsonString)

        elif self.mode == 'test':
            # load test set
            self.test_data = []
            self.test_path=Path(opt.data_root).joinpath('test_set')
            with open(Path(self.test_path).joinpath('test_tasks_0627.json')) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    jsonString = json.loads(line)
                    self.test_data.append(jsonString)

        self.train_data =  self.train_data_five + self.train_data_four + self.train_data_three + self.train_data_two

    def __len__(self):
        if self.mode == 'train':
            return(len(self.train_data_two) + len(self.train_data_three) + len(self.train_data_four))
        
        elif self.mode == 'test':
            len(self.test_data)

    def __getitem__(self, index):
        image_path =  Path(self.train_path).joinpath(self.train_data[index]['raw_file'])
        # print(image_path)
        return image_path




        



        



