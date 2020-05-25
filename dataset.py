from options import opt
from pathlib import Path
import json


class Generator(object):
    def __init__(self):
        self.train_data_five = []
        self.train_data_four = []
        self.train_data_three = []
        self.train_data_two = []
        train_path = Path(opt.data_root).joinpath('train_set')

        with open(Path(train_path).joinpath('five.json')) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_five.append(jsonString)
        
        with open(Path(train_path).joinpath('four.json')) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_four.append(jsonString)

        with open(Path(train_path).joinpath('three.json')) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_three.append(jsonString)
        
        with open(Path(train_path).joinpath('two.json')) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data_two.append(jsonString)

        self.size_train = len(self.train_data_two) + len(self.train_data_three) + len(self.train_data_four) + len(self.train_data_five)

        # load test set
        self.test_data = []
        test_path=Path(opt.data_root).joinpath('test_set/test_tasks_0627.json')
        with open(test_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.test_data.append(jsonString)

        self.size_test = len(self.test_data)