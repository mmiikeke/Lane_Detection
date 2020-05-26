import torch
import visdom
from dataset import Generator
from model import HourglassModel



def train():
    print("Training...")
    print("Initializing hyperparameters...")
    # vis = visdom.Visdom()
    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),  Y=torch.zeros((1,)).cpu(),
                           # opts=dict(xlabel='epoch', ylabel='Loss', title='Training Loss', legend=['Loss']))

    # Get dataset
    print("Get Dataset...")
    train_set = Generator()

    model = HourglassModel()

    




    


if __name__ == '__main__':
    train()