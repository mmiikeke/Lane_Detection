import torch
import visdom
from dataset import Generator



def train():
    print("Training...")
    print("Initializing hyperparameters...")
    # vis = visdom.Visdom()
    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),  Y=torch.zeros((1,)).cpu(),
                           # opts=dict(xlabel='epoch', ylabel='Loss', title='Training Loss', legend=['Loss']))

    print("Get Dataset...")
    train_loader = Generator()
    


if __name__ == '__main__':
    train()