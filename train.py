import torch
import visdom
from dataset import Generator
from model import HourglassModel
from options import opt
from utils.train_utils import make_ground_truth_point, make_ground_truth_instance




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
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    step = 0

    print("Training loop...")
    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch+1}/{opt.epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.epochs}'))

        model.train()
        for inputs, target_lanes, target_h, test_image in train_set.Generate():
            real_batch_size=len(target_lanes)
            ground_truth_point, ground_truth_binary = make_ground_truth_point(target_lanes, target_h)
            ground_truth_instance = make_ground_truth_instance(target_lanes,target_h)
            break
        break

        


    




    


if __name__ == '__main__':
    train()