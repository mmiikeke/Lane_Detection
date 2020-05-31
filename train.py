import torch
import visdom
from dataset import Generator
from model import HourglassModel
from options import opt
from utils.train_utils import make_ground_truth_point, make_ground_truth_instance
from torch.autograd import Variable
from loss import SGPNLoss
from torch.utils.data import DataLoader




def train():
    print("Training...")
    print("Initializing hyperparameters...")
    # vis = visdom.Visdom()
    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),  Y=torch.zeros((1,)).cpu(),
                           # opts=dict(xlabel='epoch', ylabel='Loss', title='Training Loss', legend=['Loss']))

    # Get dataset
    print("Get Dataset...")
    train_loader = Generator()

    model = HourglassModel()
    model = model.cuda(opt.cuda_devices)

    criterion = SGPNLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    step = 0

    print("Training loop...")
    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch+1}/{opt.epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.epochs}'))

        model.train()

        point_loss = 0.0 

        for inputs in train_set:
            print(inputs)
            

        for inputs, target_lanes, target_h, test_image in train_loader.Generate():
            real_batch_size=len(target_lanes)
            print(real_batch_size)

            # generate ground truth
            ground_truth_point, ground_binary = make_ground_truth_point(target_lanes, target_h)
            ground_truth_instance = make_ground_truth_instance(target_lanes,target_h)
            
            # convert numpy array to torch tensor
            ground_truth_point = torch.from_numpy(ground_truth_point).float()
            ground_truth_point = Variable(ground_truth_point).cuda(opt.cuda_devices)
            ground_truth_point.requires_grad=False

            ground_binary = torch.LongTensor(ground_binary.tolist()).cuda(opt.cuda_devices)
            ground_binary.requires_grad=False

            ground_truth_instance = torch.from_numpy(ground_truth_instance).float()
            ground_truth_instance = Variable(ground_truth_instance).cuda(opt.cuda_devices)
            ground_truth_instance.requires_grad=False

            inputs = torch.from_numpy(inputs).float() 
            inputs = Variable(inputs.cuda(opt.cuda_devices))

            optimizer.zero_grad()

            result = model(inputs)
            loss = criterion(result, ground_truth_point, ground_truth_instance, real_batch_size, epoch)

            point_loss.backward()
            optimizer.step()

            point_loss += loss.item() * inputs.size(0)

        training_loss = point_loss / real_batch_size

        print('training_loss: {training_loss}')




        


    




    


if __name__ == '__main__':
    train()