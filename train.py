import torch
import visdom
from dataset import Generator
from model import HourglassModel
from options import opt
from utils.train_utils import make_ground_truth_point, make_ground_truth_instance
from torch.autograd import Variable
from loss import SGPNLoss
from torch.utils.data import DataLoader
import tqdm
from evaluate import test, convert_to_original_size, find_target, write_result_json, save_result, LaneEval
import copy
import numpy as np
import cv2
from pathlib import Path


def train():
    print("Training...")
    print("Initializing hyperparameters...")
    # vis = visdom.Visdom()
    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),  Y=torch.zeros((1,)).cpu(),
                           # opts=dict(xlabel='epoch', ylabel='Loss', title='Training Loss', legend=['Loss']))

    # Get dataset
    print("Get Dataset...")
    data_loader = Generator()

    model = HourglassModel()
    model = model.cuda(opt.cuda_devices)

    criterion = SGPNLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    
    step = 0
    best_loss = float('inf')
    best_model_params = copy.deepcopy(model.state_dict())
    loss_list=[]
    record=open('record.txt','w')

    print("Training loop...")
    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch+1}/{opt.epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.epochs}'))

        model.train()

        point_loss = 0.0
        iteration = 0

        for inputs, target_lanes, target_h, test_image in tqdm.tqdm(data_loader.Generate()):
            real_batch_size=len(target_lanes)
            iteration += 1
            # print(real_batch_size)

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

            loss.backward()
            optimizer.step()

            point_loss += loss.item() * inputs.size(0)

            if step%100 ==0:
                testing(model, test_image, step, loss)

            step += 1

        training_loss = point_loss / (real_batch_size*iteration)
        loss_list.append(training_loss)
        print(f'training_loss: {training_loss:.4f}')

        if training_loss < best_loss:
            best_loss = training_loss
            best_model_params = copy.deepcopy(model.state_dict())

        if (epoch+1)%50 == 0:
            model.load_state_dict(best_model_params)
            weight_path=Path(opt.save_path).joinpath(f'model-{epoch+1}epoch-{best_loss:.02f}-best_train_loss.pth')
            torch.save(model,str(weight_path))
            record.write(f'{epoch+1}\n')
            record.write(f'Best training loss: {best_train_loss:.4f}\n\n')

        if epoch > 0 and epoch%10 == 0:
            print("evaluaton...")

            model.eval()

            th_list = [0.3, 0.5, 0.7]
            
            for th in th_list:
                print("generate result")
                print(th)
                name="test_result_"+str(epoch)+"_"+str(th)+".json"
                save_path = Path(opt.save_path).joinpath('json_test_result').joinpath(name)

                result_data = copy.deepcopy(data_loader.test_data)
                for test_image, target_h, ratio_w, ratio_h, testset_index in data_loader.Generate_Test():
                    x, y, _ = test(model, np.array([test_image]), thresh=0.81)
                    x, y = convert_to_original_size(x[0], y[0], ratio_w, ratio_h)
                    x, y = find_target(x, y, target_h, ratio_w, ratio_h)
                    result_data = write_result_json(result_data, x, y, testset_index)

                if name != None:
                    save_result(result_data, str(save_path))
                else:
                    save_result(result_data, name)
                
            for th in th_list:
                print("compute score")
                print(th)
                txt_file =  Path(opt.save_path).joinpath('txt_eval_result').joinpath("eval_result_"+str(th)+"_.txt")
                with open(str(txt_file), 'a') as make_file:
                    make_file.write( "epoch : " + str(epoch) + " loss : " + str(training_loss))
                    test_result_path = Path(opt.save_path).joinpath('json_test_result').joinpath("test_result_"+str(epoch)+"_"+str(th)+".json")
                    make_file.write(LaneEval.bench_one_submit(str(test_result_path), "test_label.json"))
                    make_file.write("\n")


def testing(model, test_image, step, loss):
    model.eval()

    _, _, ti = test(model, np.array([test_image]))

    image_path =  Path(opt.save_path).joinpath('test_result/result_'+str(step)+'_'+str(loss)+'.png')
    cv2.imwrite(str(image_path), ti[0])

    model.train()


if __name__ == '__main__':
    train()