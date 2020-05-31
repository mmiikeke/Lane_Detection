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
from visual import plt_loss


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

    feature_size =  4
    K1 =  1.0
    K2 = 2.0
    constant_exist = 1.0
    constant_nonexist = 1.0
    constant_offset =  1.0
    constant_alpha =  1.0
    constant_beta = 1.0
    constant_lane_loss = 1.0
    constant_instance_loss = 1.0
    current_epoch = 0

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

            result = model(inputs)

            lane_detection_loss = 0.0
            for (confidance, offset, feature) in result:
                #compute loss for point prediction
                offset_loss = 0
                exist_condidence_loss = 0
                nonexist_confidence_loss = 0

                #exist confidance loss
                confidance_gt = ground_truth_point[:, 0, :, :]
                confidance_gt = confidance_gt.view(real_batch_size, 1, opt.grid_y, opt.grid_x)
                exist_condidence_loss = torch.sum(    (confidance_gt[confidance_gt==1] - confidance[confidance_gt==1])**2      )/torch.sum(confidance_gt==1)

                #non exist confidance loss
                nonexist_confidence_loss = torch.sum(    (confidance_gt[confidance_gt==0] - confidance[confidance_gt==0])**2      )/torch.sum(confidance_gt==0)

                #offset loss 
                offset_x_gt = ground_truth_point[:, 1:2, :, :]
                offset_y_gt = ground_truth_point[:, 2:3, :, :]

                predict_x = offset[:, 0:1, :, :]
                predict_y = offset[:, 1:2, :, :]

                x_offset_loss = torch.sum( (offset_x_gt[confidance_gt==1] - predict_x[confidance_gt==1])**2 )/torch.sum(confidance_gt==1)
                y_offset_loss = torch.sum( (offset_y_gt[confidance_gt==1] - predict_y[confidance_gt==1])**2 )/torch.sum(confidance_gt==1)

                offset_loss = (x_offset_loss + y_offset_loss)/2

                #compute loss for similarity
                sisc_loss = 0
                disc_loss = 0

                feature_map = feature.view(real_batch_size, feature_size, 1, opt.grid_y*opt.grid_x)
                feature_map = feature_map.expand(real_batch_size, feature_size, opt.grid_y*opt.grid_x, opt.grid_y*opt.grid_x).detach()

                point_feature = feature.view(real_batch_size, feature_size, opt.grid_y*opt.grid_x,1)
                point_feature = point_feature.expand(real_batch_size, feature_size, opt.grid_y*opt.grid_x, opt.grid_y*opt.grid_x)#.detach()

                distance_map = (feature_map-point_feature)**2 
                distance_map = torch.norm( distance_map, dim=1 ).view(real_batch_size, 1, opt.grid_y*opt.grid_x, opt.grid_y*opt.grid_x)

                # same instance
                sisc_loss = torch.sum(distance_map[ground_truth_instance==1])/torch.sum(ground_truth_instance==1)

                # different instance, same class
                disc_loss = K1-distance_map[ground_truth_instance==2] #self.p.K1/distance_map[ground_truth_instance==2] + (self.p.K1-distance_map[ground_truth_instance==2])
                disc_loss[disc_loss<0] = 0
                disc_loss = torch.sum(disc_loss)/torch.sum(ground_truth_instance==2)

                print("seg loss################################################################")
                print(sisc_loss)
                print(disc_loss)

                print("point loss")
                print(exist_condidence_loss)
                print(nonexist_confidence_loss)
                print(offset_loss)

                print("lane loss")
                lane_loss = constant_exist*exist_condidence_loss + constant_nonexist*nonexist_confidence_loss + constant_offset*offset_loss
                print(lane_loss)

                print("instance loss")
                instance_loss = constant_alpha*sisc_loss + constant_beta*disc_loss
                print(instance_loss)

                lane_detection_loss = lane_detection_loss + constant_lane_loss*lane_loss + constant_instance_loss*instance_loss

            optimizer.zero_grad()
            lane_detection_loss.backward()
            optimizer.step()

            del confidance, offset, feature
            del ground_truth_point, ground_binary, ground_truth_instance
            del feature_map, point_feature, distance_map
            del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss, lane_loss, instance_loss
            
            if epoch>0 and epoch%20==0 and current_epoch != epoch:
                urrent_epoch = epoch
                if epoch>0 and (epoch == 1000):
                    constant_lane_loss += 0.5
                constant_nonexist += 0.5
                l_rate /= 2.0
                optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=opt.weight_decay)
            
            point_loss += lane_detection_loss.item() * inputs.size(0)

            if step%1000 == 0:
                testing(model, test_image, step, point_loss)

            step += 1

        training_loss = point_loss / (real_batch_size*iteration)
        loss_list.append(training_loss)
        print(f'training_loss: {training_loss:.4f}\n')

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
    
    loss_list = np.round(loss_list,4)
    plt_loss(loss_list)
    

def testing(model, test_image, step, loss):
    model.eval()

    _, _, ti = test(model, np.array([test_image]))

    image_path =  Path(opt.save_path).joinpath('test_result/result_'+str(step)+'_'+str(loss)+'.png')
    cv2.imwrite(str(image_path), ti[0])

    model.train()


if __name__ == '__main__':
    train()