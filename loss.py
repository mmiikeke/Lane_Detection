import torch
import torch.nn as nn
from options import opt


class SGPNLoss(nn.Module):
    def __init__(self):
        super(SGPNLoss,self).__init__()
        self.feature_size =  4
        self.K1 =  1.0
        self.K2 = 2.0
        self.constant_exist = 1.0
        self.constant_nonexist = 1.0
        self.constant_offset =  1.0
        self.constant_alpha =  1.0
        self.constant_beta = 1.0
        self.constant_lane_loss = 1.0
        self.constant_instance_loss = 1.0
        self.current_epoch = 0

    def forward(self, result, ground_truth_point, ground_truth_instance, real_batch_size, epoch):
        for (confidance, offset, feature) in result:
            #compute loss for point prediction
            offset_loss = 0
            exist_condidence_loss = 0
            nonexist_confidence_loss = 0

            #exist confidance loss
            confidance_gt = ground_truth_point[:, 0, :, :]
            confidance_gt = confidance_gt.view(real_batch_size, 1, opt.grid_y, opt.grid_x)
            exist_condidence_loss = torch.sum((confidance_gt[confidance_gt==1] - confidance[confidance_gt==1])**2)/torch.sum(confidance_gt==1)

            #non exist confidance loss
            nonexist_confidence_loss = torch.sum((confidance_gt[confidance_gt==0] - confidance[confidance_gt==0])**2)/torch.sum(confidance_gt==0)

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

            feature_map = feature.view(real_batch_size, self.feature_size, 1, opt.grid_y*opt.grid_x)
            feature_map = feature_map.expand(real_batch_size, self.feature_size, opt.grid_y*opt.grid_x, opt.grid_y*opt.grid_x).detach()

            point_feature = feature.view(real_batch_size, self.feature_size, opt.grid_y*opt.grid_x,1)
            point_feature = point_feature.expand(real_batch_size, self.feature_size, opt.grid_y*opt.grid_x, opt.grid_y*opt.grid_x)#.detach()

            distance_map = (feature_map-point_feature)**2 
            distance_map = torch.norm( distance_map, dim=1 ).view(real_batch_size, 1, opt.grid_y*opt.grid_x, opt.grid_y*opt.grid_x)

            # same instance
            sisc_loss = torch.sum(distance_map[ground_truth_instance==1])/torch.sum(ground_truth_instance==1)

            # different instance, same class
            disc_loss = self.K1-distance_map[ground_truth_instance==2] #self.p.K1/distance_map[ground_truth_instance==2] + (self.p.K1-distance_map[ground_truth_instance==2])
            disc_loss[disc_loss<0] = 0
            disc_loss = torch.sum(disc_loss)/torch.sum(ground_truth_instance==2)

            """print("seg loss################################################################")
            print(sisc_loss)
            print(disc_loss)

            print("point loss")
            print(exist_condidence_loss)
            print(nonexist_confidence_loss)
            print(offset_loss)

            print("lane loss")"""
            lane_loss = self.constant_exist*exist_condidence_loss + self.constant_nonexist*nonexist_confidence_loss + self.constant_offset*offset_loss
            # print(lane_loss)

            # print("instance loss")
            instance_loss = self.constant_alpha*sisc_loss + self.constant_beta*disc_loss
            # print(instance_loss)

            lane_detection_loss = self.constant_lane_loss*lane_loss + self.constant_instance_loss*instance_loss
            if (epoch+1) >= 800:
                self.constant_nonexist = 1.5
                self.constant_lane_loss = 1.5
            
            if epoch>0 and (epoch+1)%5 == 0:
                self.constant_alpha *= 2

            return lane_detection_loss