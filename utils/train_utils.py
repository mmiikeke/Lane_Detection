import numpy as np
import sys
from options import opt
print(sys.path)


def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y


# Make ground truth for key point estimation
def make_ground_truth_point(target_lanes, target_h):

    target_lanes, target_h = sort_batch_along_y(target_lanes, target_h)

    ground = np.zeros((len(target_lanes), 3, opt.grid_y, opt.grid_x))
    ground_binary = np.zeros((len(target_lanes), 1, opt.grid_y, opt.grid_x))

    for batch_index, batch in enumerate(target_lanes):
        for lane_index, lane in enumerate(batch):
            for point_index, point in enumerate(lane):
                if point > 0:
                    x_index = int(point/opt.resize_ratio)
                    y_index = int(target_h[batch_index][lane_index][point_index]/opt.resize_ratio)
                    ground[batch_index][0][y_index][x_index] = 1.0
                    ground[batch_index][1][y_index][x_index]= (point*1.0/opt.resize_ratio) - x_index
                    ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/opt.resize_ratio) - y_index
                    ground_binary[batch_index][0][y_index][x_index] = 1

    return ground, ground_binary


# Make ground truth for instance feature
def make_ground_truth_instance(target_lanes, target_h):

    ground = np.zeros((len(target_lanes), 1, opt.grid_y*opt.grid_x, opt.grid_y*opt.grid_x))

    for batch_index, batch in enumerate(target_lanes):
        temp = np.zeros((1, opt.grid_y, opt.grid_x))
        lane_cluster = 1
        for lane_index, lane in enumerate(batch):
            previous_x_index = 0
            previous_y_index = 0
            for point_index, point in enumerate(lane):
                if point > 0:
                    x_index = int(point/opt.resize_ratio)
                    y_index = int(target_h[batch_index][lane_index][point_index]/opt.resize_ratio)
                    temp[0][y_index][x_index] = lane_cluster
                if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                    temp_x = previous_x_index
                    temp_y = previous_y_index
                    while True:
                        delta_x = 0
                        delta_y = 0
                        temp[0][temp_y][temp_x] = lane_cluster
                        if temp_x < x_index:
                            temp[0][temp_y][temp_x+1] = lane_cluster
                            delta_x = 1
                        elif temp_x > x_index:
                            temp[0][temp_y][temp_x-1] = lane_cluster
                            delta_x = -1
                        if temp_y < y_index:
                            temp[0][temp_y+1][temp_x] = lane_cluster
                            delta_y = 1
                        elif temp_y > y_index:
                            temp[0][temp_y-1][temp_x] = lane_cluster
                            delta_y = -1
                        temp_x += delta_x
                        temp_y += delta_y
                        if temp_x == x_index and temp_y == y_index:
                            break
                if point > 0:
                    previous_x_index = x_index
                    previous_y_index = y_index
            lane_cluster += 1

        for i in range(opt.grid_y*opt.grid_x): #make gt
            temp = temp[temp>-1]
            gt_one = deepcopy(temp)
            if temp[i]>0:
                gt_one[temp==temp[i]] = 1   #same instance
                if temp[i] == 0:
                    gt_one[temp!=temp[i]] = 3 #different instance, different class
                else:
                    gt_one[temp!=temp[i]] = 2 #different instance, same class
                    gt_one[temp==0] = 3 #different instance, different class
                ground[batch_index][0][i] += gt_one

    return ground

