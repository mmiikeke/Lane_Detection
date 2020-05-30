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