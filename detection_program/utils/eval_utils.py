import copy
import torch
from torch.autograd import Variable
from options import opt
import math
import cv2
import json
import numpy as np


global x_size, y_size, color
x_size=512
y_size = 256
color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]



# generate raw output
def generate_result(confidance, offsets, instance, thresh):
    mask = confidance > thresh
    #print(mask)

    grid_location = np.zeros((opt.grid_y, opt.grid_x, 2))
    for y in range(opt.grid_y):
        for x in range(opt.grid_x):
            grid_location[y][x][0] = x
            grid_location[y][x][1] = y

    grid = grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*opt.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*opt.resize_ratio)
            if point_x > x_size or point_x < 0 or point_y > y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([])
                x[0].append(point_x)
                y.append([])
                y[0].append(point_y)
            else:
                flag = 0
                index = 0
                for feature_idx, j in enumerate(lane_feature):
                    index += 1
                    if index >= 12:
                        index = 12
                    if np.linalg.norm((feature[i] - j)**2) <= 0.22:
                        lane_feature[feature_idx] = (j*len(x[index-1]) + feature[i])/(len(x[index-1])+1)
                        x[index-1].append(point_x)
                        y[index-1].append(point_y)
                        flag = 1
                        break
                if flag == 0:
                    lane_feature.append(feature[i])
                    x.append([])
                    x[index].append(point_x) 
                    y.append([])
                    y[index].append(point_y)
                
    return x, y


# eliminate result that has fewer points than threshold
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y 

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y 


# post processing for eliminating outliers
def eliminate_out(sorted_x, sorted_y, confidence, image = None):
    out_x = []
    out_y = []

    for lane_x, lane_y in zip(sorted_x, sorted_y):

        lane_x_along_y = np.array(copy.deepcopy(lane_x))
        lane_y_along_y = np.array(copy.deepcopy(lane_y))

        ind = np.argsort(lane_x_along_y, axis=0)
        lane_x_along_x = np.take_along_axis(lane_x_along_y, ind, axis=0)
        lane_y_along_x = np.take_along_axis(lane_y_along_y, ind, axis=0)
        
        if lane_y_along_x[0] > lane_y_along_x[-1]: #if y of left-end point is higher than right-end
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]), (lane_x_along_y[2], lane_y_along_y[2]),
                                (lane_x_along_x[0], lane_y_along_x[0]), (lane_x_along_x[1], lane_y_along_x[1]), (lane_x_along_x[2], lane_y_along_x[2])] # some low y, some left/right x
        else:
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]), (lane_x_along_y[2], lane_y_along_y[2]),
                                (lane_x_along_x[-1], lane_y_along_x[-1]), (lane_x_along_x[-2], lane_y_along_x[-2]), (lane_x_along_x[-3], lane_y_along_x[-3])] # some low y, some left/right x            
    
        temp_x = []
        temp_y = []
        for start_point in starting_points:
            temp_lane_x, temp_lane_y = generate_cluster(start_point, lane_x, lane_y, image)
            temp_x.append(temp_lane_x)
            temp_y.append(temp_lane_y)
        
        max_lenght_x = None
        max_lenght_y = None
        max_lenght = 0
        for i, j in zip(temp_x, temp_y):
            if len(i) > max_lenght:
                max_lenght = len(i)
                max_lenght_x = i
                max_lenght_y = j
        out_x.append(max_lenght_x)
        out_y.append(max_lenght_y)

    return out_x, out_y


# generate cluster
def generate_cluster(start_point, lane_x, lane_y, image=None):
    cluster_x = [start_point[0]]
    cluster_y = [start_point[1]]

    point = start_point
    while True:
        points = get_closest_upper_point(lane_x, lane_y, point, 3)
         
        max_num = -1
        max_point = None

        if len(points) == 0:
            break
        if len(points) < 3:
            for i in points: 
                cluster_x.append(i[0])
                cluster_y.append(i[1])                
            break
        for i in points: 
            num, shortest = get_num_along_point(lane_x, lane_y, point, i, image)
            if max_num < num:
                max_num = num
                max_point = i

        total_remain = len(np.array(lane_y)[np.array(lane_y) < point[1]])
        cluster_x.append(max_point[0])
        cluster_y.append(max_point[1])
        point = max_point
        
        if len(points) == 1 or max_num < total_remain/5:
            break

    return cluster_x, cluster_y


def get_closest_upper_point(x, y, point, n):
    x = np.array(x)
    y = np.array(y)

    x = x[y<point[1]]
    y = y[y<point[1]]

    dis = (x - point[0])**2 + (y - point[1])**2

    ind = np.argsort(dis, axis=0)
    x = np.take_along_axis(x, ind, axis=0).tolist()
    y = np.take_along_axis(y, ind, axis=0).tolist()

    points = []
    for i, j in zip(x[:n], y[:n]):
        points.append((i,j))

    return points


def get_num_along_point(x, y, point1, point2, image=None): # point1 : source
    x = np.array(x)
    y = np.array(y)

    x = x[y<point1[1]]
    y = y[y<point1[1]]

    dis = np.sqrt( (x - point1[0])**2 + (y - point1[1])**2 )

    count = 0
    shortest = 1000
    target_angle = get_angle_two_points(point1, point2)
    for i in range(len(dis)):
        angle = get_angle_two_points(point1, (x[i], y[i]))
        diff_angle = abs(angle-target_angle)
        distance = dis[i] * math.sin( diff_angle*math.pi*2 )
        if distance <= 12:
            count += 1
            if distance < shortest:
                shortest = distance

    return count, shortest


def get_angle_two_points(p1, p2):
    del_x = p2[0] - p1[0]
    del_y = p2[1] - p1[1] + 0.000001    
    if p2[0] >= p1[0] and p2[1] > p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta /= 360.0
    elif  p2[0] > p1[0] and p2[1] <= p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta += 180
        theta /= 360.0
    elif  p2[0] <= p1[0] and p2[1] < p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta += 180
        theta /= 360.0
    elif  p2[0] < p1[0] and p2[1] >= p1[1]:
        theta = math.atan(float(del_x/del_y))*180/math.pi
        theta += 360
        theta /= 360.0
    
    return theta


def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 5, color[color_index], -1)

    return image

def sort_lane(x, y, sort_at):
    out_x = []
    out_y = []
    order = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        #index = min(range(len(j)), key=lambda k: abs(j[k]-sort_at))
        l = 10000
        s = 10000
        l_i = -1
        s_i = -1
        for k in range(len(j)):
            if (j[k] > sort_at) and ((j[k] - sort_at) < l):
                l = j[k] - sort_at
                l_i = k
            elif (j[k] < sort_at) and ((sort_at - j[k]) < s):
                s = sort_at - j[k]
                s_i = k
        if l_i == -1 or s_i == -1:
            a = i[0]
            b = i[-1]
            c = j[0]
            d = j[-1]
            order.append(a + (b - a) * (200 - c) / (d - c))
        else:
            a = i[s_i]
            b = i[l_i]
            order.append((a * l + b * s)/(l + s))

    iorder = np.argsort(order, axis=0)
    order.sort()

    for i in iorder:
        out_x.append(x[i])
        out_y.append(y[i])
    
    return out_x, out_y, order 


        
