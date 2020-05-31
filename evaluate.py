import copy
import numpy as np
from utils.eval_utils import generate_result, eliminate_fewer_points, sort_along_y, eliminate_out, draw_points
import json
from sklearn.linear_model import LinearRegression
import ujson
import torch
from torch.autograd import Variable
from options import opt


x_size = 512
y_size = 256


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [ujson.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [ujson.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return ujson.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])


# test on the input test image
def test(model, test_images, thresh = 0.81):
    test_images =  torch.from_numpy(test_images).float()
    test_images = Variable(test_images).cuda(opt.cuda_devices)
    result = model(test_images)
    confidences, offsets, instances = result[-1]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []

    test_images = test_images.cpu().numpy()

    for i in range(num_batch):
        # test on test data set
        image = copy.deepcopy(test_images[i])
        image =  np.rollaxis(image, axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(opt.grid_y, opt.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)

        # sort points along y 
        in_x, in_y = sort_along_y(in_x, in_y)
        in_x, in_y = eliminate_out(in_x, in_y, confidence, copy.deepcopy(image))
        in_x, in_y = sort_along_y(in_x, in_y)
        in_x, in_y = eliminate_fewer_points(in_x, in_y)

        result_image = draw_points(in_x, in_y, copy.deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y,  out_images


def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x,y):
        out_x.append((np.array(i)/ratio_w).tolist())
        out_y.append((np.array(j)/ratio_h).tolist())

    return out_x, out_y


# linear interpolation for fixed y value on the test dataset
def find_target(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    new_x_size = x_size/ratio_w
    new_y_size = y_size/ratio_h
    for i, j in zip(x,y):
        min_y = min(j)
        max_y = max(j)
        temp_x = []
        temp_y = []
        for h in target_h:
            temp_y.append(h)
            if h < min_y:
                temp_x.append(-2)
            elif min_y <= h and h <= max_y:
                for k in range(len(j)-1):
                    if j[k] >= h and h >= j[k+1]:
                        #linear regression
                        if i[k] < i[k+1]:
                            temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        else:
                            temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        break
            else:
                if i[0] < i[1]:
                    l = int(i[1] - float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > new_x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
                else:
                    l = int(i[1] + float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > new_x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y


# write result
def write_result_json(result_data, x, y, testset_index):
    for i in x:
        result_data[testset_index]['lanes'].append(i)
        result_data[testset_index]['run_time'] = 1
    return result_data


# save result by json form
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")