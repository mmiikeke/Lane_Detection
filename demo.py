from pathlib import Path
import os
import cv2
import torch
from options import opt
import numpy as np
from torch.autograd import Variable
import copy
from utils.eval_utils import generate_result, eliminate_fewer_points, sort_along_y, eliminate_out, draw_points


CLIPPATH = "data/video/frame_input"
CLIPSAVEPATH = "data/video/frame_output"
VIDSAVEPATH = "data/video/video.avi"
fps = 30

def gen_video():
    frame_array = []
    files = [f for f in os.listdir(CLIPSAVEPATH) if os.path.isfile(os.path.join(CLIPSAVEPATH, f))]
    files.sort()
    
    for i in range(len(files)):
        filename=os.path.join(CLIPSAVEPATH, files[i])
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(VIDSAVEPATH,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def gen_test(test_images, thresh = 0.81):
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


def main():
    for image in os.listdir(CLIPPATH):
        test_image = cv2.imread(str(Path(CLIPPATH).joinpath(image)))
        test_image = cv2.resize(test_image, (512,256)) 
        test_image = np.rollaxis(test_image, axis=2, start=0)/255.0
        _, _, result_image = gen_test(np.array([test_image]))
        image_path = Path(CLIPSAVEPATH).joinpath(image)
        cv2.imwrite(str(image_path), result_image[0])
        print(image + " complete")
    gen_video()
    print("generate video complete")


if __name__=='__main__':
    model = torch.load("model/model.pth", map_location='cuda:'+str(opt.cuda_devices))
    main()
