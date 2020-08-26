from pathlib import Path
import os, cv2, torch
import threading
from detection_program.options import opt
import numpy as np
from torch.autograd import Variable
import copy, time
from utils.eval_utils import generate_result, eliminate_fewer_points, sort_along_y, eliminate_out, draw_points
from PySide2.QtCore import Signal
from PySide2 import QtCore

class Lane_Detection(QtCore.QObject):

    update_progressbar = Signal(float)
    detect_callback = Signal(list)
    update_output_imgs = Signal(str, int, int)

    def __init__(self, input_path, output_path, is_inputvideo, is_outputvideo, is_outputclips, widget):
        super().__init__()

        self.input_path = input_path
        self.output_clips_path = os.path.join(output_path, 'clips')
        self.output_video_path = os.path.join(output_path, 'video')
        self.is_inputvideo = is_inputvideo
        self.is_outputvideo = is_outputvideo
        self.is_outputclips = is_outputclips
        self.model = torch.load("detection_program/model/model.pth", map_location='cuda:'+str(opt.cuda_devices))
        self.clips = list()
        self.subpaths = list()
        self.fps = 30

        if not os.path.isdir(self.output_clips_path):
            os.makedirs(self.output_clips_path)
        
        if not os.path.isdir(self.output_video_path):
            os.makedirs(self.output_video_path)

        self.widget = widget
    
    def run(self):
        
        t = threading.currentThread()

        # Video to clips
        if self.is_inputvideo:
            vid = cv2.VideoCapture(self.input_path)
            length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            #self.update_progressbar.emit(60) #Succeed
            #self.widget.progressBar.setValue(60) #Failed
            i = 0
            start_time = 0
            while(vid.isOpened() and getattr(t, "do_run", True)):
                
                #spf = 0 if start_time == 0 else 1/(time.time() - start_time)
                if start_time == 0:
                    spf = 0
                else:
                    spf = 1/(time.time() - start_time)
                start_time = time.time()

                self.widget.label_info.setText(f'Detect lane: \t{str(i).zfill(5)}.jpg\nExecution time of the previous frame: \t{spf:.2f} second')
                self.update_progressbar.emit(i*100/length)
                ret, frame = vid.read()
                if ret == False:
                    break
                self.detect(frame, str(i).zfill(5)+'.jpg', i)
                i+=1
 
            vid.release()
            cv2.destroyAllWindows()
        
        # Read clips
        if not self.is_inputvideo:
            images = os.listdir(self.input_path)
            length = len(images)

            i = 0
            for num, path in enumerate(images):
                if not getattr(t, "do_run", True):
                    break

                frame = cv2.imread(str(Path(self.input_path).joinpath(path)))
                self.widget.label_info.setText(f'Detect lane: \t{path}\n')
                self.update_progressbar.emit(i*100/length)
                self.detect(frame, path, num)
                i+=1
        
        # Generate video
        if self.is_outputvideo:
            self.gen_video()
            print("generate video complete")
        
        self.detect_callback.emit(self.subpaths)

    def detect(self, image, name, num):
        image = cv2.resize(image, (512,256))
        image = np.rollaxis(image, axis=2, start=0)/255.0
        _, _, result_image = self.gen_test(np.array([image]))

        image_path = os.path.join(self.output_clips_path, name)
        cv2.imwrite(str(image_path), result_image[0])
        self.subpaths.append(name)

        self.update_output_imgs.emit(image_path, num, 0)
        
        if self.is_outputvideo:
            self.clips.append(result_image[0])

    def gen_test(self, test_images, thresh = 0.81):
        test_images =  torch.from_numpy(test_images).float()
        test_images = Variable(test_images).cuda(opt.cuda_devices)
        result = self.model(test_images)
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
    
    def gen_video(self):
        height, width, layers = self.clips[0].shape
        size = (width,height)

        out = cv2.VideoWriter(os.path.join(self.output_video_path, 'video.avi'),cv2.VideoWriter_fourcc(*'DIVX'), self.fps, size)

        for clip in self.clips:
            # writing to a image array
            out.write(clip)
        out.release()