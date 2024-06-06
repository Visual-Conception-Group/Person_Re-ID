import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image
from UtilsFile import *
class PersonDetect(object):
    
    def __init__(self,weights,):
        self.imgsz = 640
        self.image_path = os.path.join("detect_image")
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        self.label_path = os.path.join("detect_label")
        if not os.path.exists(self.label_path):
            os.makedirs(self.label_path)
        self.weights = weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(self.weights,map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16
        self.augument = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.save_conf = False

    def get_bounding_box(self,source):
        save_img = True
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augument)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres,self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            self.annoted_image_path = os.path.join(self.image_path,p.name)  # img.jpg
            self.annoted_label_path = os.path.join(self.label_path,p.stem)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                person_count = 0
                frame_wise_folder = os.path.join(self.image_path,p.stem)
                if not os.path.exists(frame_wise_folder):
                    os.makedirs(frame_wise_folder)
                for *xyxy, conf, cls in reversed(det):
                    if int(cls.cpu().numpy()) == 0:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(self.annoted_label_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        person_count += 1
                        x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                        cv2.imwrite(os.path.join(frame_wise_folder,"{}_person_{}.jpg".format(p.stem,person_count)),
                        im0[y1:y1+(y2-y1),x1:x1+(x2-x1),:])
                for *xyxy, conf, cls in reversed(det):
                    if int(cls.cpu().numpy()) == 0:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=10)
           # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Stream results
            cv2.imshow(str(p), im0)
            cv2.waitKey(0)  # 1 millisecond
            # Save results (image with detections)
            cv2.imwrite(self.annoted_image_path, im0)
        print(f"Results saved to {self.label_path}")
        print(f'Done. ({time.time() - t0:.3f}s)')



detect_person = PersonDetect(weights="crowdhuman_yolov5m.pt")
# detect_person.get_bounding_box("../framedir/second_1_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_10_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_20_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_30_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_40_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_50_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_60_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_70_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_80_frameno_0.jpg")
# detect_person.get_bounding_box("../framedir/second_90_frameno_0.jpg")

# import numpy as np
# x = np.random.rand(360,640,3)
# x1,y1,x2,y2 = 478,27,501,103
# print(x[y1:y1+(y2-y1),x1:x1+(x2-x1),:])

