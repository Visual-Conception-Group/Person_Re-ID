from asyncio import Handle
from base64 import encodebytes
from curses import meta
from genericpath import isfile
import os
from posixpath import splitext 
import math
import random
import json
import time
from pathlib import Path
import cv2
import torchreid
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image
from UtilsFile import *
import pathlib
import io
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as transforms
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from reid_module import get_gallery_query_dataset
import sqlite3
import pickle
def get_video_info(root_folder,username):
    folder_name = os.path.join(root_folder,username)
    list_of_video = [os.path.join(folder_name, item) for item in os.listdir(folder_name)
                            if os.path.isfile(os.path.join(folder_name, item)) and os.path.splitext(item)[1] == ".mp4"][0]
    cap = cv2.VideoCapture(list_of_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count/fps
    return dict(video_name = list_of_video,
                video_duration = str(int(duration)) + " Seconds",
                no_of_frame_extract = str(int(duration)) +" Frames",
                video_path = list_of_video)

def get_video_info_v2(root_folder,username):
    folder_name = os.path.join(root_folder,username)
    list_of_videos = [os.path.join(folder_name, item) for item in os.listdir(folder_name)
                            if os.path.isfile(os.path.join(folder_name, item)) and os.path.splitext(item)[1] == ".mp4"]
    
    base_video = [video for video in list_of_videos if "video.mp4" in video][0]

    aux_videos = [video for video in list_of_videos if "aux" in video]

    cap = cv2.VideoCapture(base_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count/fps
    return dict(video_name = base_video,
                video_duration = str(int(duration)) + " Seconds",
                no_of_frame_extract = str(int(duration)) +" Frames",
                video_path = base_video,
                aux_videos = aux_videos)


class VideoProcess(object):
    
    def __init__(self,weights,root_dir,username,frame_per_second = 1):
        self.handle_progress = HandleProgress()
        self.username = username
        self.handle_progress.set_status(self.username,"Loading Human Object Detection Model|None")
        self.root_dir = root_dir
        self.frame_per_second = frame_per_second
        self.user_specific_dir = os.path.join(self.root_dir,self.username)
        self.normal_frame_save_dir = os.path.join(self.user_specific_dir,"NormalFrame")
        self.annotated_frame_save_dir = os.path.join(self.user_specific_dir,"AnnotatedFrame")
        self.required_folder_create()
        self.video_metadata = list()
        self.video_metadata_v2 = dict()
        self.weights = weights
        self.imgsz = 640
        self.weights = weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TODO: For Demo, change device to cpu
        self.device = torch.device("cpu")

        self.model = attempt_load(self.weights,map_location=self.device)
        self.osnet = torchreid.models.osnet_x1_0(pretrained=False)
        self.osnet.load_state_dict(torch.load("osnet_x1_0_imagenet.pth",map_location=self.device))
        self.osnet.eval()
        self.image_transforms = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
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

    def required_folder_create(self):
        
        if not os.path.exists(self.normal_frame_save_dir):
            os.makedirs(self.normal_frame_save_dir)
        if not os.path.exists(self.annotated_frame_save_dir):
            os.makedirs(self.annotated_frame_save_dir)
    
    def read_video(self):
        self.video_name  =  [os.path.join(self.user_specific_dir, item) for item in os.listdir(self.user_specific_dir)
                            if os.path.isfile(os.path.join(self.user_specific_dir, item)) and os.path.splitext(item)[1] == ".mp4"][0]
        self.handle_progress.set_status(self.username,"Video loading memeory|None")
        self.cap = cv2.VideoCapture(self.video_name)
        self.handle_progress.set_status(self.username,"Video loaded memeory|None")
        self.frame_rate = math.floor(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_per_second = self.frame_per_second if self.frame_rate >= self.frame_per_second  else self.frame_rate 
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = self.frame_count/self.frame_rate
        self.duration = int(self.duration)
        frame_no = 0
        second_count = 1
        second_wise_frame_list = list()
        self.handle_progress.set_status(self.username,f"Frame extracting|{str(second_count)}/{str(self.duration)}")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if(frame_no < self.frame_rate ):
                second_wise_frame_list.append(frame)
                frame_no += 1
            else:
                random_frame_selection = random.sample(second_wise_frame_list,k = self.frame_per_second)
                for index,frame_content in enumerate(random_frame_selection):
                    print("second_{}_image_{}.jpg processing...".format(second_count,index))
                    file_path = os.path.join(self.normal_frame_save_dir,"second_{}_frameno_{}.jpg".format(second_count,index))
                    cv2.imwrite(file_path,frame_content)
                    self.video_metadata.append(dict(normal_frame_path = file_path)) 
                frame_no = 0
                second_count += 1
                self.handle_progress.set_status(self.username,f"Frame extracting|{str(second_count)}/{str(self.duration)}")
                random_frame_selection = list()
        self.cap.release()
        self.handle_progress.set_status(self.username,f"Frame extracting|{str(second_count-1)}/{str(self.duration)}")
        self.handle_progress.set_status(self.username,f"Frame extracting complete|None") 

    def analyse_video(self, video_name, is_base_video=True):
        self.handle_progress.set_status(self.username,"Video name:{}|None".format(video_name))
        self.handle_progress.set_status(self.username,"Video loading memory|None")
        cap = cv2.VideoCapture(video_name)
        self.handle_progress.set_status(self.username,"Video loaded memory|None")
        frame_rate = math.floor(cap.get(cv2.CAP_PROP_FPS))
        self.frame_per_second = self.frame_per_second if frame_rate >= self.frame_per_second  else frame_rate 
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = frame_count/frame_rate
        frame_count = int(frame_count)
        frame_no = 0
        second_count = 1
        second_wise_frame_list = list()
        self.handle_progress.set_status(self.username,f"Frame extracting|{str(second_count)}/{str(frame_count)}")

        self.video_metadata_v2[video_name] = list()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if(frame_no < frame_rate ):
                second_wise_frame_list.append(frame)
                frame_no += 1
            else:
                random_frame_selection = random.sample(second_wise_frame_list,k = self.frame_per_second)
                
                for index,frame_content in enumerate(random_frame_selection):
                    aux_name = video_name.split('/')[-1]
                    print("name_{}_second_{}_image_{}.jpg processing...".format(aux_name, second_count, index))
                    file_path = os.path.join(self.normal_frame_save_dir,"name_{}_second_{}_frameno_{}.jpg".format(aux_name, second_count,index))
                    cv2.imwrite(file_path,frame_content)
                    self.video_metadata_v2[video_name].append(dict(normal_frame_path = file_path)) 
                
                frame_no = 0
                second_count += 1
                self.handle_progress.set_status(self.username,f"Frame extracting|{str(second_count)}/{str(frame_count)}")
                random_frame_selection = list()
        
        cap.release()
        self.handle_progress.set_status(self.username,f"Frame extracting|{str(second_count-1)}/{str(frame_count)}")
        self.handle_progress.set_status(self.username,f"Frame extracting complete|None")
    
    
    def read_video_v2(self):
        list_of_videos = [os.path.join(self.user_specific_dir, item) for item in os.listdir(self.user_specific_dir)
                            if os.path.isfile(os.path.join(self.user_specific_dir, item)) and os.path.splitext(item)[1] == ".mp4"]
        
        self.video_name = [video for video in list_of_videos if "video.mp4" in video][0]
        self.aux_video_names = [video for video in list_of_videos if "aux" in video]

        self.analyse_video(self.video_name)

        for aux_video in self.aux_video_names:
            self.analyse_video(aux_video)


    def get_bounding_box(self,metadata_entry):
        dataset = LoadImages(metadata_entry["normal_frame_path"], img_size=self.imgsz, stride=self.stride)
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
                metadata_entry["annotated_frame_path"] = os.path.join(self.annotated_frame_save_dir,p.name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
                if len(det):                    
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    person_count = 0
                    annotated_file_path = os.path.join(self.annotated_frame_save_dir,p.stem)
                    if not os.path.exists(annotated_file_path):
                        os.makedirs(annotated_file_path)
                    metadata_entry["annotated_frame_wise_person_info"] = list()                    
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls.cpu().numpy()) == 0:
                            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            person_count += 1
                            x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                            person_image_path = os.path.join(annotated_file_path,"{}_person_{}.pkl".format(p.stem,person_count))
                            cv2.imwrite(os.path.join(self.user_specific_dir,"temp.jpg"),im0[y1:y1+(y2-y1),x1:x1+(x2-x1),:])
                            image = Image.open(os.path.join(self.user_specific_dir,"temp.jpg")).convert("RGB")
                            image = self.image_transforms(image)
                            # image = image.to(self.device)
                            image = image
                            image = torch.unsqueeze(image , 0)
                            image = self.osnet(image)
                            with open(person_image_path,"wb") as file_obj:
                                pickle.dump(image,file_obj)
                                file_obj.close()
                            metadata_entry["annotated_frame_wise_person_info"].append(
                                dict(person_image_path = person_image_path,
                                bounding_box = [x1,y1,x2,y2])
                            )
                    metadata_entry["no_of_person_in_frame"] = len(os.listdir(annotated_file_path))
                    # for *xyxy, conf, cls in reversed(det):
                    #     if int(cls.cpu().numpy()) == 0:
                    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #         line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                    #         label = f'{self.names[int(cls)]} {conf:.2f}'
                    #         plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=10)
                # cv2.imwrite(metadata_entry["annotated_frame_path"], im0)
        return metadata_entry
    
    def annotate_each_frame(self):
        self.handle_progress.set_status(self.username,f"Extacting person from frame|{str(0)}/{str(len(self.video_metadata))}")
        count = 0
        
        for index,frame_metadata_entry in enumerate(self.video_metadata):
            frame_metadata_entry = self.get_bounding_box(frame_metadata_entry)
            print(f"Object Annotation Complete for frame :{index + 1}")
            count += 1
            self.handle_progress.set_status(self.username,f"Extacting person from frame|{str(count)}/{str(len(self.video_metadata))}")
        
        with open(os.path.join(self.user_specific_dir,"metadata.json"),"w") as file_obj:
            json.dump(self.video_metadata,file_obj)
            file_obj.close()
        
        self.handle_progress.conn.close()
    
    def annotate_each_frame_v2(self):
        self.handle_progress.set_status(self.username,f"Extacting person from frame|{str(0)}/{str(len(self.video_metadata))}")
        count = 0
        
        for key in self.video_metadata_v2.keys():
            print(f"Object Annotation Starting for video :{key}")
            for index,frame_metadata_entry in enumerate(self.video_metadata_v2[key]):
                frame_metadata_entry = self.get_bounding_box(frame_metadata_entry)
                print(f"Object Annotation Complete for frame :{index + 1}")
                count += 1
                self.handle_progress.set_status(self.username,f"Extacting person from frame|{str(count)}/{str(len(self.video_metadata))}")
        
        with open(os.path.join(self.user_specific_dir,"metadata_v2.json"),"w") as file_obj:
            json.dump(self.video_metadata_v2,file_obj)
            file_obj.close()
        
        self.handle_progress.conn.close()

def get_frame_count(root_dir,username):
    frame_dir = os.path.join(root_dir,username,"NormalFrame")
    second_count = []
    for index,file_name in enumerate(os.listdir(frame_dir)):
        second_count.append(f"second_{index+1}")
    return second_count

def get_frame_count_v2(root_dir,username):
    frame_dir = os.path.join(root_dir,username,"NormalFrame")
    second_count = []
    cnt = 0
    for index,file_name in enumerate(os.listdir(frame_dir)):
        if "name_video" in file_name:
            a = file_name.split("_")
            second_count.append(f"second_{cnt+1}")
            cnt += 1
    return second_count

def get_total_frame_count(root_dir,username):
    frame_dir = os.path.join(root_dir,username,"NormalFrame")
    cnt = 0
    for file_name in os.listdir(frame_dir):
        cnt += 1
    return cnt


def get_second_info(root_dir,username,second_name):
    # get original frame 
    second_name = second_name.lower()
    video_metadata_json = os.path.join(root_dir,username,"metadata.json")
    data = dict()
    with open(video_metadata_json,"r") as file_obj:
        video_metadata = json.load(file_obj)
        for frame_metadata in video_metadata:
            path = pathlib.PurePath(frame_metadata["normal_frame_path"])
            a = path.name.split("_")
            a = a[0]+"_"+a[1]
            if a == second_name:
                data["normal_frame_path"] = frame_metadata["normal_frame_path"]
                data["annotated_frame_path"] = frame_metadata["annotated_frame_path"]
                data["annotated_frame_wise_person_info"] = frame_metadata["annotated_frame_wise_person_info"]
    file_obj.close()
    return data

def get_second_info_v2(root_dir,username,second_name):
    # get original frame 
    second_name = second_name.lower()
    video_metadata_json = os.path.join(root_dir,username,"metadata_v2.json")
    data = dict()
    with open(video_metadata_json,"r") as file_obj:
        video_metadata = json.load(file_obj)
        key = ""
        for dict_key in video_metadata.keys():
            if "video.mp4" in dict_key:
                key = dict_key
                break
        
        for frame_metadata in video_metadata[key]:
            path = pathlib.PurePath(frame_metadata["normal_frame_path"])
            a = path.name.split("_")
            a = a[2] + "_" + a[3]
            if second_name == a:
                data["normal_frame_path"] = frame_metadata["normal_frame_path"]
                data["annotated_frame_path"] = frame_metadata["annotated_frame_path"]
                try:
                    data["annotated_frame_wise_person_info"] = frame_metadata["annotated_frame_wise_person_info"]
                    data["no_of_person_in_frame"] = frame_metadata["no_of_person_in_frame"]
                except:
                    data["no_of_person_in_frame"] = 0
                
    file_obj.close()
    return data

def get_image_response(image_path):
    image_name = pathlib.Path(image_path).stem
    pil_image = Image.open(image_path,mode = 'r')
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr,format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    return dict(image = encoded_img , image_name = image_name)

class CustomgalleryDataset(Dataset):
    
    def __init__(self,images,apply_image_transforms):
        self.images = images
        self.apply_image_transforms = apply_image_transforms

    def __self__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        frame_name = self.images[index]["frame_name"]
        person_image_name = self.images[index]["person_name"]
        bounding_box = self.images[index]["bounding_box"]
        with open(person_image_name , "rb") as file_obj:
            person_image = pickle.load(file_obj)
        return (frame_name,person_image_name,bounding_box,person_image)

class PersonReidentification(object):
    
    def __init__(self,root_dir,username,frame_info,query_image,k=5):
        self.handle_progress = HandleProgress()
        self.frame_info = frame_info
        self.username = username
        self.handle_progress.set_status(self.username,"Loading person reidentification model|None")
        self.root_dir = os.path.join(root_dir,username)
        self.k = k
        self.annotated_dir = os.path.join(self.root_dir,"AnnotatedFrame")
        self.video_metadata = os.path.join(self.root_dir,"metadata.json")
        self.video_metadata_v2 = os.path.join(self.root_dir,"metadata_v2.json")
        self.resptective_frame_name = os.path.join(self.annotated_dir,self.frame_info)
        self.query_image_name = query_image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: For Demo, change device to cpu
        self.device = torch.device("cpu")

        self.osnet = torchreid.models.osnet_x1_0(pretrained=False)
        self.osnet.load_state_dict(torch.load("osnet_x1_0_imagenet.pth",map_location=self.device))
        self.handle_progress.set_status(self.username,"Loaded person reidentification model|None")
    
    def get_all_gallery_image(self):
        gallery_image = list()
        query_image = self.query_image_name
        with open(self.video_metadata,"r") as file_obj:
            metadata = json.load(file_obj)
            for frame_metadata in metadata:
                if frame_metadata["annotated_frame_path"] != self.resptective_frame_name+".jpg" \
                       and "annotated_frame_wise_person_info" in list(frame_metadata.keys()) :
                    for person_image in frame_metadata["annotated_frame_wise_person_info"]:
                        gallery_image.append(dict(
                            frame_name = frame_metadata["normal_frame_path"],
                            person_name = person_image["person_image_path"],
                            bounding_box = person_image["bounding_box"]
                        ))
            file_obj.close()
        return query_image,gallery_image
    
    def get_all_gallery_image_v2(self):
        gallery_image = list()
        query_image = self.query_image_name

        with open(self.video_metadata_v2,"r") as file_obj:
            metadata = json.load(file_obj)
            
            for video in metadata.keys():
                if "aux" in video:
                    
                    for frame_metadata in metadata[video]:
                        if "annotated_frame_wise_person_info" in frame_metadata.keys() :
                            
                            for person_image in frame_metadata["annotated_frame_wise_person_info"]:
                                gallery_image.append(dict(
                                    frame_name = frame_metadata["normal_frame_path"],
                                    person_name = person_image["person_image_path"],
                                    bounding_box = person_image["bounding_box"]
                                ))
            
        return query_image,gallery_image

    def prepare_dataset(self,query_image,gallery_image):
        image_transforms = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        query_dataset = image_transforms(Image.open(query_image).convert('RGB'))
        gallery_dataset =  CustomgalleryDataset(gallery_image,image_transforms)
        return query_dataset,gallery_dataset

    def extract_features(self,query_dataset,gallery_dataset,count_gallery):
        query_features = None
        calculated_distance = list()
        self.osnet.eval()
        time_to_extract_start = time.perf_counter()
        with torch.no_grad():
                person_image = query_dataset.to(self.device)
                person_image = torch.unsqueeze(person_image , 0)

                self.osnet = self.osnet.to(self.device)

                person_image = self.osnet(person_image)
                query_features = person_image
        time_to_extract_end = time.perf_counter()

        time_to_match_start = time.perf_counter()
        with torch.no_grad():
            count = 0
            self.handle_progress.set_status(self.username,f"Extracting gallery features|{str(count)}/{str(count_gallery)}")
            for frame_name,person_image_name,bounding_box,person_image in tqdm(gallery_dataset):
                gallery_image_features = person_image.to(self.device)
                # person_image = torch.unsqueeze(person_image , 0)
                # gallery_image_features = self.osnet(person_image)
                
                calculated_distance.append(dict(frame_name = frame_name,
                                    person_image_name = person_image_name,
                                    bounding_box = bounding_box,
                                    distance = cosine_similarity(query_features,gallery_image_features)))
                count += 1
                self.handle_progress.set_status(self.username,f"Extracting gallery features|{str(count)}/{str(count_gallery)}")
        time_to_match_end = time.perf_counter()
        self.handle_progress.set_status(self.username,"Calculating distance|None")
        calculated_distance = sorted(calculated_distance,key = lambda x : x["distance"],reverse=True)
        return calculated_distance[:self.k], (time_to_extract_end-time_to_extract_start), (time_to_match_end-time_to_match_start)
    
    def plot_rectangle(self,top_k):
        prid_result = list()
        for person in top_k:
            image = cv2.imread(person["frame_name"])
            plot_one_box(person["bounding_box"],image,single_frame = True)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_name = pathlib.PurePath(person["person_image_name"]).stem
            byte_arr = io.BytesIO()
            image.save(byte_arr,format='PNG')
            encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
            prid_result.append(dict(image = encoded_img , image_name = image_name))
        return prid_result
    
    def start_prid(self):
        self.handle_progress.set_status(self.username,"Collecting all gallery images|None")
        query_images , gallery_images = self.get_all_gallery_image()
        # # gallery_images = gallery_images[:500]
        print("get query and gallery images ..")
        query_dataset , gallery_dataset = self.prepare_dataset(query_images,gallery_images)
        print("get query and gallery dataset ...")
        top_k, time_to_extract, time_to_match = self.extract_features(query_dataset,gallery_dataset,len(gallery_images))
        # print(top_k)
        print("get query and gallery features ...")
        self.handle_progress.set_status(self.username,"Selecting person in frame|None")
        self.handle_progress.set_status(self.username,"Person reidentification completed, result will be load soon|None")
        self.handle_progress.conn.close()
        return self.plot_rectangle(top_k)
    
    def start_prid_v2(self):
        self.handle_progress.set_status(self.username,"Collecting all gallery images|None")
        
        time_images_start = time.perf_counter()
        query_images , gallery_images = self.get_all_gallery_image_v2()
        # # gallery_images = gallery_images[:500]
        time_images_end = time.perf_counter()
        print("get query and gallery images ..")
        
        query_dataset , gallery_dataset = self.prepare_dataset(query_images,gallery_images)
        print("get query and gallery dataset ...")
        
        
        top_k, time_to_extract, time_to_match = self.extract_features(query_dataset,gallery_dataset,len(gallery_images))
        print(top_k)
        
        print("get query and gallery features ...")
        self.handle_progress.set_status(self.username,"Selecting person in frame|None")
        self.handle_progress.set_status(self.username,"Person reidentification completed, result will be load soon|None")
        self.handle_progress.conn.close()

        prid_benchmark = {
            "time_to_get_images" : round(time_images_end - time_images_start, 4),
            "time_to_extract" : round(time_to_extract, 4),
            "time_to_match" : round(time_to_match, 4)
        }
        return self.plot_rectangle(top_k), prid_benchmark

class HandleProgress(object):
    def __init__(self):
        self.conn = sqlite3.connect("prid_database.db")
    
    def make_new_user_entry(self,username,password,status = "Wait|None"):
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'UserLogin';")
        result = cursor.fetchall()
        
        if len(result) == 0:
            self.conn.execute("CREATE TABLE UserLogin(ID INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT NOT NULL , password TEXT NOT NULL)")
            self.conn.execute("CREATE TABLE StatusTable(username TEXT NOT NULL,status TEXT NOT NULL)")
            self.conn.commit()
        if not self.authorize_user(username,password):
            self.conn.execute("INSERT INTO  UserLogin(username,password) VALUES (?,?)",(str(username),str(password)))
            self.conn.execute("INSERT INTO StatusTable(username,status)VALUES(?,?)",(username,"Wait"))
            self.conn.commit()
        else:
            print("username is already exitst !!")
        cursor.close()

    def authorize_user(self,username,password):
        cursor = self.conn.execute("SELECT ID FROM UserLogin WHERE username = '{0}' AND password = '{1}'".format(str(username),str(password)))
        result = cursor.fetchall()
        if len(result) == 0:
            cursor.close()
            return False
        else:
            cursor.close()
            return True
 
    def make_username_entry(self,username,status = "Wait|None"):
        self.conn.execute(f"UPDATE StatusTable SET status = '{status}' WHERE username = '{username}'")
        self.conn.commit()
    
    
    def get_status(self,username):
        cursor = self.conn.execute("SELECT status from StatusTable WHERE username = '{}';".format(username))
        result = cursor.fetchall()
        cursor.close()
        return result[0][0]
    
    def set_status(self,username,status):
        self.conn.execute(f"UPDATE StatusTable SET status = '{status}' WHERE username = '{username}'")
        self.conn.commit()

# handle_progress = HandleProgress()
# handle_progress.make_new_user_entry('IIITD@Prid','IIITD@Prid#369')

# prid = PersonReidentification("static/uploads","prid","second_1_frameno_0","query_image.png",k=10)
# prid.start_prid()
# video_process = VideoProcess("crowdhuman_yolov5m.pt","static/uploads","prid")
# video_process.read_video()
# video_process.annotate_each_frame()

