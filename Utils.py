import cv2
import math
import os
import random
from matplotlib import pyplot as plt
class ExtractFrame(object):
    
    def __init__(self,file_name,save_dir,frame_per_second):
        self.file_name = file_name
        self.root_save_dir = save_dir 
        self.frame_per_second = frame_per_second
        self. cap = cv2.VideoCapture(self.file_name)
        self.frame_rate = math.floor(self. cap.get(cv2.CAP_PROP_FPS))
        self.total_no_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_duration = math.floor(self.total_no_frames) / math.floor(self.frame_rate)
    def frame_extract(self):
        self.root_save_dir = os.path.join(self.root_save_dir)
        if not os.path.exists(self.root_save_dir):
            os.makedirs(self.root_save_dir)
        self.frame_per_second = self.frame_per_second if self.frame_rate >= self.frame_per_second  else self.frame_rate 
        frame_no = 0
        second_count = 1
        second_wise_frame_list = list()
        while self. cap.isOpened():
            ret, frame = self. cap.read()
            if not ret:
                break
            if(frame_no < self.frame_rate ):
                second_wise_frame_list.append(frame)
                frame_no += 1
            else:
                random_frame_selection = random.sample(second_wise_frame_list,k = self.frame_per_second)
                for index,frame_content in enumerate(random_frame_selection):
                    print("second_{}_image_{}.jpg processing...".format(second_count,index))
                    file_path = os.path.join(self.root_save_dir,"second_{}_frameno_{}.jpg".format(second_count,index))
                    cv2.imwrite(file_path,frame_content) 
                frame_no = 0
                second_count += 1
                random_frame_selection = list()
        self. cap.release()
        cv2.destroyAllWindows() 
        self.no_of_frame_extract = len(os.listdir(self.root_save_dir))

def extract_second_wise_image(folder_name,second_id):
    folder_name = os.path.join(folder_name)
    if os.path.exists(folder_name):
        second_id = "second_{}".format(str(second_id))
        list_of_files = os.listdir(folder_name)
        selected_files = list()
        for file_name in list_of_files:
            if file_name.split("frameno")[0][:-1].lower() == second_id.lower():
                selected_files.append(file_name)
        return [os.path.join(folder_name,file_name)for file_name in selected_files]
    else:
        return list()


def cropped_bounding_box_image():
    pass


file_names = extract_second_wise_image("framedir",10)

# extract_frame = ExtractFrame("video.mp4","framedir",1)
# extract_frame.frame_extract()

