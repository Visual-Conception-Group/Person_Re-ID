import cv2
from configparser import Interpolation
import os
import torch 
import torchreid
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as transforms
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
class CustomDataset(Dataset):
    def __init__(self,skip_or_not,dataset_type,image_folder,apply_image_transform = None):
        self.skip_or_not = skip_or_not
        self.image_folder = image_folder
        self.apply_image_transform = apply_image_transform
        self.dataset_type = dataset_type
        if self.dataset_type == "query":
            self.list_of_files = [os.path.join(self.image_folder,self.skip_or_not)]
        else:
            self.list_of_files = [os.path.join(self.image_folder,file_name) for file_name in os.listdir(self.image_folder) 
                    if file_name.lower() != self.skip_or_not.lower()]

    def __len__(self):
        return len(self.list_of_files)
    
    def __getitem__(self,index):
        file_name = self.list_of_files[index]
        image_name = Image.open(file_name)
        if self.apply_image_transform:
            image_content = self.apply_image_transform(image_name)
        return (file_name , image_content)


def get_gallery_query_dataset(image_folder,query_image):
    image_transforms = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((256,128),interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    query_dataset = CustomDataset(skip_or_not=query_image,
                                dataset_type="query",
                                image_folder=image_folder,
                                apply_image_transform=image_transforms)
    gallery_dataset = CustomDataset(skip_or_not=query_image,
                                    dataset_type = "gallery",
                                    image_folder= image_folder,
                                    apply_image_transform=image_transforms)
    return query_dataset,gallery_dataset

def extract_features(dataset,model):
    feature_list = list()
    model.eval()
    with torch.no_grad():
        for image_name,image_tensor in tqdm(dataset):
            image_tensor = torch.unsqueeze(image_tensor,dim = 0)
            output = model(image_tensor)
            feature_list.append(dict(image_name = image_name,
                                    image_feature = output))
    return feature_list

def get_top_k_person(query_features,gallery_features,k):
    calculated_distance = list()
    for gallery_feature in gallery_features:
        distance = cosine_similarity(query_features[0]["image_feature"],gallery_feature["image_feature"])
        calculated_distance.append(dict(query_image = query_features[0]["image_name"],
                                    gallery_image = gallery_feature["image_name"],
                                    distance = distance))
    calculated_distance = sorted(calculated_distance,key = lambda x : x["distance"],reverse=True)
    return calculated_distance[:k]

# print("Loading model....")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# osnet = torchreid.models.osnet_x1_0(pretrained=False)
# osnet.load_state_dict(torch.load("osnet_x1_0_imagenet.pth",map_location=device))
# print("preparing Query & Gallery dataset....")
# query_dataset , gallery_dataset = get_gallery_query_dataset(image_folder=os.path.join("Gallery"),
#                         query_image="second_1_frameno_0_person_12.jpg")

# print("Extarcting Query features .......")
# query_features = extract_features(query_dataset,osnet)
# print("Extracting Gallery Features ......")
# gallery_features = extract_features(gallery_dataset,osnet)
# print("Calculating distance between query and gallery images")
# top_k_distance = get_top_k_person(query_features=query_features,gallery_features=gallery_features,k=10)
# print("Query Image : {}".format(top_k_distance[0]["query_image"]))
# image = cv2.imread(top_k_distance[0]["query_image"])
# cv2.imshow("Query Image",image)
# for rank,image in enumerate(top_k_distance):
#     print("Rank : {} , image Name : {}".format(rank+1,image["gallery_image"]))
#     image = cv2.imread(image["gallery_image"])
#     cv2.imshow("Gallery Rank - {}".format(rank+1),image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#detect_person = PersonDetect(weights="crowdhuman_yolov5m.pt")
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
