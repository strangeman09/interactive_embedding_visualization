import torch
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from tqdm import tqdm
from PIL import Image
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

transform1 = transforms.Compose([           
                                transforms.Resize(520),
                                transforms.CenterCrop(518),                
                                transforms.ToTensor(),                    
                                transforms.Normalize(mean=0.5, std=0.2)
                                ])

parser = argparse.ArgumentParser(description="Extract features from images using DINOv2 model")
parser.add_argument('folder_path', type=str, help='Path to the folder containing image subfolders')


args = parser.parse_args()

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
features = {}

handle=model.head.register_forward_hook(get_features('feats'))

folder_path = args.folder_path

data_dict = []

with torch.no_grad():
    for class_folder in (os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for img_name in tqdm(os.listdir(class_path)):
                captured_output = None
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img_t = transform1(img)

             
                with torch.no_grad():
                    model(img_t.unsqueeze(0))

              
                data_dict.append({
                    'file_path': img_path,
                    'class_label': class_folder,
                    'embeddings': features['feats']
                })

handle.remove()

for i in range(len(data_dict)):
    data_dict[i]['embeddings'] = data_dict[i]['embeddings'].detach().numpy()
    data_dict[i]['embeddings'] = data_dict[i]['embeddings'].flatten()

for item in data_dict:
    item["embeddings"] = item["embeddings"].tolist()


json_data = json.dumps(data_dict, indent=4)

write_path = folder_path.split('/')[-1] + '.json'

with open(write_path, 'w') as json_file:
    json_file.write(json_data)


