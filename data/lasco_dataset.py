from torch.utils.data import Dataset
import json, os, random
import PIL 
from PIL import Image
from utils import get_image
import torch.nn.functional as F
Image.MAX_IMAGE_PIXELS = 2300000000
 
home_path = "/path/LaSCo"
        
class LaSCoDataset(Dataset):

    def __init__(self, split: str, mode: str, preprocess: callable, llava: bool):

        self.lasco_path_prefix = home_path
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.llava = llava

        if split not in ['train', 'val']:
            raise ValueError("split should be in ['train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']") 
        # get triplets made by (reference_image, target_image, relative caption)
        self.type = "llava"
        with open(f'{self.lasco_path_prefix}/{self.split}_llavasco.json') as f:
            self.triplets = json.load(f)[:350000:70]

        # get a mapping from image name to relative path
        self.image_folder_path = f'{self.lasco_path_prefix}'
        
        with open(f'{self.lasco_path_prefix}/lasco_{self.split}_corpus.json') as f:
            imgs_info = json.load(f)

        # Coprus
        #self.name_list = [img_info['id'] for img_info in imgs_info] #[:40000]
        if self.split == "val": 
            self.name_list = [img_info['id'] for img_info in imgs_info]
        else:
            self.name_list = list(imgs_info.keys())
        print(f"LaSCo {split} dataset in {mode} mode initialized")
 
    def __getitem__(self, index):
        # Format image_id
        def image_id2name(image_id):
            return f"{self.split}2014/COCO_{self.split}2014_" + str(image_id).zfill(12) + ".jpg"
        try:
            if self.mode == 'relative':
                reference_image_id = self.triplets[index]["reference_image_id"]
                reference_image_path = os.path.join(self.lasco_path_prefix, image_id2name(reference_image_id))

                #print(reference_image_id)
                target_image_id = self.triplets[index]["target_image_id"]
                target_image_path = os.path.join(self.lasco_path_prefix, image_id2name(target_image_id))

                # Read and preprocess images
                reference_image = self.preprocess(get_image(reference_image_path)) #.unsqueeze(0) #.to(device)
                reference_image = F.normalize(reference_image, dim = -1)
                relative_caption = self.triplets[index]["relative_caption"]
                target_image = self.preprocess(get_image(target_image_path)) #.unsqueeze(0) #.to(device)
                target_image = F.normalize(target_image, dim = -1)
                reference_caption = self.triplets[index][f"{self.type}_reference_caption"].split(".")[0]
                target_caption = self.triplets[index][f"{self.type}_target_caption"]
                
                if self.split == "val":
                    reference_image_name = reference_image_id 
                    rel_caption = relative_caption + " with " + target_caption 
                    target_image_name = target_image_id  
                    return reference_image_name, target_image_name, rel_caption, reference_image, reference_caption

                if self.llava == True:
                    relative_caption += ' with ' + target_caption
                    #print(relative_caption) 

                return reference_image, relative_caption, target_image, reference_caption, target_caption #reference_image_id, target_image_id
            
            elif self.mode == 'classic':
                image_name = self.name_list[index]
                image_path = f"{self.image_folder_path}/" + image_id2name(image_name)
                image = self.preprocess(get_image(image_path))
                image = F.normalize(image, dim = -1)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_list)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
