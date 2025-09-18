from torch.utils.data import Dataset
from typing import List
import json 
import PIL 

home_path = "/path/dtin"

class DTINDataset(Dataset):

    def __init__(self, split: str, domain_type: List[str], mode: str, preprocess: callable):
       
        self.fiq_path_prefix = home_path
        self.mode = mode
        self.domain_type = domain_type
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['train', 'val']")
    
        self.all_domains = ['cartoon', 'origami', 'toy', 'sculpture']
        self.target_domain = self.domain_type 
        self.domain_id = self.all_domains.index(self.target_domain)

        self.preprocess = preprocess

        # get queries
        self.queries = open(f"{home_path}/imgnet_real_query.txt").readlines()

        # get the image names and captions
        self.targets = open(f"{home_path}/imgnet_targets.txt").readlines()

        print(f"DTIN dataset finished!")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                query = self.queries[index]
                #image_caption = "this {} object and same shape"
                image_caption = "the similar image with {} style"
                if self.split == 'val':
                    reference_image_path, class_id = query.split()
                    #print(self.domain_id)
                    reference_image_path = home_path + "/" + reference_image_path  
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.domain_id * 1000 + int(class_id) 
                    image_caption = image_caption.format(self.target_domain)
                    return class_id, target_name, image_caption, reference_image, ""

            elif self.mode == 'classic':
                target = self.targets[index]
                image_path, iid = target.split()
                image_path = home_path + "/" + image_path 
                image = self.preprocess(PIL.Image.open(image_path))
                return int(iid), image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.queries)
        elif self.mode == 'classic':
            return len(self.targets)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
