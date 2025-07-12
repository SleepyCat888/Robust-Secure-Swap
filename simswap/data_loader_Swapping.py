import os
import glob
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.num_images = len(loader)
        self.preload()

    def preload(self):
        try:
            self.src_image1, self.src_image2 = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.src_image1, self.src_image2 = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.src_image1  = self.src_image1.cuda(non_blocking=True)
            # self.src_image1  = self.src_image1.sub_(self.mean).div_(self.std)
            self.src_image2  = self.src_image2.cuda(non_blocking=True)
            self.src_image2  = self.src_image2.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        src_image1  = self.src_image1
        src_image2  = self.src_image2
        self.preload()
        return src_image1, src_image2
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

class SwappingDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                    image_dir,
                    img_transform,
                    subffix='jpg',
                    random_seed=1234):
        """Initialize and preprocess the Swapping dataset."""
        self.image_dir      = image_dir
        self.img_transform  = img_transform   
        self.subffix        = subffix
        self.dataset        = []
        self.random_seed    = random_seed
        self.preprocess()
        self.num_images = len(self.dataset)
        print(self.num_images )

    def preprocess(self):
        """Preprocess the Swapping dataset."""
        print("processing Swapping dataset images...")

        temp_path   = os.path.join(self.image_dir,'*/')
        pathes      = glob.glob(temp_path)
        self.dataset = []
        for dir_item in pathes:
            join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
            print("processing %s"%dir_item,end='\r')
            temp_list = []
            for item in join_path:
                temp_list.append(item)
            self.dataset.append(temp_list)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        print('Finished preprocessing the Swapping dataset, total dirs number: %d...'%len(self.dataset))
             
    def __getitem__(self, index):
        """Return two src domain images and two dst domain images."""
        dir_tmp1        = self.dataset[index]
        dir_tmp1_len    = len(dir_tmp1)

        filename1   = dir_tmp1[random.randint(0,dir_tmp1_len-1)]
        filename2   = dir_tmp1[random.randint(0,dir_tmp1_len-1)]
        image1      = self.img_transform(Image.open(filename1))
        image2      = self.img_transform(Image.open(filename2))
        # print(filename1,filename2)
        return image1, image2
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

def GetLoader(  dataset_roots,
                batch_size=16,
                dataloader_workers=8,
                random_seed = 1234
                ):
    """Build and return a data loader."""
        
    num_workers         = dataloader_workers
    data_root           = dataset_roots
    random_seed         = random_seed
    
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    content_dataset = SwappingDataset(
                            data_root, 
                            c_transforms,
                            "jpg",
                            random_seed)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    prefetcher = data_prefetcher(content_data_loader)
    return prefetcher

# def denorm(x):
#     out = (x + 1) / 2
#     return out.clamp_(0, 1)

# self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
# self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
class CustomImageDataset(data.Dataset):
    def __init__(self, ID_num = 16, train_num = 0.8):
        self.main_dir = '/media/coolboy-3/dys/00-dataset/pubfig83'
        self.transform = []
        self.transform.append(T.RandomHorizontalFlip())
        self.transform.append(T.Resize(112))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = T.Compose(self.transform)

        self.ID_num = ID_num
        self.train_num = train_num
        self.classes = os.listdir(self.main_dir)[:self.ID_num]
        self.images = []
        for i, clss in enumerate(self.classes):
            img_num = int(len(os.listdir(os.path.join(self.main_dir, clss))) * self.train_num)
            for _, img in enumerate(os.listdir(os.path.join(self.main_dir, clss))[:img_num]):
                self.images.append((os.path.join(self.main_dir, clss, img), i))

        self.images = self.images * 1000
        self.total_imgs = len(self.images)

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        img_loc, _ = self.images[idx]
        image = Image.open(img_loc)
        tensor_image = self.transform(image)

        return tensor_image

class CustomImageDatasetVal(data.Dataset):
    def __init__(self, ID_num = 16, train_num = 0.8):
        self.main_dir = '/media/coolboy-3/dys/00-dataset/pubfig83'
        self.transform = []
        self.transform.append(T.RandomHorizontalFlip())
        self.transform.append(T.Resize(112))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = T.Compose(self.transform)

        self.ID_num = ID_num
        self.train_num = train_num
        self.classes = os.listdir(self.main_dir)[:self.ID_num]
        self.images = []
        for i, clss in enumerate(self.classes):
            img_num = int(len(os.listdir(os.path.join(self.main_dir, clss))) * self.train_num)
            for _, img in enumerate(os.listdir(os.path.join(self.main_dir, clss))[img_num:]):
                self.images.append((os.path.join(self.main_dir, clss, img), i))

        self.images = self.images
        self.total_imgs = len(self.images)

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        img_loc, _ = self.images[idx]
        image = Image.open(img_loc)
        tensor_image = self.transform(image)

        return tensor_image

import math
class CelebAImageDataset(data.Dataset):
    def __init__(self, ID_num = 128, train_num = 0.8):
        self.main_dir = '/media/coolboy-3/dys/00-dataset/celeba/images'
        self.id_file = '/media/coolboy-3/dys/00-dataset/celeba/identity_CelebA.txt'

        self.ID_num = ID_num
        self.train_num = train_num

        self.transform = []
        self.transform.append(T.CenterCrop(178))
        self.transform.append(T.Resize(112))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = T.Compose(self.transform)

        self.identity_dict = {}
        with open(self.id_file, 'r') as file:
            for line in file:
                image, identity = line.strip().split()
                if identity not in self.identity_dict:
                    self.identity_dict[identity] = []
                self.identity_dict[identity].append(os.path.join(self.main_dir,image))

        self.reduced_identity_dict = {}
        for identity, images in self.identity_dict.items():
            num_to_keep = max(1, math.ceil(len(images) * self.train_num))
            self.reduced_identity_dict[identity] = images[:num_to_keep]

        self.grouped_images = sorted(self.reduced_identity_dict.values(), key=len, reverse=True)[:self.ID_num]    
        print(len(self.grouped_images[-1]))
        self.images = [item for sublist in self.grouped_images for item in sublist]
        print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = self.images[idx]
        image = Image.open(img_loc)
        tensor_image = self.transform(image)

        return tensor_image
    

# ipr_set = CustomImageDataset()
# ipr_loader = data.DataLoader(dataset=ipr_set, drop_last=True, batch_size=32, shuffle=True, num_workers=8)