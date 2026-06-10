# form https://raw.githubusercontent.com/ukaukaaaa/GazeGNN/refs/heads/main/read_data.py

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import numpy as np
import os
from glob import glob
from torchvision.transforms import functional as F


dataset_options = {
    'name': 'CXR',
    'data_root': '../drive_folder/CXR/',
    'n_class': 3
}


def pad_to_square(img, fill=0):
    '''
    Pad image to make it square

    Args:
        img: PIL Image
        fill: pixel fill value for padding
    Returns:
        padded_img: PIL Image
    '''
    # img: PIL Image
    w, h = img.size
    if w == h:
        return img
    if w < h:
        diff = h - w
        left = diff // 2
        right = diff - left
        top = bottom = 0
    else:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        left = right = 0
    # padding = (left, top, right, bottom)
    return F.pad(img, (left, top, right, bottom), fill=fill, padding_mode='constant')


def get_gaze_image(id, data_dir = '../CXR'):
    '''
    Get gaze heatmap image for given id
    Args:
        id: image id
        data_dir: root directory of CXR dataset
    Returns:
        gaze_img: PIL Image of gaze heatmap
    '''
    # On CXR there is a folder fixation_:maps and then a fofolder fore each id, then there is the image, return it
    gaze_path = os.path.join(data_dir, "gaze", "fixation_heatmaps", id, "heatmap.png")
    if os.path.exists(gaze_path):
        return Image.open(gaze_path).convert("L")
    return None


def get_dataloaders(batchsize, data_dir = '../CXR', img_size=448, num_workers=0):
    '''
    Get CXR dataloaders with standard transforms
    Args:
        batchsize: batch size
        data_dir: root directory of CXR dataset
        img_size: image size after transforms
        num_workers: number of workers for data loading
    Returns:
        data_loader_train: training dataloader
        data_loader_test: testing dataloader
    '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-5,5)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #re added normalization
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(img_size/0.875)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #re added normalization
        ]),
    }
    
    image_datasets = {x: dataset(mode=x, data_dir=data_dir, transform=data_transforms[x])
                      for x in ['train', 'test']}
    
    data_loader_train = DataLoader(dataset=image_datasets['train'],
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=num_workers
                                   )
    data_loader_test = DataLoader(dataset=image_datasets['test'],
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers
                                  )
    
    return data_loader_train,data_loader_test

def get_exp_dataloaders(batchsize, data_dir = '../CXR', img_size=448, num_workers=0, use_padding = True):
    '''
    Get CXR dataloaders with simple transforms for explainability experiments
    Args:
        batchsize: batch size
        data_dir: root directory of CXR dataset
        img_size: image size after transforms
        num_workers: number of workers for data loading
        use_padding: whether to pad images to square before resizing
    Returns:
        data_loader_train: training dataloader
        data_loader_test: testing dataloader
    '''
    transform_list = [
        transforms.Lambda(lambda img: pad_to_square(img)) if use_padding else transforms.Lambda(lambda img: img),
        transforms.Resize(size=(int(img_size), int(img_size))),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                    std=(0.229, 0.224, 0.225))
    ]

    train_dataset = dataset(mode="train", data_dir=data_dir, transform=transforms.Compose(transform_list))
    test_dataset = dataset(mode="test", data_dir=data_dir, transform=transforms.Compose(transform_list))

    data_loader_train = DataLoader(dataset=train_dataset,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=num_workers
                                   )
    data_loader_test = DataLoader(dataset=test_dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers
                                  )
    
    return data_loader_train,data_loader_test

class dataset(Dataset):
    def __init__(self, data_dir='../CXR', mode="train", transform=None):
        '''
        CXR dataset loader
        Args:
            data_dir: root directory of CXR dataset
            mode: "train" or "test"
            transform: torchvision transforms to apply to images
        '''
        self.root = data_dir
        self.mode = mode
        self.T = transform
        self.csv = pd.read_csv(os.path.join(self.root, "gaze", "fixations.csv"))
        self.labels = ["CHF", "Normal", "pneumonia"]
        self.labelsdict = {"CHF": 0, "Normal": 1, "pneumonia": 2}
        self.idlist = []
        for i in range(len(self.labels)):
            self.idlist.extend(glob(os.path.join(self.root, self.mode, self.labels[i], "*.jpg")))
        
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, idx):
        '''
        Get image, label, id for given index
        Args:
            idx: index of the image to retrieve
        Returns:
            img: transformed image tensor
            label: integer label of the image
            id: image id string
        '''
        # get path
        imgpath = self.idlist[idx]
        id = imgpath.split("/")[-1].split(".jpg")[0]
        # gazepath = os.path.join(self.root, "gaze", "fixations", "{}.npy".format(id))

        # extract image
        with open(imgpath, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")


        # extract label
        label = self.labelsdict[imgpath.split("/")[-2]]

        # transform
        state = torch.get_rng_state()
        img = self.T(img)
        #img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        torch.set_rng_state(state)
        #gaze = self.T(gimg)
        #gaze = self.getPatchGaze(gaze[0])


        return img, label, id

    def getPatchGaze(self, gaze):
        '''
        Get patch-based gaze heatmap from full-resolution gaze heatmap
        Args:
            gaze: full-resolution gaze heatmap tensor
        Returns:
            g: patch-based gaze heatmap numpy array
        '''
        g = np.zeros((56,56), dtype=np.float32)
        for i in range(56):
            for j in range(56):
                x1 = 4*i-7
                x2 = 4*i+7
                y1 = 4*j-7
                y2 = 4*j+7
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > 223:
                    x2 = 223
                if y2 > 223:
                    y2 = 223
                g[i,j] = gaze[x1:x2, y1:y2].sum()
        if g.max()-g.min() != 0:
            g = (g-g.min())/(g.max()-g.min())
        return g
