from torchvision import transforms
from torch.utils import data
from PIL import Image, ImageOps
import pandas as pd
import torch, os
import numpy as np

class CUB(data.Dataset):
    def __init__(self, root, dataset_type='train', train_ratio=1, valid_seed=123, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        df_img = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
        df_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
        df_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
        df = pd.concat([df_img, df_label, df_split], axis=1)
        # relabel
        df['Label'] = df['Label'] - 1

        # split data
        if dataset_type == 'test':
            df = df[df['Train'] == 0]
        elif dataset_type == 'train' or dataset_type == 'valid':
            df = df[df['Train'] == 1]
            # random split train, valid
            if train_ratio != 1:
                np.random.seed(valid_seed)
                indices = list(range(len(df)))
                np.random.shuffle(indices)
                split_idx = int(len(indices) * train_ratio) + 1
            elif dataset_type == 'valid':
                raise ValueError('train_ratio should be less than 1!')
            if dataset_type == 'train':
                df = df.iloc[indices[:split_idx]]
            else:    # dataset_type == 'valid'
                df = df.iloc[indices[split_idx:]]
        else:
            raise ValueError('Unsupported dataset_type!')
            
        self.img_name_list = df['Image'].tolist()
        self.label_list = df['Label'].tolist()
        self.id_list = df.index.tolist() 
        
        # Convert greyscale images to RGB mode
        self._convert2rgb()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.img_name_list[idx])
        with Image.open(img_path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            target = self.label_list[idx]
            image_id = self.id_list[idx]
            
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                target = self.target_transform(target)
            
            if not isinstance(image, torch.Tensor) or image.shape[0] != 3:
                raise ValueError(f"Image {img_path} has wrong shape after transform: {image.shape if isinstance(image, torch.Tensor) else 'not a tensor'}")
            
            return image, target, image_id
    

    def _convert2rgb(self):
        for i, img_name in enumerate(self.img_name_list):
            img_path = os.path.join(self.root, 'images', img_name)
            try:
                with Image.open(img_path) as image:
                    if image.mode != 'RGB':
                        # Convert and save back if not RGB
                        image_rgb = image.convert('RGB')
                        image_rgb.save(img_path)
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")