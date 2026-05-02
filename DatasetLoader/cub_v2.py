import os
import pickle
import tarfile

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, TarIO
from torchvision.transforms import functional as F

dataset_options = {
    'name': 'CUB_200_2011',
    'data_root': '../drive_folder/Bridging Human and Model Attention_ Explainability Analysis of CNN, Mamba, and ViT Architectures with Gaze-Based Validation/CUB/DATASET',
    'gaze_map_dir': '../drive_folder/Bridging Human and Model Attention_ Explainability Analysis of CNN, Mamba, and ViT Architectures with Gaze-Based Validation/CUB/GAZE_DATASET/CUB_GHA',
    'n_class': 200
}


def pad_to_square(img, fill=0):
    '''
    Pad image to make it square
    
    Args:
        img: PIL Image
        fill: pixel fill value for padding
    Returns:
        padded_img: PIL Image'''
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


def get_dataloaders(root, batch_size, img_size = 448, num_workers=0):
    '''
    Get CUB-200-2011 dataloaders with standard transforms
    Args:
        root: root directory of CUB-200-2011 dataset
        batch_size: batch size
        img_size: image size after transforms
        num_workers: number of workers for data loading
    Returns:
        data_loader_train: training dataloader
        data_loader_test: testing dataloader'''
    train_transform_list = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
    ]
    test_transforms_list = [
        transforms.Resize(int(img_size/0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
    ]

    train_data = cub200(root, train=True, transform=transforms.Compose(train_transform_list))
    test_data = cub200(root, train=False, transform=transforms.Compose(test_transforms_list))

    data_loader_train = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=num_workers
                                   )
    data_loader_test = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers
                                  )
    
    return data_loader_train, data_loader_test

'''
Use this dataloader function to retrieve the corrected transformed images for tests
'''
def get_exp_dataloaders(root, batch_size, img_size = 448, num_workers=0, use_padding = True):
    '''
    Get CUB-200-2011 dataloaders with transforms used for explainability experiments
    Args:
        root: root directory of CUB-200-2011 dataset
        batch_size: batch size
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
        #transforms.CenterCrop(options['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    ]

    train_data = cub200(root, train=True, transform=transforms.Compose(transform_list))
    test_data = cub200(root, train=False, transform=transforms.Compose(transform_list))

    data_loader_train = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=num_workers
                                   )
    data_loader_test = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers
                                  )
    
    return data_loader_train, data_loader_test

class cub200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        '''
        CUB-200-2011 dataset loader
        Args:
            root: root directory of CUB-200-2011 dataset
            train: whether to load training set
            transform: torchvision transforms to apply to images
        '''
        super(cub200, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform

        '''
        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()
        '''
        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )
        

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx

    def _check_processed(self):
        assert os.path.isdir(self.root) == True
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _extract(self):
        '''
        Extract CUB-200-2011 dataset from tgz file and save processed train/test files
        '''
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            print('Extract image.txt and train_test_split.txt Error!')
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        print('Finish loading images.txt and train_test_split.txt')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        for _id in range(id2name.shape[0]):
            cnt += 1

            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                print('get image: '+image_path + ' error')
                raise RuntimeError
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                train_cnt += 1
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
            if cnt%1000 == 0:
                print('{} images have been extracted'.format(cnt))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        tar.close()
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))