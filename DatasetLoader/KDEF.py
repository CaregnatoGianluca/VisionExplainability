import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split

dataset_options = {
    'name': 'KDEF',
    'root_dir': '../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation/KDEF/dataset/',
    'n_class': 7,
    'gaze_dir': '../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation/KDEF/karolinska',
    'heatmap_dir': '../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation/KDEF/DensityByExpressionAndImage_GPU_sigma20',
}

def pad_to_square(img, fill=0):
    '''
    Pad image to make it square
    
    Args:
        img: PIL Image
        fill: pixel fill value for padding
    Returns:
        padded_img: PIL Image'''
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
    return F.pad(img, (left, top, right, bottom), fill=fill, padding_mode='constant')


class KDEFDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx

def get_dataloaders(batchsize, data_dir = '../KDEF', img_size=448, num_workers=0, gaze_dir = "../karolinska"):
    root_path = data_dir
    img_size = img_size
    batch_size = batchsize
    num_workers = num_workers

    all_image_paths = []
    all_labels = []
    emotion_to_idx = {'HA': 0, 'SA': 1, 'NE': 2, 'AF': 3, 'SU': 4, 'DI': 5, 'AN': 6}

    subjects = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    
    for subj in subjects:
        subj_path = os.path.join(root_path, subj)
        for filename in os.listdir(subj_path):
            if filename.endswith(".JPG"):
                emotion_code = filename[4:6]
                if emotion_code in emotion_to_idx:
                    all_image_paths.append(os.path.join(subj_path, filename))
                    all_labels.append(emotion_to_idx[emotion_code])

    gaze_filenames = set()
    if os.path.exists(gaze_dir):
        print(f"Looking for gaze heatmaps in {gaze_dir}...")
        for root, _, files in os.walk(gaze_dir):
            for f in files:
                gaze_filenames.add(f)

    if gaze_filenames:
        print(f"Found {len(gaze_filenames)} gaze heatmaps. Ensuring they are in the test set.")
        gaze_paths, gaze_labels = [], []
        other_paths, other_labels = [], []
        
        for path, label in zip(all_image_paths, all_labels):
            if os.path.basename(path) in gaze_filenames:
                gaze_paths.append(path)
                gaze_labels.append(label)
            else:
                other_paths.append(path)
                other_labels.append(label)
                
        target_test_size = int(len(all_image_paths) * 0.2)
        remaining_test_size = target_test_size - len(gaze_paths)
        
        if 0 < remaining_test_size < len(other_paths):
            try:
                train_paths, add_test_paths, train_labels, add_test_labels = train_test_split(
                    other_paths, other_labels, test_size=remaining_test_size, 
                    random_state=42, stratify=other_labels
                )
                print(f"Stratified split successful. Added {len(add_test_paths)} non-gaze samples to test set.")
            except ValueError:
                print("Stratified split failed due to insufficient samples in some classes. Performing non-stratified split.")
                train_paths, add_test_paths, train_labels, add_test_labels = train_test_split(
                    other_paths, other_labels, test_size=remaining_test_size, 
                    random_state=42
                )
            test_paths = gaze_paths + add_test_paths
            test_labels = gaze_labels + add_test_labels
        else:
            train_paths = other_paths
            train_labels = other_labels
            test_paths = gaze_paths
            test_labels = gaze_labels
    else:
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_image_paths, 
            all_labels, 
            test_size=0.2, 
            random_state=42,
            stratify=all_labels
        )

    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda img: pad_to_square(img)),
            transforms.RandomRotation((-5, 5)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Lambda(lambda img: pad_to_square(img)), 
            transforms.Resize(int(img_size / 0.875)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = KDEFDataset(train_paths, train_labels, transform=data_transforms['train'])
    test_dataset = KDEFDataset(test_paths, test_labels, transform=data_transforms['test'])
    print(batch_size)
    print(train_dataset.__len__())
    print(train_paths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              pin_memory=True, num_workers=num_workers)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader

def get_gaze_data_loader(batchsize, data_dir = '../karolinska', img_size=448, num_workers=0):
    all_image_paths = []
    all_labels = []
    emotion_to_idx = {'HA': 0, 'SA': 1, 'NE': 2, 'AF': 3, 'SU': 4, 'DI': 5, 'AN': 6}

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".JPG"):
            emotion_code = filename[4:6]
            if emotion_code in emotion_to_idx:
                all_image_paths.append(os.path.join(data_dir, filename))
                all_labels.append(emotion_to_idx[emotion_code])

    data_transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_square(img)), 
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = KDEFDataset(all_image_paths, all_labels, transform=data_transform)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, 
                        pin_memory=True, num_workers=num_workers)

    return loader

def get_gaze_image(original_image_name, images_dir, heatmap_dir):
    '''
    Get gaze heatmap image
    Args:
        original_image_name: original image filename
        images_dir: directory containing the original images
    Returns:
        gaze_img: PIL Image of gaze heatmap
    '''    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".JPG")])

    print(f"Looking for original image {original_image_name} in {images_dir}...")
    print(f"Hitting gaze heatmap directory: {heatmap_dir}...")
    
    if original_image_name not in image_files:
        print (f"Original image {original_image_name} not found in {images_dir}. Cannot load gaze heatmap.")
        return None
        
    index = image_files.index(original_image_name)
    emotion_code = original_image_name[4:6]

    heatmap_dir = heatmap_dir + '/' + emotion_code

    if os.path.exists(heatmap_dir):
        npy_files = sorted([f for f in os.listdir(heatmap_dir) if f.endswith(".npy")])
        if index < len(npy_files):
            gaze_path = os.path.join(heatmap_dir, npy_files[index])
            data = np.load(gaze_path)       
            if data.max() - data.min() != 0:
                data = (data - data.min()) / (data.max() - data.min()) * 255
            else:
                data = data * 0         
            return Image.fromarray(data.astype(np.uint8)).convert("L")
    
    return None