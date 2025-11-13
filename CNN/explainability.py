#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install grad-cam

# In[2]:


import sys
import os

# Set up the path to include heatmap similarity metrics and dataset loader
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import heatmap_similarity_metrices as hsm
import torch
import os
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
from DatasetLoader.cub_v2 import cub200
import CNN.resnet50 as rn50


MODEL_SAVE_PATH = './model_save'
DATASET_ROOT = '../CUB/DATASET/'
DATASET_IMAGES = os.path.join(DATASET_ROOT, "CUB_200_2011")

GAZE_MAP_DIR = '../CUB/GAZE_DATASET/CUB_GHA'

HEATMAP_SCORES_PATH = '../heatmap_scores'
HEATMAP_SCORES_JSON_PATH = os.path.join(HEATMAP_SCORES_PATH, 'cub.json')

DEFAULT_BATCH_SIZE   = 1
DEFAULT_BASE_LR      = 0.001
DEFAULT_EPOCHS       = 95
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GPU_ID       = 0
DEFAULT_IMG_SIZE     = 448 #448 previously

MODEL_CHOICES        = [50, 101, 152]

options = {
    'net_choice': "ResNet",
    'model_choice': MODEL_CHOICES[0],
    'epochs': DEFAULT_EPOCHS,
    'batch_size': DEFAULT_BATCH_SIZE,
    'base_lr': DEFAULT_BASE_LR,
    'weight_decay': DEFAULT_WEIGHT_DECAY,
    'momentum': DEFAULT_MOMENTUM,
    'img_size': DEFAULT_IMG_SIZE,
    'device': torch.device('cuda:'+str(DEFAULT_GPU_ID) if torch.cuda.is_available() else 'cpu')
}

path = {
    'data': DATASET_ROOT,
    'model_save': MODEL_SAVE_PATH
}


# In[3]:


def print_gpu_memory_usage(stage=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3) # Convert bytes to GiB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # Convert bytes to GiB
        print(f"--- GPU Memory Usage ({stage}) ---")
        print(f"Allocated: {allocated:.2f} GiB")
        print(f"Reserved: {reserved:.2f} GiB")
        # Note: 'free' is not directly exposed as easily as allocated/reserved by PyTorch
        # The total_memory - allocated - cached is a rough estimate
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory:.2f} GiB")
        print(f"Estimated Free: {total_memory - allocated:.2f} GiB (rough estimate)")
        print("-----------------------------------")
    else:
        print("CUDA not available.")


# In[4]:


def pad_to_square(img, fill=0):
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


# In[5]:



# ---- MODEL SETUP ----
model = rn50.load_resnet50_checkpoint(checkpoint_path=os.path.join(MODEL_SAVE_PATH, 'ResNet', 'ResNet50.pkl'), pre_trained=True, n_class=200, model_choice=50)

model.to(options['device'])
model.eval()

print(model)

MODEL_NAME = "ResNet50"

grad_cams = {
    'GradCAM': rn50.wrap_resnet50_cam(model, GradCAM),
    'ScoreCAM': rn50.wrap_resnet50_cam(model, ScoreCAM),
    'AblationCAM': rn50.wrap_resnet50_cam(model, AblationCAM)
}

print(next(model.parameters()).device)

# In[6]:


transform_list = [
    transforms.Lambda(lambda img: pad_to_square(img)),
    transforms.Resize(int(options['img_size'])),
    #transforms.CenterCrop(options['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
]

# In[7]:


def plot_image_and_gaze(rgb_image, gaze_map, cam_images, title=""):
    ''' Plots the original image with gaze map and the CAM images side by side.
    Args:
        rgb_image (numpy.ndarray): The original RGB image as a numpy array.
        gaze_map (numpy.ndarray): The ground truth gaze map as a numpy array.
        cam_images (dict): A dictionary where keys are CAM method names and values are CAM images.
        title (str): Title for the entire plot.
    '''
    f, ax = plt.subplots(1, len(cam_images) + 1, figsize=(15, 5))
    visualization = show_cam_on_image(rgb_image, gaze_map, use_rgb=True, image_weight=0.5)
    ax[0].imshow(visualization)
    ax[0].set_title("Ground Truth Gaze Map")
    ax[0].axis('off')
    for j, (cam_name, cam_output) in enumerate(cam_images.items()):
        # Get the specific CAM image for the current image in the batch
        visualization = show_cam_on_image(rgb_image, cam_output, use_rgb=True, image_weight=0.5)
        ax[j+1].imshow(visualization, cmap='gray')
        ax[j+1].set_title(f"{cam_name}")
        ax[j+1].axis('off')
    plt.show()
    plt.close(f)

# In[8]:


dataset_scores_dict = {}

if os.path.exists(HEATMAP_SCORES_JSON_PATH):
    with open(HEATMAP_SCORES_JSON_PATH, 'r') as f:
        dataset_scores_dict = json.load(f)
        print(f"Loaded existing heatmap scores.")
else:
    os.makedirs(HEATMAP_SCORES_PATH, exist_ok=True)
    

# In[9]:


df_img = pd.read_csv(os.path.join(DATASET_IMAGES, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
df_label = pd.read_csv(os.path.join(DATASET_IMAGES, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
df_split = pd.read_csv(os.path.join(DATASET_IMAGES, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
df = pd.concat([df_img, df_label, df_split], axis=1)
# relabel
df['Label'] = df['Label'] - 1

# ## TEST SET HEATMAPS

# In[10]:


print("TEST DATASET")

test_data = cub200(path['data'], train=False, transform=transforms.Compose(transform_list))

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=options['batch_size'], shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available()
)

# In[11]:


#take only test set
df_test = df[df['Train']==0]
df_test_indices = df_test.index.to_list()

#print_gpu_memory_usage("BEFORE INFERENCE")

for images, labels, image_indices in test_loader:
    torch.cuda.empty_cache()

    images = images.to(options['device'])
    #labels = labels.to(options['device'])
    batch_cam_images = {cam_name: [] for cam_name in grad_cams.keys()}
    #print_gpu_memory_usage("BATCH {} LOADED".format(image_indices))
    # ---- GRAD-CAM ----
    targets = [ClassifierOutputTarget(label.item()) for label in labels]

    cam_images = {}
    for cam_name, cam in grad_cams.items():
        grayscale_cam_batch = cam(input_tensor=images, targets=targets)
        batch_cam_images[cam_name].extend(grayscale_cam_batch)
    #print_gpu_memory_usage("AFTER GRAD-CAM INFERENCE")

    for i in range(images.shape[0]):
        current_image_idx_in_dataset = df_test_indices[image_indices[i]]

        gaze_map_path = os.path.join(GAZE_MAP_DIR, "{}.jpg".format(current_image_idx_in_dataset))

        image_filepath = df_test.loc[current_image_idx_in_dataset, 'Image']
        image_filename = os.path.basename(image_filepath)

        print(f"Processing Image: {image_filename}, Gaze Map Path: {gaze_map_path}"
              )
        if not os.path.exists(gaze_map_path):
            print(f"Gaze Map File not found, skipping: {gaze_map_path}")
            continue

        target_class = labels[i].item()

        # ---- CONVERT IMAGE TO RGB (0-1) for visualization ----
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = images[i].permute(1, 2, 0).cpu().numpy()
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)

        # ---- GAZE MAP ----
        gt_img = Image.open(gaze_map_path).convert("L")
        gt_img = pad_to_square(gt_img)
        gt_img = gt_img.resize((options['img_size'], options['img_size']))
        
        gt_img = np.array(gt_img).astype(np.float32)
        gt_img = gt_img / 255.0  # Normalize to [0, 1]

        if image_filename not in dataset_scores_dict:
            dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": False}

        if MODEL_NAME not in dataset_scores_dict[image_filename]:
            dataset_scores_dict[image_filename][MODEL_NAME] = {}

        curr_cam_images = {cam_name: batch_cam_images[cam_name][i] for cam_name in batch_cam_images}
        for cam_name, cam_image in curr_cam_images.items():
            scores = hsm.calc_jss_chi2_pcc_scores(cam_image, gt_img)
            #print(f"Scores for {cam_name}: {scores}")
            dataset_scores_dict[image_filename][MODEL_NAME][cam_name] = scores

        with open(HEATMAP_SCORES_JSON_PATH, 'w') as f:
            json.dump(dataset_scores_dict, f, indent=4)
            print(f"Saved heatmap scores to {HEATMAP_SCORES_JSON_PATH}")

        # ---- VISUALIZATION ----
        #plot_image_and_gaze(rgb_img, np.array(gt_img).astype(np.float32), curr_cam_images, title=f"Image ID: {current_image_idx_in_dataset}, Class: {target_class}")

# Save the dataset scores dictionary to JSON
with open(HEATMAP_SCORES_JSON_PATH, 'w') as f:
    json.dump(dataset_scores_dict, f, indent=4)
    print(f"Saved heatmap scores to {HEATMAP_SCORES_JSON_PATH}")


# ## TRAIN SET HEATMAPS

# In[ ]:


print("TRAIN DATASET")

train_data = cub200(path['data'], train=True, transform=transforms.Compose(transform_list))

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=options['batch_size'], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available()
)

# In[ ]:


#take only train set
df_train = df[df['Train']==1]
df_train_indices = df_train.index.to_list()

#print_gpu_memory_usage("BEFORE INFERENCE")

for images, labels, image_indices in train_loader:
    torch.cuda.empty_cache()

    images = images.to(options['device'])
    #labels = labels.to(options['device'])
    batch_cam_images = {cam_name: [] for cam_name in grad_cams.keys()}
    #print_gpu_memory_usage("BATCH {} LOADED".format(image_indices))
    # ---- GRAD-CAM ----
    targets = [ClassifierOutputTarget(label.item()) for label in labels]

    cam_images = {}
    for cam_name, cam in grad_cams.items():
        grayscale_cam_batch = cam(input_tensor=images, targets=targets)
        batch_cam_images[cam_name].extend(grayscale_cam_batch)
    #print_gpu_memory_usage("AFTER GRAD-CAM INFERENCE")

    for i in range(images.shape[0]):
        current_image_idx_in_dataset = df_train_indices[image_indices[i]]

        gaze_map_path = os.path.join(GAZE_MAP_DIR, "{}.jpg".format(current_image_idx_in_dataset))

        image_filepath = df_train.loc[current_image_idx_in_dataset, 'Image']
        image_filename = os.path.basename(image_filepath)

        print(f"Processing Image: {image_filename}, Gaze Map Path: {gaze_map_path}"
              )
        if not os.path.exists(gaze_map_path):
            print(f"Gaze Map File not found, skipping: {gaze_map_path}")
            continue

        target_class = labels[i].item()

        # ---- CONVERT IMAGE TO RGB (0-1) for visualization ----
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = images[i].permute(1, 2, 0).cpu().numpy()
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)

        # ---- GAZE MAP ----
        gt_img = Image.open(gaze_map_path).convert("L")
        gt_img = pad_to_square(gt_img)
        gt_img = gt_img.resize((options['img_size'], options['img_size']))
        
        gt_img = np.array(gt_img).astype(np.float32)
        gt_img = gt_img / 255.0  # Normalize to [0, 1]

        if image_filename not in dataset_scores_dict:
            dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": True}

        if MODEL_NAME not in dataset_scores_dict[image_filename]:
            dataset_scores_dict[image_filename][MODEL_NAME] = {}

        curr_cam_images = {cam_name: batch_cam_images[cam_name][i] for cam_name in batch_cam_images}
        for cam_name, cam_image in curr_cam_images.items():
            scores = hsm.calc_jss_chi2_pcc_scores(cam_image, gt_img)
            print(f"Scores for {cam_name}: {scores}")
            dataset_scores_dict[image_filename][MODEL_NAME][cam_name] = scores

        with open(HEATMAP_SCORES_JSON_PATH, 'w') as f:
            json.dump(dataset_scores_dict, f, indent=4)
            print(f"Saved heatmap scores to {HEATMAP_SCORES_JSON_PATH}")

        # ---- VISUALIZATION ----
        #plot_image_and_gaze(rgb_img, np.array(gt_img).astype(np.float32), curr_cam_images, title=f"Image ID: {current_image_idx_in_dataset}, Class: {target_class}")

# Save the dataset scores dictionary to JSON
with open(HEATMAP_SCORES_JSON_PATH, 'w') as f:
    json.dump(dataset_scores_dict, f, indent=4)
    print(f"Saved heatmap scores to {HEATMAP_SCORES_JSON_PATH}")

