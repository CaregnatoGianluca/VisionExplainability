# %% [markdown]
# # Training

# %%
import sys
import os

# Add parent directory to path so we can import DatasetLoader
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import torch
from DatasetLoader import cub_v2 as cub
from DatasetLoader import CXR as cxr
from DatasetLoader import KDEF as kdef
import NetworkManager

# %%
DEFAULT_BATCH_SIZE   = 64
DEFAULT_BASE_LR      = 0.001
DEFAULT_EPOCHS       = 95
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GPU_ID       = 0
DEFAULT_IMG_SIZE     = 448

MODEL_CHOICES        = [50, 101, 152]


net_options = {
    'net_choice': "ResNet",
    'model_choice': MODEL_CHOICES[0],
    'epochs': DEFAULT_EPOCHS,
    'batch_size': DEFAULT_BATCH_SIZE,
    'base_lr': DEFAULT_BASE_LR,
    'weight_decay': DEFAULT_WEIGHT_DECAY,
    'momentum': DEFAULT_MOMENTUM,
    'img_size': DEFAULT_IMG_SIZE,
    'device': torch.device('cuda:'+str(DEFAULT_GPU_ID) if torch.cuda.is_available() else 'cpu'),
    'save_folder_path': './model_save'
}


cxr_dataset_options = cxr.dataset_options
cub_dataset_options = cub.dataset_options
kdef_dataset_options = kdef.dataset_options

# %%
# --------------------- EDIT THIS TO CHANGE DATASET --------------------- #
DATASET = "kdef"

# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_dataloaders(DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'])
    dataset_options = cxr_dataset_options
elif DATASET == "cub":
    train_loader, test_loader = cub.get_dataloaders(DEFAULT_BATCH_SIZE,
                                             data_dir=cub_dataset_options['data_root'],
                                             gaze_map_dir=cub_dataset_options['gaze_map_dir'])
    dataset_options = cub_dataset_options
elif DATASET == "kdef":
    train_loader, test_loader = kdef.get_dataloaders(kdef_dataset_options)
    dataset_options = kdef_dataset_options

print("OPTIONS VALUES")
print(dataset_options)

manager = NetworkManager.NetworkManager(net_options, dataset_options, train_loader, test_loader)

# %%
manager.train()


