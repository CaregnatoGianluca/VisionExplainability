
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
from huggingface_hub import hf_hub_download

# %% [markdown]
# finetuning from https://deepwiki.com/hustvl/Vim/3.2-fine-tuning-vision-mamba

# %%
DEFAULT_BATCH_SIZE   = 32
DEFAULT_BASE_LR      = 1e-5 #much lower to avoid catastrophic forgetting (5e-6)
DEFAULT_EPOCHS       = 600 #new finetuning uses 30 but is too low for a low LR and WD
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-8 #reduced to adapt more
DEFAULT_GPU_ID       = 0
DEFAULT_IMG_SIZE     = 448
DEFAULT_NUM_WORKERS  = 4

MODEL_CHOICES        = ["vim_base_patch16_224"]


net_options = {
    'net_choice': "Mamba",
    'model_choice': MODEL_CHOICES[0],
    'epochs': DEFAULT_EPOCHS,
    'batch_size': DEFAULT_BATCH_SIZE,
    'base_lr': DEFAULT_BASE_LR,
    'weight_decay': DEFAULT_WEIGHT_DECAY,
    'momentum': DEFAULT_MOMENTUM,
    'img_size': DEFAULT_IMG_SIZE,
    'device': torch.device('cuda:'+str(DEFAULT_GPU_ID) if torch.cuda.is_available() else 'cpu'),
    'checkpoint_path': hf_hub_download(repo_id="hustvl/Vim-base-midclstok", filename="vim_b_midclstok_81p9acc.pth"),
    'freeze_params': True,
    'model_type': MODEL_CHOICES[0],
    'save_folder_path': './model_save'
}


cxr_dataset_options = cxr.dataset_options
cub_dataset_options = cub.dataset_options
kdef_dataset_options = kdef.dataset_options

# %%
# --------------------- EDIT THIS TO CHANGE DATASET --------------------- #
DATASET = "kdef" # "cub", "cxr", "kdef"

# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_dataloaders(batchsize=DEFAULT_BATCH_SIZE, root=cxr_dataset_options['data_root'])
    dataset_options = cxr_dataset_options
elif DATASET == "cub":
    train_loader, test_loader = cub.get_dataloaders(batch_size=DEFAULT_BATCH_SIZE, root=cub_dataset_options['data_root'])
    dataset_options = cub_dataset_options
elif DATASET == "kdef":
    train_loader, test_loader = kdef.get_dataloaders(DEFAULT_BATCH_SIZE,
                                             data_dir=kdef_dataset_options['root_dir'],
                                             gaze_dir=kdef_dataset_options['gaze_dir'])
    dataset_options = kdef_dataset_options


print("OPTIONS VALUES")
print(dataset_options)

manager = NetworkManager.NetworkManager(net_options, dataset_options, train_loader, test_loader)

# %%
manager.train()


