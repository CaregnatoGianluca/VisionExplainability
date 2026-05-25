# %%
import sys
import os

# Add parent directory to path so we can import DatasetLoader
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import torch
from DatasetLoader import cub_v2 as cub
from DatasetLoader import CXR as cxr
from DatasetLoader import KDEF as kdef
from huggingface_hub import hf_hub_download

import NetworkManager

# %%
DEFAULT_BATCH_SIZE   = 64
DEFAULT_IMG_SIZE     = 448
#dummy values since we are not training
DEFAULT_BASE_LR      = 5e-6 #much lower to avoid catastrophic forgetting (5e-6)
DEFAULT_EPOCHS       = 100 #new finetuning uses 30 but is too low for a low LR and WD
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-8 #reduced to adapt more
DEFAULT_GPU_ID       = 0


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
    'freeze_params': True,
    'model_type': MODEL_CHOICES[0],
    'save_folder_path': './model_save'
}

cxr_dataset_options = cxr.dataset_options
cub_dataset_options = cub.dataset_options
kdef_dataset_options = kdef.dataset_options

# %%
# --------------------- EDIT THIS TO CHANGE DATASET AND MODEL STATE --------------------- #
DATASET = "kdef" # "cub", "cxr", "kdef"
BASE_WEIGHTS_DIR = "../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation/Mamba"

# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_dataloaders(DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'])
    dataset_options = cxr_dataset_options
    dataset_folder = "CXR_weights"
elif DATASET == "cub":
    train_loader, test_loader = cub.get_dataloaders(batch_size=DEFAULT_BATCH_SIZE, root=cub_dataset_options['data_root'])
    dataset_options = cub_dataset_options
    dataset_folder = "CUB_weights"
elif DATASET == "kdef":
    train_loader, test_loader = kdef.get_dataloaders(DEFAULT_BATCH_SIZE,
                                             data_dir=kdef_dataset_options['root_dir'],
                                             gaze_dir=kdef_dataset_options['gaze_dir'])
    dataset_options = kdef_dataset_options
    dataset_folder = "KDEF_weights"

print("OPTIONS VALUES")
print(dataset_options)

checkpoint_name = 'Mambavim_base_patch16_224.pkl'
checkpoint_path = os.path.join(BASE_WEIGHTS_DIR, dataset_folder, checkpoint_name)
net_options['checkpoint_path'] = checkpoint_path

manager = NetworkManager.NetworkManager(net_options, dataset_options, train_loader, test_loader, mode='eval', checkpoint_path=net_options['checkpoint_path'])

# %%
stats = manager.evaluate_detailed()

# %%
from sklearn.metrics import ConfusionMatrixDisplay
#show confusion matrix
disp = ConfusionMatrixDisplay(stats['confusion_matrix'])
disp.plot()

import matplotlib.pyplot as plt
plt.show()


