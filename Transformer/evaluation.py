# %% [markdown]
# # Evaluation

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
DEFAULT_BATCH_SIZE   = 1
DEFAULT_IMG_SIZE     = 448
#dummy values since we are not training
DEFAULT_BASE_LR      = 5e-5
DEFAULT_EPOCHS       = 95
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GPU_ID       = 0


MODEL_CHOICES        = ['vit_base_patch16_224']

net_options = {
    'net_choice': "Transformer",
    'model_choice': MODEL_CHOICES[0],
    'epochs': DEFAULT_EPOCHS,
    'batch_size': DEFAULT_BATCH_SIZE,
    'base_lr': DEFAULT_BASE_LR,
    'weight_decay': DEFAULT_WEIGHT_DECAY,
    'momentum': DEFAULT_MOMENTUM,
    'img_size': DEFAULT_IMG_SIZE,
    'device': torch.device('cuda:'+str(DEFAULT_GPU_ID) if torch.cuda.is_available() else 'cpu'),
    'model_type': MODEL_CHOICES[0],
    'save_folder_path': './model_save'
}


cxr_dataset_options = cxr.dataset_options
cub_dataset_options = cub.dataset_options
kdef_dataset_options = kdef.dataset_options

# %%
# --------------------- EDIT THIS TO CHANGE DATASET AND MODEL STATE --------------------- #
DATASET = "kdef" # "cub", "cxr", "kdef"
FROZEN = False   # True for Frozen weights, False for Unfrozen weights
BASE_WEIGHTS_DIR = "../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation/Transformer"

# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_dataloaders(batchsize=DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'], img_size=net_options['img_size'])
    dataset_options = cxr_dataset_options
    dataset_folder = "CXR_weights"
elif DATASET == "cub":
    train_loader, test_loader = cub.get_dataloaders(batch_size=DEFAULT_BATCH_SIZE, root=cub_dataset_options['data_root'], img_size=net_options['img_size'])
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

state_str = "Frozen" if FROZEN else "Unfrozen"
checkpoint_name = f'vit_base_patch16_224_{state_str}.pkl'
checkpoint_path = os.path.join(BASE_WEIGHTS_DIR, dataset_folder, checkpoint_name)
print(f"Loading checkpoint from {checkpoint_path}...")

manager = NetworkManager.NetworkManager(net_options, dataset_options, train_loader, test_loader, mode='train', checkpoint_path=checkpoint_path)

# %%
stats = manager.evaluate_detailed()

# %%
from sklearn.metrics import ConfusionMatrixDisplay
#show confusion matrix
disp = ConfusionMatrixDisplay(stats['confusion_matrix'])
disp.plot()

import matplotlib.pyplot as plt
plt.show()


