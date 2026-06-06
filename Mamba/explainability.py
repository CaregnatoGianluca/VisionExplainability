# %%
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import mamba_ssm
print(f"VERIFICATION - Mamba-SSM location: {mamba_ssm.__file__}")
from mamba_ssm.ops.triton.layer_norm import RMSNorm
print("RMSNorm successfully loaded from custom source!")


from mamba_ssm.modules.mamba_simple import Mamba
def get_divide_out(self):
    # Return the correctly spelled one if it exists, otherwise False
    return getattr(self, 'if_divide_out', False)
Mamba.if_devide_out = property(get_divide_out)

import NetworkManager
import torch
from DatasetLoader import cub_v2 as cub
from DatasetLoader import CXR as cxr
from DatasetLoader import KDEF as kdef
from huggingface_hub import hf_hub_download
from mamba_lrp.lrp.utils import vision_relevance_propagation
from mamba_lrp.model.vision_mamba import ModifiedVisionMamba
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import heatmap_similarity_metrices as hsm
from pytorch_grad_cam.utils.image import show_cam_on_image
import json
import pandas as pd


# %%
import torch
import torch.nn as nn
import mamba_lrp.lrp.core as lrp_core
import mamba_lrp.model.vision_mamba as lrp_model

# 1. Get the working RMSNorm that you said works in your notebook
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    print("Found valid RMSNorm in notebook.")
except ImportError:
    # Fallback to a dummy class if Triton isn't working so isinstance doesn't crash
    class RMSNorm(nn.Module): pass
    print("Using dummy RMSNorm (Triton kernels not found).")

# 2. Inject the valid RMSNorm into the MambaLRP modules
# This fixes the "TypeError: isinstance() arg 2 must be a type"
lrp_core.RMSNorm = RMSNorm
lrp_model.RMSNorm = RMSNorm

# 3. Define and Inject ModifiedLayerNorm
# The library is missing this class, which causes a NameError in the 'else' branch
class ModifiedLayerNorm(nn.Module):
    def __init__(self, norm, zero_bias=False):
        super().__init__()
        self.norm = norm
        if zero_bias and hasattr(self.norm, 'bias') and self.norm.bias is not None:
            with torch.no_grad():
                self.norm.bias.zero_()
    def forward(self, x):
        return self.norm(x)
    def relprop(self, relevance, **kwargs):
        return relevance

lrp_core.ModifiedLayerNorm = ModifiedLayerNorm
lrp_model.ModifiedLayerNorm = ModifiedLayerNorm

# 4. Fix the 'if_devide_out' typo on the Mamba class itself
from mamba_ssm.modules.mamba_simple import Mamba
Mamba.if_devide_out = property(lambda self: getattr(self, 'if_divide_out', False))

print("MambaLRP modules successfully patched.")

# %%
# --------------------- EDIT THIS TO CHANGE DATASET AND MODEL STATE --------------------- #
DATASET = "kdef" # "cub", "cxr", "kdef"
BASE_WEIGHTS_DIR = "../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation"


# %%
DEFAULT_BATCH_SIZE   = 4
DEFAULT_IMG_SIZE     = 448
#dummy values since we are not training
DEFAULT_BASE_LR      = 5e-6 #much lower to avoid catastrophic forgetting (5e-6)
DEFAULT_EPOCHS       = 100 #new finetuning uses 30 but is too low for a low LR and WD
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-8 #reduced to adapt more
DEFAULT_GPU_ID       = 0


MODEL_CHOICES        = ["vim_base_patch16_224"]


USE_PADDING = False


dataset_folder = "CXR_weights" if DATASET == "cxr" else ("CUB_weights" if DATASET == "cub" else "KDEF_weights")
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
    'checkpoint_path': f'{BASE_WEIGHTS_DIR}/Mamba/{dataset_folder}/Mambavim_base_patch16_224.pkl',
    'freeze_params': True,
    'model_type': MODEL_CHOICES[0],
    'save_folder_path': './model_save'
}

output_options = {
    'output_folder_path': './output_heatmaps',
    'save_heatmaps':True,
    'heatmap_save_path': 'heatmaps',
    'save_metrics':True,
    'metrics_filename': 'heatmap_scores.json',
    'save_only_gaze': True,
    'gaze_output_folder_path': './output_gaze',
    'only_gaze_save_path': 'only_gaze'
}

cxr_dataset_options = cxr.dataset_options
cub_dataset_options = cub.dataset_options
kdef_dataset_options = kdef.dataset_options


# %%
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


# %%
def vision_relevance(model, images, labels, n_classes):
    '''
    Compute relevance scores for Mamba models using LRP.
    Args:
        model: ModifiedVisionMamba model
        images: input images (Tensor), shape: (B, C, H, W)
        labels: target labels (Tensor), shape: (B,)
        n_classes: number of classes (int)
    Returns:
        R: relevance scores (Tensor), shape: (B, C, H, W)
        prediction: predicted labels (Tensor), shape: (B,)
        logits: model logits (Tensor), shape: (B, n_classes)

    '''
    # get patch embeddings (Tensor), shape: (B, num_patches, embed_dim)
    embeddings = model.patch_embed(images)

    R, prediction, logits = vision_relevance_propagation(
        model = model,
        embeddings = embeddings,
        targets = labels,
        n_classes = n_classes
    )

    return R, prediction, logits

# %%
def generate_visualization(original_image: torch.Tensor, R):
    '''
    generate heatmap from mamba relevance

    return: \n
            original_heatmap (red spots = pixels useful to the classification, blue spots = pixel confusing the classification, yellow = neutral)\n
            adjusted_heatmap (yellow and blue spots become all blue, to "compare" with other models and explainability methods)
    '''
    attributions = R
    num_patches = attributions.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Il numero di patch ({num_patches}) non è un quadrato perfetto!")
    original_image = original_image.detach().cpu().permute(1, 2, 0).numpy()

    heatmap = attributions.reshape(grid_size, grid_size) # Shape: [28, 28]
    heatmap_tensor = torch.tensor(heatmap).to(net_options['device']).unsqueeze(0).unsqueeze(0)
    upscaled_heatmap = F.interpolate(
        heatmap_tensor,
        size=original_image.shape[:2], # Prende Altezza e Larghezza (es. [224, 224])
        mode='bilinear',
        align_corners=False
    )
    upsc_heat_np = upscaled_heatmap.squeeze().squeeze().detach().cpu().numpy() # Shape: [224, 224]

    upsc_heat_np = (upsc_heat_np - upsc_heat_np.min()) / (upsc_heat_np.max() - upsc_heat_np.min())

    heatmap = upsc_heat_np

    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    overlap_heatmap = show_cam_on_image(img=original_image, mask=upsc_heat_np, use_rgb=True, image_weight=0.5)

    upsc_heat_np = (upsc_heat_np - 0.5)/0.5
    upsc_heat_np[upsc_heat_np<=0] = 0

    adjusted_heatmap = show_cam_on_image(img=original_image, mask=upsc_heat_np, use_rgb=True, image_weight=0.5)

    return heatmap, overlap_heatmap, adjusted_heatmap


# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_exp_dataloaders(batchsize=DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'], use_padding = USE_PADDING)
    dataset_options = cxr_dataset_options
elif DATASET == "cub":
    train_loader, test_loader = cub.get_exp_dataloaders(batch_size=DEFAULT_BATCH_SIZE, root=cub_dataset_options['data_root'], use_padding = USE_PADDING)
    dataset_options = cub_dataset_options
elif DATASET == "kdef":
    gaze_loader = kdef.get_gaze_data_loader(
        batchsize=DEFAULT_BATCH_SIZE,
        data_dir=kdef_dataset_options['gaze_dir'],
        heatmap_dir=kdef_dataset_options['heatmap_dir']
    )
    train_loader, test_loader = None, None
    dataset_options = kdef_dataset_options

print("OPTIONS VALUES")
print(dataset_options)

manager = NetworkManager.NetworkManager(net_options, dataset_options, train_loader, test_loader, mode='eval')

model_lrp = ModifiedVisionMamba(manager.net.module, zero_bias=False)
model_lrp.to(net_options['device'])
model_lrp.eval()

print(net_options['device'])

# %%
dataset_scores_dict = {}

if os.path.exists(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])):
    with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'r') as f:
        dataset_scores_dict = json.load(f)
        print(f"Loaded existing heatmap scores.")
else:
    os.makedirs(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path']), exist_ok=True)
    

# %%
if DATASET == "cub":
    df_img = pd.read_csv(os.path.join(cub_dataset_options['data_root'], 'CUB_200_2011', 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
    df_label = pd.read_csv(os.path.join(cub_dataset_options['data_root'], 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
    df_split = pd.read_csv(os.path.join(cub_dataset_options['data_root'], 'CUB_200_2011', 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
    df = pd.concat([df_img, df_label, df_split], axis=1)
    # relabel
    df['Label'] = df['Label'] - 1

# %% [markdown]
# ## TEST HEATMAPS

# %%
if DATASET == "cub":
    #take only test set
    df_test = df[df['Train']==0]
    df_test_indices = df_test.index.to_list()

    for images, labels, image_indices in test_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        labels = labels.to(net_options['device'])

        
        target_classes = labels
        R, _,_ = vision_relevance(model_lrp, images, target_classes, dataset_options['n_class'])

        for i in range(images.shape[0]):
            current_image_idx_in_dataset = df_test_indices[image_indices[i]]

            gaze_map_path = os.path.join(cub_dataset_options['gaze_map_dir'], "{}.jpg".format(current_image_idx_in_dataset))

            image_filepath = df_test.loc[current_image_idx_in_dataset, 'Image']
            image_filename = os.path.basename(image_filepath)

            print(f"Processing Image: {image_filename}, Gaze Map Path: {gaze_map_path}"
                )
            if not os.path.exists(gaze_map_path):
                print(f"Gaze Map File not found, skipping: {gaze_map_path}")
                continue
            
            heatmap, overlapped_heatmap, adjusted_heatmap = generate_visualization(images[i], R[i, 1:])
            gt_img = Image.open(gaze_map_path).convert("L")

            if USE_PADDING:
                gt_img = pad_to_square(gt_img)
                heatmap = pad_to_square(heatmap)
                overlapped_heatmap = pad_to_square(overlapped_heatmap)
                adjusted_heatmap = pad_to_square(adjusted_heatmap)
            
            gt_img = gt_img.resize((net_options['img_size'], net_options['img_size']))
            gt_img = np.array(gt_img).astype(np.float32)
            gt_img = gt_img / 255.0  # Normalize to [0, 1]

            model_key = net_options['net_choice'] + str(net_options['model_choice'])
            if image_filename in dataset_scores_dict and model_key in dataset_scores_dict[image_filename]:
                print(f"Skipping already processed image: {image_filename}")
                continue

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": False}

            if model_key not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][model_key] = {}
            print(type(gt_img), gt_img.dtype, gt_img.shape)
            scores = hsm.calc_jss_chi2_pcc_scores(heatmap, gt_img)
            dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlapped_heatmap = Image.fromarray(overlapped_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "original", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlapped_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

                adjusted_heatmap = Image.fromarray(adjusted_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "adjusted", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                adjusted_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")
    
elif DATASET == "cxr":
    for images, labels, ids in test_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        labels = labels.to(net_options['device'])

        target_classes = labels
        R, _,_ = vision_relevance(model_lrp, images, target_classes, dataset_options['n_class'])

        for i in range(images.shape[0]):
            id = ids[i]
            gaze_image = cxr.get_gaze_image(id, data_dir=cxr_dataset_options['data_root'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {id}")
                continue

            target_class = labels[i].item()

            heatmap, overlapped_heatmap, adjusted_heatmap = generate_visualization(images[i], R[i, 1:])

            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
                heatmap = pad_to_square(heatmap)
                overlapped_heatmap = pad_to_square(overlapped_heatmap)
                adjusted_heatmap = pad_to_square(adjusted_heatmap)
            
            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            model_key = net_options['net_choice'] + str(net_options['model_choice'])
            if id in dataset_scores_dict and model_key in dataset_scores_dict[id]:
                print(f"Skipping already processed image: {id}")
                continue

            if id not in dataset_scores_dict:
                dataset_scores_dict[id] = {"index": id, "train": False}

            if model_key not in dataset_scores_dict[id]:
                dataset_scores_dict[id][model_key] = {}
            print(type(gaze), gaze.dtype, gaze.shape)
            scores = hsm.calc_jss_chi2_pcc_scores(heatmap, gaze)
            dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlapped_heatmap = Image.fromarray(overlapped_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "original", id+'.jpg')
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlapped_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

                adjusted_heatmap = Image.fromarray(adjusted_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "adjusted", id+'.jpg')
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                adjusted_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

elif DATASET == "kdef":
    for images, labels, ids in gaze_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        labels = labels.to(net_options['device'])

        target_classes = labels
        R, _,_ = vision_relevance(model_lrp, images, target_classes, dataset_options['n_class'])

        for i in range(images.shape[0]):
            idx = ids[i].item()
            image_filepath = gaze_loader.dataset.image_paths[idx]
            image_filename = os.path.basename(image_filepath)

            print(f"Processing Image: {image_filename}, ID: {idx}, Image Path: {image_filepath}")

            gaze_image = kdef.get_gaze_image(original_image_name=image_filename,
                                            heatmap_dir=kdef_dataset_options['heatmap_dir'],
                                            images_dir=kdef_dataset_options['gaze_dir'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {image_filename}")
                continue

            target_class = labels[i].item()

            heatmap, overlapped_heatmap, adjusted_heatmap = generate_visualization(images[i], R[i, 1:])

            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
                heatmap = pad_to_square(heatmap)
                overlapped_heatmap = pad_to_square(overlapped_heatmap)
                adjusted_heatmap = pad_to_square(adjusted_heatmap)

            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            model_key = net_options['net_choice'] + str(net_options['model_choice'])
            if image_filename in dataset_scores_dict and model_key in dataset_scores_dict[image_filename]:
                print(f"Skipping already processed image: {image_filename}")
                continue

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": idx, "train": False}

            if model_key not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][model_key] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap, gaze)
            dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlapped_heatmap = Image.fromarray(overlapped_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "original", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlapped_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

                adjusted_heatmap = Image.fromarray(adjusted_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "adjusted", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                adjusted_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

            if output_options['save_only_gaze']:
                # Save the raw relevance/attention map alone (grayscale, no image overlay)
                only_gaze_img = Image.fromarray(np.uint8(255 * heatmap)).convert("L")
                only_gaze_path = os.path.join(output_options['gaze_output_folder_path'], dataset_options['name'], output_options['only_gaze_save_path'], net_options['net_choice']+str(net_options['model_choice']), "original", image_filename)
                os.makedirs(os.path.dirname(only_gaze_path), exist_ok=True)
                only_gaze_img.save(only_gaze_path)
                print(f"Saved only_gaze to {only_gaze_path}")

# %% [markdown]
# ## TRAIN HEATMAPS

# %%
if DATASET == "cub":
    #take only test set
    df_train = df[df['Train']==1]
    df_train_indices = df_train.index.to_list()

    for images, labels, image_indices in train_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        labels = labels.to(net_options['device'])
        
        target_classes = labels
        R, _,_ = vision_relevance(model_lrp, images, target_classes, dataset_options['n_class'])
        
        for i in range(images.shape[0]):
            current_image_idx_in_dataset = df_train_indices[image_indices[i]]

            gaze_map_path = os.path.join(cub_dataset_options['gaze_map_dir'], "{}.jpg".format(current_image_idx_in_dataset))

            image_filepath = df_test.loc[current_image_idx_in_dataset, 'Image']
            image_filename = os.path.basename(image_filepath)

            print(f"Processing Image: {image_filename}, Gaze Map Path: {gaze_map_path}"
                )
            if not os.path.exists(gaze_map_path):
                print(f"Gaze Map File not found, skipping: {gaze_map_path}")
                continue

            target_class = labels[i].item()
            
            
            heatmap, overlapped_heatmap, adjusted_heatmap = generate_visualization(images[i], R[i, 1:])
            gt_img = Image.open(gaze_map_path).convert("L")

            if USE_PADDING:
                gt_img = pad_to_square(gt_img)
                heatmap = pad_to_square(heatmap)
                overlapped_heatmap = pad_to_square(overlapped_heatmap)
                adjusted_heatmap = pad_to_square(adjusted_heatmap)
            
            gt_img = gt_img.resize((net_options['img_size'], net_options['img_size']))
            gt_img = np.array(gt_img).astype(np.float32)
            gt_img = gt_img / 255.0  # Normalize to [0, 1]

            model_key = net_options['net_choice'] + str(net_options['model_choice'])
            if image_filename in dataset_scores_dict and model_key in dataset_scores_dict[image_filename]:
                print(f"Skipping already processed image: {image_filename}")
                continue

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": False}

            if model_key not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][model_key] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap, gt_img)
            dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlapped_heatmap = Image.fromarray(overlapped_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "original", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlapped_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

                adjusted_heatmap = Image.fromarray(adjusted_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "adjusted", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                adjusted_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")
    
elif DATASET == "cxr":
    for images, labels, ids in train_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        labels = labels.to(net_options['device'])
        
        target_classes = labels
        R, _,_ = vision_relevance(model_lrp, images, target_classes, dataset_options['n_class'])

        for i in range(images.shape[0]):
            id = ids[i]
            gaze_image = cxr.get_gaze_image(id, data_dir=cxr_dataset_options['data_root'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {id}")
                continue

            target_class = labels[i].item()

            R, _,_ = vision_relevance(model_lrp, images[i], target_class, dataset_options['n_class'])
            heatmap, overlapped_heatmap, adjusted_heatmap = generate_visualization(images[i], R[i, 1:])

            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
                heatmap = pad_to_square(heatmap)
                overlapped_heatmap = pad_to_square(overlapped_heatmap)
                adjusted_heatmap = pad_to_square(adjusted_heatmap)
            
            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            model_key = net_options['net_choice'] + str(net_options['model_choice'])
            if id in dataset_scores_dict and model_key in dataset_scores_dict[id]:
                print(f"Skipping already processed image: {id}")
                continue

            if id not in dataset_scores_dict:
                dataset_scores_dict[id] = {"index": id, "train": False}

            if model_key not in dataset_scores_dict[id]:
                dataset_scores_dict[id][model_key] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(adjusted_heatmap, gaze)
            dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlapped_heatmap = Image.fromarray(overlapped_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "original", id+'.jpg')
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlapped_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

                adjusted_heatmap = Image.fromarray(adjusted_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "adjusted", id+'.jpg')
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                adjusted_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

elif DATASET == "kdef":
    for images, labels, ids in gaze_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        labels = labels.to(net_options['device'])

        target_classes = labels
        R, _,_ = vision_relevance(model_lrp, images, target_classes, dataset_options['n_class'])

        for i in range(images.shape[0]):
            idx = ids[i].item()
            image_filepath = gaze_loader.dataset.image_paths[idx]
            image_filename = os.path.basename(image_filepath)

            print(f"Processing Image: {image_filename}, ID: {idx}, Image Path: {image_filepath}")

            gaze_image = kdef.get_gaze_image(original_image_name=image_filename,
                                            heatmap_dir=kdef_dataset_options['heatmap_dir'],
                                            images_dir=kdef_dataset_options['gaze_dir'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {image_filename}")
                continue

            target_class = labels[i].item()

            heatmap, overlapped_heatmap, adjusted_heatmap = generate_visualization(images[i], R[i, 1:])

            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
                heatmap = pad_to_square(heatmap)
                overlapped_heatmap = pad_to_square(overlapped_heatmap)
                adjusted_heatmap = pad_to_square(adjusted_heatmap)

            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            model_key = net_options['net_choice'] + str(net_options['model_choice'])
            if image_filename in dataset_scores_dict and model_key in dataset_scores_dict[image_filename]:
                print(f"Skipping already processed image: {image_filename}")
                continue

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": idx, "train": True}

            if model_key not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][model_key] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap, gaze)
            dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlapped_heatmap = Image.fromarray(overlapped_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "original", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlapped_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")

                adjusted_heatmap = Image.fromarray(adjusted_heatmap)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), "adjusted", image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                adjusted_heatmap.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")


