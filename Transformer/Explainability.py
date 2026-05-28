# %%
import sys
import os
# Set up the path to include heatmap similarity metrics and dataset loader
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'Transformer-Explainability')))

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
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
from DatasetLoader import cub_v2 as cub
from DatasetLoader import CXR as cxr
from DatasetLoader import KDEF as kdef
import NetworkManager
import cv2



# %%
DEFAULT_BATCH_SIZE   = 64
DEFAULT_IMG_SIZE     = 448
#dummy values since we are not training
DEFAULT_BASE_LR      = 0.001
DEFAULT_EPOCHS       = 95
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GPU_ID       = 0


MODEL_CHOICES        = ['vit_base_patch16_224']


USE_PADDING = False


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
    'save_folder_path': './model_save'
}

output_options = {
    'output_folder_path': './output_heatmaps',
    'save_heatmaps':True,
    'heatmap_save_path': 'heatmaps',
    'save_metrics':True,
    'metrics_filename': 'heatmap_scores.json'
}


cxr_dataset_options = cxr.dataset_options
cub_dataset_options = cub.dataset_options
kdef_dataset_options = kdef.dataset_options

# %%
# --------------------- EDIT THIS TO CHANGE DATASET AND MODEL STATE --------------------- #
DATASET = "kdef"  # Options: "cxr", "cub" or "kdef"
FROZEN = False   # True for Frozen weights, False for Unfrozen weights
BASE_WEIGHTS_DIR = "../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation"


# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_exp_dataloaders(DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'], use_padding=USE_PADDING)
    dataset_options = cxr_dataset_options
elif DATASET == "cub":
    train_loader, test_loader = cub.get_exp_dataloaders(batch_size=DEFAULT_BATCH_SIZE,
                                             root=cub_dataset_options['data_root'],
                                             img_size=DEFAULT_IMG_SIZE,
                                             use_padding=USE_PADDING)
    dataset_options = cub_dataset_options
elif DATASET == "kdef":
    gaze_loader = kdef.get_gaze_data_loader(batchsize=DEFAULT_BATCH_SIZE,
                                            data_dir=kdef_dataset_options['gaze_dir'],
                                            img_size=DEFAULT_IMG_SIZE,
                                            heatmap_dir=kdef_dataset_options['heatmap_dir'])
    train_loader, test_loader = None, None
    dataset_options = kdef_dataset_options

print("OPTIONS VALUES")
print(dataset_options)

vit_model = vit_LRP(pretrained=False,
                num_classes=dataset_options['n_class'],
                img_size = net_options['img_size'],
                )


dataset_folder = "CXR_weights" if DATASET == "cxr" else ("CUB_weights" if DATASET == "cub" else "KDEF_weights")
state_str = "Frozen" if FROZEN else "Unfrozen"
weights = torch.load(
    f'{BASE_WEIGHTS_DIR}/Transformer/{dataset_folder}/vit_base_patch16_224_{state_str}.pkl',
    map_location=net_options['device']
)
print(weights.keys())
#remove "module.vit." from the keys
new_weights = {}
for k, v in weights.items():
    if k.startswith("module.vit."):
        new_k = k[len("module.vit."):]
        if "head.0" in k:
            new_k = new_k.replace("head.0", "head")
    else:

        new_k = k

    new_weights[new_k] = v

print(weights['module.vit.head.0.weight'])
print(new_weights.keys())

# %%
vit_model.load_state_dict(new_weights)
vit_model.eval()
attribution_generator = LRP(vit_model)

vit_model.to(net_options['device'])

# %%
def generate_visualization(original_image, class_index=None, use_thresholding=False):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).to(net_options['device']), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 28, 28)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(448, 448).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
      transformer_attribution = transformer_attribution * 255
      transformer_attribution = transformer_attribution.astype(np.uint8)
      ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    return image_transformer_attribution, transformer_attribution

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
# ## TEST SET HEATMAPS

# %%
if DATASET == "cub":
    #take only test set
    df_test = df[df['Train']==0]
    df_test_indices = df_test.index.to_list()

    for images, labels, image_indices in test_loader:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        
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

            target_class = labels[i].item()

            orig_img, heatmap_img = generate_visualization(images[i], class_index=target_class, use_thresholding=False)
            overlap_img = show_cam_on_image(orig_img, heatmap_img, use_rgb=True, image_weight=0.5)

            plt.imshow(heatmap_img)
            plt.show()
            
            gt_img = Image.open(gaze_map_path).convert("L")

            if USE_PADDING:
                gt_img = pad_to_square(gt_img)
                heatmap_img = pad_to_square(heatmap_img)
                overlap_img = pad_to_square(overlap_img)
            
            gt_img = gt_img.resize((net_options['img_size'], net_options['img_size']))
            gt_img = np.array(gt_img).astype(np.float32)
            gt_img = gt_img / 255.0  # Normalize to [0, 1]

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": False}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap_img, gt_img)
            dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlap_img = Image.fromarray(overlap_img)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlap_img.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")
    
elif DATASET == "cxr":
    for images, labels, ids in test_loader:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        
        for i in range(images.shape[0]):
            id = ids[i]
            gaze_image = cxr.get_gaze_image(id, data_dir=cxr_dataset_options['data_root'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {id}")
                continue

            target_class = labels[i].item()

            orig_img, heatmap_img = generate_visualization(images[i], class_index=target_class, use_thresholding=False)
            overlap_img = show_cam_on_image(orig_img, heatmap_img, use_rgb=True, image_weight=0.5)

            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
                heatmap_img = pad_to_square(heatmap_img)
                overlap_img = pad_to_square(overlap_img)

            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            if id not in dataset_scores_dict:
                dataset_scores_dict[id] = {"index": id, "train": False}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[id]:
                dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap_img, gaze)
            dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:                
                overlap_img = Image.fromarray(overlap_img)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), id+'.jpg')
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlap_img.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")


elif DATASET == "kdef":
    for images, labels, ids in gaze_loader:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        
        for i in range(images.shape[0]):
            idx = ids[i].item()
            image_filepath = gaze_loader.dataset.image_paths[idx]
            image_filename = os.path.basename(image_filepath)

            print(f"Processing Image: {image_filename}, ID: {idx}")
            gaze_image = kdef.get_gaze_image(original_image_name=image_filename,
                                            heatmap_dir=kdef_dataset_options['heatmap_dir'],
                                            images_dir=kdef_dataset_options['gaze_dir'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {image_filename}")
                continue

            target_class = labels[i].item()

            orig_img, heatmap_img = generate_visualization(images[i], class_index=target_class, use_thresholding=False)
            overlap_img = show_cam_on_image(orig_img, heatmap_img, use_rgb=True, image_weight=0.5)

            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
                heatmap_img = pad_to_square(heatmap_img)
                overlap_img = pad_to_square(overlap_img)

            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": idx, "train": False}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap_img, gaze)
            dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlap_img = Image.fromarray(overlap_img)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlap_img.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")



# %% [markdown]
# ## TRAIN SET HEATMAPS

# %%
if DATASET == "cub":
    #take only test set
    df_train = df[df['Train']==1]
    df_train_indices = df_train.index.to_list()

    for images, labels, image_indices in train_loader:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        
        for i in range(images.shape[0]):
            current_image_idx_in_dataset = df_train_indices[image_indices[i]]

            gaze_map_path = os.path.join(cub_dataset_options['gaze_map_dir'], "{}.jpg".format(current_image_idx_in_dataset))

            image_filepath = df_train.loc[current_image_idx_in_dataset, 'Image']
            image_filename = os.path.basename(image_filepath)

            print(f"Processing Image: {image_filename}, Gaze Map Path: {gaze_map_path}"
                )
            if not os.path.exists(gaze_map_path):
                print(f"Gaze Map File not found, skipping: {gaze_map_path}")
                continue

            target_class = labels[i].item()

            orig_img, heatmap_img = generate_visualization(images[i], class_index=target_class, use_thresholding=False)
            overlap_img = show_cam_on_image(orig_img, heatmap_img, use_rgb=True, image_weight=0.5)
            gt_img = Image.open(gaze_map_path).convert("L")

            if USE_PADDING:
                gt_img = pad_to_square(gt_img)
                heatmap_img = pad_to_square(heatmap_img)
                overlap_img = pad_to_square(overlap_img)
            
            gt_img = gt_img.resize((net_options['img_size'], net_options['img_size']))
            gt_img = np.array(gt_img).astype(np.float32)
            gt_img = gt_img / 255.0  # Normalize to [0, 1]

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": True}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap_img, gt_img)
            dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                overlap_img = Image.fromarray(overlap_img)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), image_filename)
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlap_img.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")
    
elif DATASET == "cxr":
    for images, labels, ids in train_loader:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        
        for i in range(images.shape[0]):
            id = ids[i]
            gaze_image = cxr.get_gaze_image(id, data_dir=cxr_dataset_options['data_root'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {id}")
                continue

            target_class = labels[i].item()

            orig_img, heatmap_img = generate_visualization(images[i], class_index=target_class, use_thresholding=False)
            overlap_img = show_cam_on_image(orig_img, heatmap_img, use_rgb=True, image_weight=0.5)

            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
                heatmap_img = pad_to_square(heatmap_img)
                overlap_img = pad_to_square(overlap_img)
                
            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            if id not in dataset_scores_dict:
                dataset_scores_dict[id] = {"index": id, "train": True}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[id]:
                dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = {}

            scores = hsm.calc_jss_chi2_pcc_scores(heatmap_img, gaze)
            dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:                
                overlap_img = Image.fromarray(overlap_img)
                heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), id+'.jpg')
                os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                overlap_img.save(heatmap_save_path)
                print(f"Saved heatmap to {heatmap_save_path}")




