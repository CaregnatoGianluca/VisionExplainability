# %%
#!pip install grad-cam

# %%
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
from DatasetLoader import cub_v2 as cub
from DatasetLoader import CXR as cxr
from DatasetLoader import KDEF as kdef
import NetworkManager

# %%
DEFAULT_BATCH_SIZE   = 1
DEFAULT_IMG_SIZE     = 448
#dummy values since we are not training the model here
DEFAULT_BASE_LR      = 0.001
DEFAULT_EPOCHS       = 95
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GPU_ID       = 0


MODEL_CHOICE        = 50


USE_PADDING = False


net_options = {
    'net_choice': "ResNet",
    'model_choice': MODEL_CHOICE,
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
DATASET = "kdef"  # Options: "cxr" or "cub" or "kdef"
BASE_WEIGHTS_DIR = "../drive_folder/Bridging_Human_and_Model_Attention_Explainability_Analysis_of_CNN_Mamba_and_ViT_Architectures_with_Gaze-Based_Validation"


# %%
if DATASET == "cxr":
    train_loader, test_loader = cxr.get_exp_dataloaders(DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'], use_padding=USE_PADDING)
    dataset_options = cxr_dataset_options
elif DATASET == "cub":
    train_loader, test_loader = cub.get_exp_dataloaders(batch_size=DEFAULT_BATCH_SIZE, root=cub_dataset_options['data_root'], img_size=DEFAULT_IMG_SIZE, use_padding=USE_PADDING)
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

#create network manager and load the pretrained model
if DATASET == "cxr":
    dataset_folder = "CXR_weights"  
elif DATASET == "cub":
    dataset_folder = "CUB_weights"
else:
    dataset_folder = "KDEF_weights"

manager = NetworkManager.NetworkManager(net_options, dataset_options, train_loader, test_loader, checkpoint_path=f"{BASE_WEIGHTS_DIR}/CNN/{dataset_folder}/ResNet50.pkl")
model = manager.net.module


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
import resnet50 as rn50

#wrap the model with different CAM methods
grad_cams = {
    'GradCAM': rn50.wrap_resnet50_cam(model, GradCAM),
    'ScoreCAM': rn50.wrap_resnet50_cam(model, ScoreCAM),
    'AblationCAM': rn50.wrap_resnet50_cam(model, AblationCAM)
}

print(next(model.parameters()).device)

# %%
#load existing heatmap scores if available
dataset_scores_dict = {}

if os.path.exists(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])):
    with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'r') as f:
        dataset_scores_dict = json.load(f)
        print(f"Loaded existing heatmap scores.")
else:
    os.makedirs(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path']), exist_ok=True)
    

# %%
if DATASET == "cub":
    #load CUB dataset annotations
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
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
        batch_cam_images = {cam_name: [] for cam_name in grad_cams.keys()}
        # ---- GRAD-CAM ----
        targets = [ClassifierOutputTarget(label.item()) for label in labels]
        
        #result of grad-cam for each method
        cam_images = {}
        for cam_name, cam in grad_cams.items():
            #apply CAM method
            grayscale_cam_batch = cam(input_tensor=images, targets=targets)
            batch_cam_images[cam_name].extend(grayscale_cam_batch)

        for i in range(images.shape[0]):
            #parse gaze path
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
            #load gaze img
            gt_img = Image.open(gaze_map_path).convert("L")
            if USE_PADDING:
                gt_img = pad_to_square(gt_img)
            gt_img = gt_img.resize((net_options['img_size'], net_options['img_size']))
            gt_img = np.array(gt_img).astype(np.float32)
            gt_img = gt_img / 255.0  # Normalize to [0, 1]

            #add entry to dataset scores dict if not present
            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": False}
            #add entry for current model for image if not present
            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = {}

            curr_cam_images = {cam_name: batch_cam_images[cam_name][i] for cam_name in batch_cam_images}
            for cam_name, cam_image in curr_cam_images.items():
                #compute heatmap similarity scores
                scores = hsm.calc_jss_chi2_pcc_scores(cam_image, gt_img)
                dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])][cam_name] = scores

            if output_options['save_metrics']:
                # Save heatmap scores
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                # Save CAM images
                for cam_name, cam_image in curr_cam_images.items():
                    #normalize image to [0,1]
                    
                    images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
                    heatmap = show_cam_on_image(images[i].permute(1, 2, 0).cpu().numpy(), cam_image, use_rgb=True, image_weight=0.5)
                    heatmap_img = Image.fromarray(heatmap)
                    heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), cam_name, image_filename)
                    os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                    heatmap_img.save(heatmap_save_path)
                    print(f"Saved heatmap to {heatmap_save_path}")
    
elif DATASET == "cxr":
    for images, labels, ids in test_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
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
            id = ids[i]
            gaze_image = cxr.get_gaze_image(id, data_dir=cxr_dataset_options['data_root'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {id}")
                continue
            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            target_class = labels[i].item()

            if id not in dataset_scores_dict:
                dataset_scores_dict[id] = {"index": id, "train": False}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[id]:
                dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = {}

            curr_cam_images = {cam_name: batch_cam_images[cam_name][i] for cam_name in batch_cam_images}
            for cam_name, cam_image in curr_cam_images.items():
                scores = hsm.calc_jss_chi2_pcc_scores(cam_image, gaze)
                #print(f"Scores for {cam_name}: {scores}")
                dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])][cam_name] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                # Save CAM images
                for cam_name, cam_image in curr_cam_images.items():
                    images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
                    heatmap = show_cam_on_image(images[i].permute(1, 2, 0).cpu().numpy(), cam_image, use_rgb=True, image_weight=0.5)
                    heatmap_img = Image.fromarray(heatmap)
                    heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), cam_name, id+'.jpg')
                    os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                    heatmap_img.save(heatmap_save_path)
                    print(f"Saved heatmap to {heatmap_save_path}")

elif DATASET == "kdef":
    for images, labels, ids in gaze_loader:
            torch.cuda.empty_cache()

            images = images.to(net_options['device'])
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
                idx = ids[i].item()
                image_filepath = gaze_loader.dataset.image_paths[idx]
                image_filename = os.path.basename(image_filepath)

                print(f"Processing Image: {image_filename}, ID: {idx}, Image Path: {image_filepath}") 

                gaze_image = kdef.get_gaze_image(original_image_name = image_filename, 
                                                heatmap_dir = kdef_dataset_options['heatmap_dir'], 
                                                images_dir = kdef_dataset_options['gaze_dir'])
                if gaze_image is None:
                    print(f"Gaze Map File not found, skipping: {image_filename}")
                    continue
                if USE_PADDING:
                    gaze_image = pad_to_square(gaze_image)
                gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
                gaze = np.array(gaze_image).astype(np.float32)
                gaze = gaze / 255.0  # Normalize to [0, 1]

                target_class = labels[i].item()

                # Usa image_filename come chiave per i dizionari e i salvataggi
                if image_filename not in dataset_scores_dict:
                    dataset_scores_dict[image_filename] = {"index": idx, "train": False}

                if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[image_filename]:
                    dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = {}

                curr_cam_images = {cam_name: batch_cam_images[cam_name][i] for cam_name in batch_cam_images}
                for cam_name, cam_image in curr_cam_images.items():
                    scores = hsm.calc_jss_chi2_pcc_scores(cam_image, gaze)
                    #print(f"Scores for {cam_name}: {scores}")
                    dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])][cam_name] = scores

                if output_options['save_metrics']:
                    with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                        json.dump(dataset_scores_dict, f, indent=4)
                        print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'],  output_options['metrics_filename'])}")
                if output_options['save_heatmaps']:
                    # Save CAM images
                    for cam_name, cam_image in curr_cam_images.items():
                        images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
                        heatmap = show_cam_on_image(images[i].permute(1, 2, 0).cpu().numpy(), cam_image, use_rgb=True, image_weight=0.5)
                        heatmap_img = Image.fromarray(heatmap)
                        heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), cam_name, image_filename+'.jpg')
                        os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                        heatmap_img.save(heatmap_save_path)
                        print(f"Saved heatmap to {heatmap_save_path}")




# %% [markdown]
# ## TRAIN SET HEATMAPS

# %%
if DATASET == "cub":
    #take only test set
    df_train = df[df['Train']==1]
    df_train_indices = df_train.index.to_list()

    #print_gpu_memory_usage("BEFORE INFERENCE")

    for images, labels, image_indices in train_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
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

            gaze_map_path = os.path.join(cub_dataset_options['gaze_map_dir'], "{}.jpg".format(current_image_idx_in_dataset))

            image_filepath = df_train.loc[current_image_idx_in_dataset, 'Image']
            image_filename = os.path.basename(image_filepath)

            print(f"Processing Image: {image_filename}, Gaze Map Path: {gaze_map_path}"
                )
            if not os.path.exists(gaze_map_path):
                print(f"Gaze Map File not found, skipping: {gaze_map_path}")
                continue

            target_class = labels[i].item()

            # ---- GAZE MAP ----
            gt_img = Image.open(gaze_map_path).convert("L")
            if USE_PADDING:
                gt_img = pad_to_square(gt_img)
            gt_img = gt_img.resize((net_options['img_size'], net_options['img_size']))
            
            gt_img = np.array(gt_img).astype(np.float32)
            gt_img = gt_img / 255.0  # Normalize to [0, 1]

            if image_filename not in dataset_scores_dict:
                dataset_scores_dict[image_filename] = {"index": current_image_idx_in_dataset, "train": True}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[image_filename]:
                dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])] = {}

            curr_cam_images = {cam_name: batch_cam_images[cam_name][i] for cam_name in batch_cam_images}
            for cam_name, cam_image in curr_cam_images.items():
                scores = hsm.calc_jss_chi2_pcc_scores(cam_image, gt_img)
                #print(f"Scores for {cam_name}: {scores}")
                dataset_scores_dict[image_filename][net_options['net_choice']+str(net_options['model_choice'])][cam_name] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                # Save CAM images
                for cam_name, cam_image in curr_cam_images.items():
                    images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
                    heatmap = show_cam_on_image(images[i].permute(1, 2, 0).cpu().numpy(), cam_image, use_rgb=True, image_weight=0.5)
                    heatmap_img = Image.fromarray(heatmap)
                    heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), cam_name, image_filename)
                    os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                    heatmap_img.save(heatmap_save_path)
                    print(f"Saved heatmap to {heatmap_save_path}")
            
elif DATASET == "cxr":
    for images, labels, ids in train_loader:
        torch.cuda.empty_cache()

        images = images.to(net_options['device'])
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
            id = ids[i]
            gaze_image = cxr.get_gaze_image(id, data_dir=cxr_dataset_options['data_root'])
            if gaze_image is None:
                print(f"Gaze Map File not found, skipping: {id}")
                continue
            if USE_PADDING:
                gaze_image = pad_to_square(gaze_image)
            gaze_image = gaze_image.resize((net_options['img_size'], net_options['img_size']))
            gaze = np.array(gaze_image).astype(np.float32)
            gaze = gaze / 255.0  # Normalize to [0, 1]

            target_class = labels[i].item()

            if id not in dataset_scores_dict:
                dataset_scores_dict[id] = {"index": id, "train": True}

            if net_options['net_choice']+str(net_options['model_choice']) not in dataset_scores_dict[id]:
                dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])] = {}

            curr_cam_images = {cam_name: batch_cam_images[cam_name][i] for cam_name in batch_cam_images}
            for cam_name, cam_image in curr_cam_images.items():
                scores = hsm.calc_jss_chi2_pcc_scores(cam_image, gaze)
                #print(f"Scores for {cam_name}: {scores}")
                dataset_scores_dict[id][net_options['net_choice']+str(net_options['model_choice'])][cam_name] = scores

            if output_options['save_metrics']:
                with open(os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename']), 'w') as f:
                    json.dump(dataset_scores_dict, f, indent=4)
                    print(f"Saved heatmap scores to {os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['metrics_filename'])}")
            if output_options['save_heatmaps']:
                # Save CAM images
                for cam_name, cam_image in curr_cam_images.items():
                    images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
                    heatmap = show_cam_on_image(images[i].permute(1, 2, 0).cpu().numpy(), cam_image, use_rgb=True, image_weight=0.5)
                    heatmap_img = Image.fromarray(heatmap)
                    heatmap_save_path = os.path.join(output_options['output_folder_path'], dataset_options['name'], output_options['heatmap_save_path'], net_options['net_choice']+str(net_options['model_choice']), cam_name, id+'.jpg')
                    os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
                    heatmap_img.save(heatmap_save_path)
                    print(f"Saved heatmap to {heatmap_save_path}")


