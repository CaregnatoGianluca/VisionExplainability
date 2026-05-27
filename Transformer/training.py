# %% [markdown]
# # **Transformer Training**


import sys
import os

# Set up the path to include heatmap similarity metrics and dataset loader
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


import torch
import NetworkManager
from DatasetLoader import cub_v2 as cub
from DatasetLoader import CXR as cxr
from DatasetLoader import KDEF as kdef

#import folder 
#from Transformer-Explainability.baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP


# %%
'''
We not use Directly the vit_LRP model for training, we use timm library to load a pre-trained ViT model and then we fine tune it on our dataset. Then, we will use the vit_LRP model for explainability purposes, just loading the model.

TIMM: https://huggingface.co/docs/timm/quickstart

or this...https://huggingface.co/learn/cookbook/fine_tuning_vit_custom_dataset

'''

# %%
DEFAULT_BATCH_SIZE   = 32
DEFAULT_BASE_LR      = 5e-5
DEFAULT_EPOCHS       = 500
DEFAULT_MOMENTUM     = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GPU_ID       = 0
DEFAULT_IMG_SIZE     = 448

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
    'freeze_backbone': False,
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
    train_loader, test_loader = cxr.get_dataloaders(DEFAULT_BATCH_SIZE, data_dir=cxr_dataset_options['data_root'], img_size=net_options['img_size'])
    dataset_options = cxr_dataset_options
elif DATASET == "cub":
    train_loader, test_loader = cub.get_dataloaders(DEFAULT_BATCH_SIZE,
                                             data_dir=cub_dataset_options['data_root'],
                                             gaze_map_dir=cub_dataset_options['gaze_map_dir'],
                                             img_size=net_options['img_size'])
    dataset_options = cub_dataset_options
elif DATASET == "kdef":
    train_loader, test_loader = kdef.get_dataloaders(DEFAULT_BATCH_SIZE,
                                             data_dir=kdef_dataset_options['root_dir'],
                                             gaze_dir=kdef_dataset_options['gaze_dir'])
    dataset_options = kdef_dataset_options

print("OPTIONS VALUES")
print(dataset_options)

manager = NetworkManager.NetworkManager(net_options, dataset_options, train_loader, test_loader, mode='train')

# %%
manager.train()

# %% [markdown]
# # Method 1: Using timm library to load a pre-trained ViT model

# %%
#import timm
#m = timm.create_model('vit_base_patch16_224', pretrained=True)



# %%
#print parameters
#print(m.state_dict().keys())

# %% [markdown]
# # Method 2: using transformer library
# 
# Da https://medium.com/@imabhi1216/fine-tuning-a-vision-transformer-vit-model-with-a-custom-dataset-37840e4e9268

# %%
'''
from transformers import TrainingArguments, Trainer
from transformers import ViTImageProcessor

model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
)

# Get configurations from ViT processor
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
print(image_mean, image_std, size)
# Normalizes the image pixels by subtracting the mean and dividing by the std from the pretrained model configurations
normalize = Normalize(mean=image_mean, std=image_std)

# Compose: Combines a series of image transformations into one pipeline.
train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)
test_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)

def apply_train_transforms(examples):
    examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples


def apply_test_transforms(examples):
    examples["pixel_values"] = [test_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

train_ds.set_transform(apply_train_transforms)
test_ds.set_transform(apply_test_transforms)


def collate_fn(examples):
    # Stacks the pixel values of all examples into a single tensor and collects labels into a tensor
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Create a DataLoader for the training dataset, with custom collation and a batch size of 4
train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)

from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels = len(100),
    ignore_mismatched_sizes=True
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="vit_fine_tuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    load_best_model_at_end=True,
    push_to_hub=False
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    #eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor,
)

# Start training
trainer.train()

outputs = trainer.predict(test_ds)
print(outputs.metrics)
'''

# %% [markdown]
# # Final upload on explainability

# %%
'''
# Just load the vit_LRP model for explainability purposes to check if it works
import torch
vit_model = vit_LRP(pretrained=True)
vit_model.load_state_dict(torch.load('./path/to/vit_base_patch16_224.pth'))
'''


