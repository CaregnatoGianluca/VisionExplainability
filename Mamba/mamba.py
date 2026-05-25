from Mamba.models_mamba_original import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from Mamba.models_mamba_original import vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from Mamba.models_mamba_original import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from Mamba.models_mamba_original import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from Mamba.models_mamba_original import vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
from huggingface_hub import hf_hub_download
from torchvision import transforms
from torch.utils import data
import numpy as np
import argparse
import torch
from peft import LoraConfig, get_peft_model

#dictionary to map model type strings to functions
model_types = {
    'vim_tiny_patch16_224': vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,
    'vim_tiny_patch16_stride8_224': vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,
    'vim_small_patch16_224': vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,
    'vim_small_patch16_stride8_224': vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,
    'vim_base_patch16_224': vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
}


def load_model_from_checkpoint(checkpoint_path: str, model_type: str, n_class: int, freeze: bool, img_size: int, mode: str):
    '''
    Load a ViM model from a checkpoint, modify the head for n_class classes,
    and optionally freeze the backbone.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model_type (str): Type of the model to load (must be a key in model_types).
        n_class (int): Number of classes for the classification head.
        freeze (bool): Whether to freeze the backbone.
        img_size (int): Input image size for the model.
        mode (str): 'eval' to load for evaluation, 'train' to load for training.
    Returns:
        model (torch.nn.Module): The loaded and modified model.
    '''
    model = model_types.get(model_type)(pretrained=False, img_size=img_size)

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if checkpoint.get('model'):
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
        #remove "module." prefix if present
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}
    
    # interpolate position embedding
    # find positional embedding key in checkpoint (handle prefixes like 'base_model.model.pos_embed' or 'module.base_model.model.pos_embed')
    pos_embed_key = None
    if 'pos_embed' in checkpoint_model:
        pos_embed_key = 'pos_embed'
    else:
        for k in checkpoint_model.keys():
            if k.endswith('pos_embed'):
                pos_embed_key = k
                break
    if pos_embed_key is None:
        raise KeyError("pos_embed not found in checkpoint. Available keys: " + ", ".join(list(checkpoint_model.keys())[:50]))
    pos_embed_checkpoint = checkpoint_model[pos_embed_key]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    # replace the same key we found with the resized positional embedding
    checkpoint_model[pos_embed_key] = new_pos_embed

    model.head = torch.nn.Linear(model.head.in_features, n_class)

    if mode =='eval':
        model.load_state_dict(checkpoint_model, strict=False)
    elif mode == 'train':
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

        if freeze:
            #freeze backbone parameters
            for p in model.parameters():
                p.requires_grad = False

            print("Linear Probe Mode: Backbone frozen, training Head and CLS token only.")

            #unfreeze head parameters
            try:
                model.head.weight.requires_grad = True
                model.head.bias.requires_grad = True
            except AttributeError:
                print('no head found to unfreeze')
            #unfreeze cls_token if present
            if hasattr(model, 'cls_token'):
                model.cls_token.requires_grad = True
            #unfreeze pos_embed if present
            try:
                for p in model.patch_embed.parameters():
                    p.requires_grad = False
            except:
                print('no patch embed')


        #modify head for n_class classes instead of original 1000
        model.head = torch.nn.Linear(model.head.in_features, n_class)
        #init head with xavier uniform
        torch.nn.init.xavier_uniform_(model.head.weight)
        torch.nn.init.constant_(model.head.bias, 0.)
    
    return model


def apply_lora_to_model(model: torch.nn.Module):
    '''
    Apply LoRA to the given model.
    Args:
        model (torch.nn.Module): The model to which LoRA will be applied.
    Returns:
        lora_model (torch.nn.Module): The model with LoRA applied.
    '''
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TARGET_MODULES = [
       'in_proj',
        'x_proj', 
        'dt_proj',
        'x_proj_b',
        'dt_proj_b',
        'out_proj'
    ]
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        #task_type="SEQ_CLS", # Sequence Classification
        target_modules=TARGET_MODULES,
    )
    #apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)
    
    print("Unlocking CLS token, position embedding and head parameters for training...")
    for name, param in lora_model.named_parameters():
        if name in ['base_model.model.cls_token', 'base_model.model.pos_embed', 'base_model.model.head.weight', 'base_model.model.head.bias']:
            param.requires_grad = True
            print(f"Unlocked: {name}")

    lora_model.print_trainable_parameters()
    
    return lora_model