import sys
import os
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.nn import init
from collections import OrderedDict
from pytorch_grad_cam.base_cam import BaseCAM

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

EXPANSION = 4


def weight_init_kaiming(m):
    '''
    Kaiming weight initialization
    
    :param m: module
    :return: initialized module
    '''
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)


class ResNet(nn.Module):
    
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        '''
        ResNet model class

        :param pre_trained: bool, whether to use a pre-trained model
        :param n_class: int, number of output classes
        :param model_choice: int, ResNet model choice (50, 101, or 152)
        '''
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(512*EXPANSION, n_class)
        self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        '''
        Forward pass of the model

        :param x: input tensor of shape (N, 3, 448, 448)
        :return: output tensor of shape (N, n_class)
        '''
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        '''
        Selects the ResNet model based on the choice
        :param pre_trained: bool, whether to use a pre-trained model
        :param model_choice: int, ResNet model choice (50, 101, or 152)
        :return: ResNet model instance'''
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)
              
    def load_checkpoint(self, checkpoint_path):
        '''
        Loads model weights from a checkpoint file.
        :param checkpoint_path: str, path to the checkpoint file
        '''
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            #MODEL WEIGHTS LOADING ADAPTS TO DataParallel OR SINGLE GPU MODELS
            # support checkpoints saved as {'state_dict': ...} or plain state_dict
            state_dict = checkpoint.get('state_dict', checkpoint)

            # detect "module." prefix in checkpoint vs current model
            ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())
            model_has_module = any(k.startswith('module.') for k in self.state_dict().keys())

            new_state_dict = OrderedDict()
            if ckpt_has_module and not model_has_module:
                # checkpoint was saved from DataParallel model; remove "module." prefix
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
            elif not ckpt_has_module and model_has_module:
                # checkpoint was saved from single-GPU model but current model is DataParallel; add prefix
                for k, v in state_dict.items():
                    new_state_dict['module.' + k] = v
            else:
                # prefixes already match
                new_state_dict = state_dict

            try:
                self.load_state_dict(new_state_dict)
            except RuntimeError as e:
                # fallback to non-strict load if shapes/keys mismatch
                print("Warning: strict load failed ({}). Retrying with strict=False.".format(e))
                self.load_state_dict(new_state_dict, strict=False)

            # self.solver.load_state_dict(checkpoint['optimizer'])
            #print("=> loaded checkpoint '{}' (epoch {})"
            #      .format(checkpoint_path, new_state_dict['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    


def load_resnet50_checkpoint(checkpoint_path, pre_trained:bool=True, n_class=200, model_choice=50):
    '''
    Loads a ResNet50 model from a checkpoint file.
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        pre_trained (bool): Whether to use a pre-trained model.
        n_class (int): Number of classes for the model.
        model_choice (int): ResNet model choice (50, 101, or 152).
    '''
    model = ResNet(pre_trained=pre_trained, n_class=n_class, model_choice=model_choice)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        model.load_checkpoint(checkpoint_path)
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return model

def wrap_resnet50_cam(model:nn.Module, cam:BaseCAM):
    '''
    Wraps a ResNet50 model with a X-CAM instance.
    Args:
        model (nn.Module): ResNet50 model instance.
        cam (BaseCAM): X-CAM class (e.g., GradCAM, ScoreCAM, etc.)
    Returns:
        BaseCAM: Wrapped X-CAM instance.
    '''
    target_layers = [model.base_model.layer4[-1]]
    cam_instance = cam(model=model, target_layers=target_layers)
    return cam_instance
