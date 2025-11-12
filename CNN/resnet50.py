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
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        #init.constant_(m.bias.data, 0.0)


class ResNet(nn.Module):
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(512*EXPANSION, n_class)
        self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)
              
    def load_checkpoint(self, checkpoint_path):
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
    


class NetworkManager(object):
    def __init__(self, options, path, train_data: torch.utils.data.Dataset, test_data: torch.utils.data.Dataset):
        self.options = options
        self.path = path
        self.device = options['device']
        
        print('Starting to prepare network and data...')

        self.net = nn.DataParallel(self._net_choice(self.options['net_choice'])).to(self.device)
        #self.net.load_state_dict(torch.load('/home/zhangyongshun/se_base_model/model_save/ResNet/backup/epoch120/ResNet50-finetune_fc_cub.pkl'))
        print('Network is as follows:')
        print(self.net)
        #print(self.net.state_dict())
        self.criterion = nn.CrossEntropyLoss()
        self.solver = torch.optim.SGD(
            self.net.parameters(), lr=self.options['base_lr'], momentum=self.options['momentum'], weight_decay=self.options['weight_decay']
        )
        self.schedule = torch.optim.lr_scheduler.StepLR(self.solver, step_size=30, gamma=0.1)
        #self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.solver, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4
        #)
        
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.options['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

    def train(self):
        epochs  = np.arange(1, self.options['epochs']+1)
        test_acc = list()
        train_acc = list()
        print('Training process starts:...')
        if torch.cuda.device_count() > 1:
            print('More than one GPU are used...')
        print('Epoch\tTrainLoss\tTrainAcc\tTestAcc')
        print('-'*50)
        best_acc = 0.0
        best_epoch = 0
        self.net.train(True)
        for epoch in range(self.options['epochs']):
            num_correct = 0
            train_loss_epoch = list()
            num_total = 0
            for imgs, labels in self.train_loader:
                self.solver.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                loss = self.criterion(output, labels)
                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
                train_loss_epoch.append(loss.item())
                loss.backward()
                #nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.solver.step()

            train_acc_epoch = num_correct.detach().cpu().numpy()*100 / num_total
            avg_train_loss_epoch  = sum(train_loss_epoch)/len(train_loss_epoch)
            test_acc_epoch = self._accuracy()
            test_acc.append(test_acc_epoch)
            train_acc.append(train_acc_epoch)
            self.schedule.step()
            if test_acc_epoch>best_acc:
                best_acc = test_acc_epoch
                best_epoch = epoch+1
                print('*', end='')
                torch.save(self.net.state_dict(), os.path.join(self.path['model_save'], self.options['net_choice'], self.options['net_choice']+str(self.options['model_choice'])+'.pkl'))
                
            print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch+1, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch))
        plt.figure()
        plt.plot(epochs, test_acc, color='r', label='Test Acc')
        plt.plot(epochs, train_acc, color='b', label='Train Acc')

        plt.xlabel('epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.title(self.options['net_choice']+str(self.options['model_choice']))
        # plt.savefig(self.options['net_choice']+str(self.options['model_choice'])+'.png')

    def _accuracy(self):
        self.net.eval()
        num_total = 0
        num_acc = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                _, pred = torch.max(output, 1)
                num_acc += torch.sum(pred==labels.detach_())
                num_total += labels.size(0)
        return num_acc.detach().cpu().numpy()*100/num_total

    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
    
    def _net_choice(self, net_choice):
        if net_choice=='ResNet':
            return ResNet(pre_trained=True, n_class=200, model_choice=self.options['model_choice'])
        # elif net_choice=='ResNet_ED':
        #     return ResNet_ED(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])
        # elif net_choice == 'ResNet_SE':
        #     return ResNet_SE(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])
        # elif net_choice == 'ResNet_self':
        #     return ResNet_self(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])

    def adjust_learning_rate(optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def load_resnet50_checkpoint(checkpoint_path, pre_trained:bool=True, n_class=200, model_choice=50):
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
