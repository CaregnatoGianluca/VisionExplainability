import sys
import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from transformers import PretrainedConfig #used by mamba to load default parameters to mimic a transformer (it is never used in the training process, just to not generate errors)
import matplotlib.pyplot as plt
import shutil

from CNN.resnet50 import ResNet
import Mamba.mamba as Mamba

from Transformer.transformer import TransformerModule

class NetworkManager(object):
    def __init__(self, net_options, dataset_options, train_loader, test_loader, checkpoint_path = None, mode='train'):
        self.net_options = net_options
        self.dataset_options = dataset_options
        self.device = net_options['device']
        
        print('Starting to prepare network and data...')

        self.net = nn.DataParallel(self._net_choice(self.net_options['net_choice'], mode)).to(self.device)
        if not self.net_options['net_choice'] == 'Mamba':
            if checkpoint_path:
                print(f'Loading checkpoint from {checkpoint_path}...')
                weights = torch.load(checkpoint_path)
                self.net.load_state_dict(weights)
            
        #self.net.load_state_dict(torch.load('/home/zhangyongshun/se_base_model/model_save/ResNet/backup/epoch120/ResNet50-finetune_fc_cub.pkl'))
        print('Network is as follows:')
        print(self.net)
        #print(self.net.state_dict())
        self.criterion = nn.CrossEntropyLoss()
        
        optimizer_choice = self.net_options.get('optimizer', 'SGD')
        if self.net_options['net_choice'] == 'Mamba' or optimizer_choice == 'AdamW':
            print("Using AdamW optimizer.")
            self.solver = torch.optim.AdamW(
                self.net.parameters(), 
                lr=self.net_options['base_lr'], 
                weight_decay=self.net_options['weight_decay']
            )
        else:
            print("Using SGD optimizer.")
            self.solver = torch.optim.SGD(
                self.net.parameters(), 
                lr=self.net_options['base_lr'], 
                momentum=self.net_options['momentum'], 
                weight_decay=self.net_options['weight_decay']
            )

        scheduler_choice = self.net_options.get('scheduler', 'StepLR')
        if scheduler_choice == 'CosineAnnealingLR':
            print("Using CosineAnnealingLR scheduler.")
            self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.solver, T_max=self.net_options['epochs'], eta_min=1e-6)
        else:
            print("Using StepLR scheduler.")
            self.schedule = torch.optim.lr_scheduler.StepLR(self.solver, step_size=200, gamma=0.1)
        #self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.solver, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4
        #)
        
        #self.train_loader = torch.utils.data.DataLoader(
        #    train_data, batch_size=self.options['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        #)
        #self.test_loader = torch.utils.data.DataLoader(
        #    test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        #)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        epochs  = np.arange(1, self.net_options['epochs']+1)
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
        for epoch in range(self.net_options['epochs']):
            num_correct = 0
            train_loss_epoch = list()
            num_total = 0
            for imgs, labels, _ in self.train_loader:
                self.solver.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                loss = self.criterion(output, labels)
                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
                
                loss.backward()
                #nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.solver.step()
                train_loss_epoch.append(loss.item())

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
                if not os.path.exists(os.path.join(self.net_options['save_folder_path'], self.dataset_options['name'], self.net_options['net_choice'])):
                    os.makedirs(os.path.join(self.net_options['save_folder_path'], self.dataset_options['name'], self.net_options['net_choice']), exist_ok=True)
                torch.save(self.net.state_dict(), os.path.join(self.net_options['save_folder_path'], self.dataset_options['name'], self.net_options['net_choice'], str(self.net_options['model_choice'])+'.pkl'))
                
            print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch+1, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch))
        plt.figure()
        plt.plot(epochs, test_acc, color='r', label='Test Acc')
        plt.plot(epochs, train_acc, color='b', label='Train Acc')

        plt.xlabel('epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.title(self.net_options['net_choice']+str(self.net_options['model_choice']))
        # plt.savefig(self.options['net_choice']+str(self.options['model_choice'])+'.png')

    def _accuracy(self):
        self.net.eval()
        num_total = 0
        num_acc = 0
        with torch.no_grad():
            for imgs, labels, _ in self.test_loader:
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
    
    def _net_choice(self, net_choice, mode):
        if net_choice=='ResNet':
            return ResNet(pre_trained=True, n_class=self.dataset_options['n_class'], model_choice=self.net_options['model_choice'])
        if net_choice=='Mamba':
            import Mamba.mamba as Mamba
            base_model = Mamba.load_model_from_checkpoint(
                checkpoint_path = self.net_options['checkpoint_path'],
                n_class = self.dataset_options['n_class'],
                img_size = self.net_options['img_size'],
                model_type = self.net_options['model_type'],
                freeze = False,
                mode = mode
            )

            base_model.config = PretrainedConfig()

            use_lora = self.net_options.get('use_lora', True)

            if use_lora:
                 base_model = Mamba.apply_lora_to_model(base_model)
            
            return base_model

        if net_choice=='Transformer':
            return TransformerModule(
                pre_trained=True, 
                n_class=self.dataset_options['n_class'], 
                model_choice=self.net_options['model_choice'], 
                img_size = self.net_options['img_size'],
                freeze_backbone=self.net_options.get('freeze_backbone', True)
            )
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


    def evaluate_detailed(self):
        """
        Accuracy, Precision, Recall, F1-Score, AUC and print confusion matrix.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
        
        self.net.eval() 
        all_targets = []
        all_preds_scores = []
        all_preds_labels = []

        #n_class = self.net.module.head.out_features

        self.net.eval()

        print('\n--- Detailed Evaluation ---')
        #print(f'Class number: {n_class}')

        with torch.no_grad():
            for imgs, labels, _ in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(imgs)
                
                # 1. Calcolo Punteggi per AUC (Probabilità Softmax)
                scores = torch.softmax(outputs, dim=1)
                
                # 2. Etichette Previste (Predizioni)
                _, predicted_labels = torch.max(outputs, 1)

                # Accumula risultati su CPU
                all_targets.extend(labels.cpu().numpy())
                all_preds_scores.extend(scores.cpu().numpy())
                all_preds_labels.extend(predicted_labels.cpu().numpy())

        # Conversione in array NumPy
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds_labels)
        y_score = np.array(all_preds_scores)

        # ==========================================================================
        # METRICS
        # ==========================================================================
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1-Score
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # AUC (Area Under the ROC Curve)
        try:
            # Multi-classe OVR (One-vs-Rest)
            auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except ValueError as e:
            print(f"Attenzione: Impossibile calcolare l'AUC. {e}")
            auc = np.nan

        print("--- Performance results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro Avg): {precision_macro:.4f}")
        print(f"Recall (Macro Avg): {recall_macro:.4f}")
        print(f"F1-Score (Macro Avg): {f1_macro:.4f}")
        print(f"AUC (Multi-class OVR): {auc:.4f}")

        conf_matrix = confusion_matrix(y_true, y_pred)
        self.net.train() # train mode

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'auc': auc,
            'confusion_matrix': conf_matrix
        }