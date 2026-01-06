import torch
import timm


#From https://github.com/wuhanstudio/timm-tutorial/blob/main/timm_vit_train.py
#model_choice = 'vit_base_patch16_224'

class TransformerModule(torch.nn.Module):

    def __init__(self, n_class, model_choice = 'vit_base_patch16_224', pre_trained = True, in_chans=3, img_size=224, freeze_backbone=True):
        '''
        Initialize the Transformer model.
        Args:
            n_class (int): Number of output classes.
            model_choice (str): Model architecture from timm library.
            pre_trained (bool): Whether to use pre-trained weights.
            in_chans (int): Number of input channels.
            img_size (int): Size of the input images.
            freeze_backbone (bool): Whether to freeze the backbone layers.
        '''
        super(TransformerModule, self).__init__()

        self.vit = timm.create_model(model_choice, pretrained = pre_trained, in_chans=in_chans, img_size=img_size)

        # Change the classifier
        num_in_features = self.vit.get_classifier().in_features

        self.vit.head = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(num_in_features),
            # torch.nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features=num_in_features, out_features=n_class, bias=True),
            #torch.nn.Softmax(dim=1)
        )

        # Alternatively
        # model.reset_classifier(10, 'max')
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

            # cls_token is a Parameter, not a module; set requires_grad directly
            self.vit.cls_token.requires_grad = True
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.vit(x)