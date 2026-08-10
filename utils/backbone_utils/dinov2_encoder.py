import torch
import torch.nn as nn
from torchvision import transforms
from typing import Union, List, Tuple
from PIL import Image

class dinov2_featurizer(nn.Module):
    """
        DINOv2 featurizer :
            Latest implementation - uses the output activation  for further implementation
    """
    def __init__(self,arch,register=0,device='cuda'):
        """
        Parameters:
            arch: vit_small,vit_base
            patch_size: 8,16
            registers: 4 
                DINOv2 with register
            device: cuda or cpu
        """
        super(dinov2_featurizer,self).__init__()
        # self.patch_size = patch_size
        self.device = device
        self.registers = register
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        if self.registers>0:
            if arch == 'dinov2_vit_small':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            elif arch == 'dinov2_vit_base':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            elif arch == 'dinov2_vit_large':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            elif arch == 'dinov2_vit_giant2':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
            else:
                raise ValueError("Unknown arch and patch size")
        else:
            if arch == 'dinov2_vit_small':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            elif arch == 'dinov2_vit_base':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            elif arch == 'dinov2_vit_large':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            elif arch == 'dinov2_vit_giant2':
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            else:
                raise ValueError("Unknown arch and patch size")

        if arch == "vit_small":
            self.n_feats = 384
        elif arch == "vit_base":
            self.n_feats = 768
        elif arch == "vit_large":
            self.n_feats = 1024
        elif arch == "vit_gaint":
            self.n_feats = 1536      
              
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(device)
    def preprocess(self, image: torch.Tensor,
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        # pil_image = image.convert('RGB')
        # if load_size is not None:
        #     pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(load_size, antialias=None),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(image)[None, ...]
        return prep_img


    def featurizer(self,img,feat_type='feat',n=1):
        with torch.no_grad():
            if feat_type == 'feat':
                feat = self.model.get_intermediate_layers(img.to(self.device),1)[0]
        return feat