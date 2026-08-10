import torch
import torch.nn as nn
from torchvision import transforms
from typing import Union, List, Tuple
from PIL import Image

class dino_featurizer(nn.Module):   
    """
        DINO featurizer :
            Latest implementation - uses the output activation map for further implementation
    """
    def __init__(self,arch,device='cuda'):
        """
        Parameters:
            arch: vit_small,vit_base
            patch_size: 8,16
            device: cuda or cpu
        """
        super(dino_featurizer,self).__init__()
        self.device = device
        if arch == "dino_vits16":
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        elif arch == "dino_vits8":
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        elif arch == "dino_vitb16":
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        elif arch == "dino_vitb8":
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        else:
            raise ValueError("Unknown arch and patch size")
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
        
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

    def featurizer(self,img,feat_type="feat",n=1):
        with torch.no_grad():
            if feat_type == 'feat':
                feat = self.model.get_intermediate_layers(img.to(self.device),1)
                feat = feat[0][:,1:,:]
            elif feat_type == "attn":
                attn = self.model.get_last_selfattention(img.to(self.device))
                nh = attn.shape[1]
                feat = attn[0, :, 0, 1:].reshape(nh, -1)
            elif feat_type == "qkv":
                pass
            else:
                raise ValueError("Unknown feature type: {}",feat_type)
        return feat