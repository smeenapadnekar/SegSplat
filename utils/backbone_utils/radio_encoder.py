from PIL import Image

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
import torchvision
from torchvision import transforms
from typing import Union, List, Tuple
from PIL import Image

def upsampler(feature, upsampled_height, upsampled_width, max_chunk=None):
    """
    Upsample the feature tensor to the specified height and width.

    Args:
    - feature (torch.Tensor): The input tensor with size [B, H, W, C].
    - upsampled_height (int): The target height after upsampling.
    - upsampled_width (int): The target width after upsampling.

    Returns:
    - upsampled_feature (torch.Tensor): The upsampled tensor with size [B, upsampled_height, upsampled_width, C].
    """
    # Permute the tensor to [B, C, H, W] for interpolation
    feature = feature.permute(0, 3, 1, 2)
    
    # Perform the upsampling
    if max_chunk:
        upsampled_chunks = []

        for i in range(0, len(feature), max_chunk):
            chunk = feature[i:i+max_chunk]
            
            upsampled_chunk = F.interpolate(chunk, size=(upsampled_height, upsampled_width), mode='bilinear', align_corners=False)
            upsampled_chunks.append(upsampled_chunk)
        
        upsampled_feature = torch.cat(upsampled_chunks, dim=0)
    else:
        upsampled_feature = F.interpolate(feature, size=(upsampled_height, upsampled_width), mode='bilinear', align_corners=False)
    
    # Permute back to [B, H, W, C]
    upsampled_feature = upsampled_feature.permute(0, 2, 3, 1)
    
    return upsampled_feature

class AM_Radio:
    def __init__(self, model_type: str = 'dino_vits8',device: 'str'= 'cuda'):
        """
            param model_type: a string specifying which model to load. [ nvidia/E-RADIO | nvidia/RADIO-B | nvidia/RADIO 
            | nvidia/RADIO-g | nvidia/C-RADIO | nvidia/RADIO-L ]
        """

        self.model_type = model_type
        self.device = device
        self.process = torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        # def create_model(self, model_type: str):
        # image_processor = CLIPImageProcessor.from_pretrained(model_type)
        self.model = AutoModel.from_pretrained(self.model_type, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.p = self.model.config.patch_size
    
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
            # transforms.ToTensor(),
            transforms.Resize(load_size, antialias=None),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(image)[None, ...]
        return prep_img
    
    def _extract_features(self,images):
        images = self.preprocess(images,(368,496)).half().squeeze(0)
        B, C, H, W = images.shape
        with torch.no_grad():
            _ , spatial_features = self.model(images)
        spatial_features = spatial_features.reshape(B,H//self.p,W//self.p,-1)
        return spatial_features