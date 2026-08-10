import os
import torch
import torchvision.transforms as tvf
import torch.nn.functional as F
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

class FeatureExtractor:
    def __init__(self,feat_type,feat_dim,vis_feat,device):
        """
        feat_type (list): VFM, choose in ["dust3r", "mast3r", "dift", "dino_b16", "dinov2_b14", "radio", "clip_b16", "mae_b16", "midas_l16", "sam_base", "iuvrgb"].
        feat_dim (int): PCA dimensions.
        img_base_path (str): Training view data directory path.
        model_path (str): Model path, './submodules/mast3r/checkpoints/'.
        vis_feat (bool): Visualize and save feature maps.
        device (str): 'cuda'.
        """
        self.feat_type = feat_type
        self.feat_dim = feat_dim
        self.vis_feat = vis_feat
        self.device = device

        if "dinov2" in self.feat_type:
            print("Knowledge distilled from DINOv2")
            from .backbone_utils.dino_encoder import ViTExtractor
            self.feat_extractor = ViTExtractor(feat_type)
        elif "dino" in self.feat_type: 
            print("Knowledge distilled from DINO")
            from .backbone_utils.dino_encoder import ViTExtractor
            self.feat_extractor = ViTExtractor(feat_type)
        elif "clip" in self.feat_type:
            print("Knowledge distilled from CLIP")
            from .backbone_utils.clip_encoder import CLIPNetwork
            self.feat_extractor = CLIPNetwork()
        elif "RADIO" in self.feat_type:
            print("Knowledge distilled from RADIO")
            from .backbone_utils.radio_encoder import AM_Radio
            self.feat_extractor = AM_Radio(feat_type)
        else:
            raise TypeError(f"{self.feat_type} is not a supported.")
    
    def feat(self,images):
        return self.feat_extractor._extract_features(images)
    # def dinov2_feat(self,images):
    #     return self.feat_extractor.featurizer(images)
    # def radio_feat(self,images):
    #     return self.feat_extractor._extract_feature(images)
    # def clip_feat(self,images):
    #     return self.feat_extractor._extract_feature(images)
        
# backbone = FeatureExtractor("clip",756,False,"cuda")
# folder_path = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/dataset/nerf_llff_data/fern/3_views/images"
# images = []
# for filename in os.listdir(folder_path):
#     image_path = os.path.join(folder_path, filename)
#     image = Image.open(image_path)
#     image = backbone.feat_extractor.preprocess(image,(224,224))
#     images.append(image)
# images = torch.stack(images)
# feat = backbone.clip_feat(images.squeeze(1).cuda())
# print(feat.shape)