# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Union, List, Tuple
from PIL import Image
import torch
import torchvision
from torchvision import transforms

try:
    import clip
except ImportError:
    assert False, "clip is not installed, install it with `pip install clip`"

# BaseImageEncoderConfig
from .image_encoder import BaseImageEncoder
@dataclass
class CLIPNetworkConfig():
    clip_model_type: str = "ViT-B/16"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")


class CLIPNetwork(BaseImageEncoder):
    def __init__(self, config=CLIPNetworkConfig):
        super().__init__()
        self.config = config
        
        model, _ = clip.load(self.config.clip_model_type)
        model.eval()
        self.tokenizer = clip.tokenize
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims
        self.positives = ["hand sanitizer"]
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225) 
        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "clip_openai_{}".format(self.config.clip_model_type)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
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
    
    def _extract_features(self, input):
       with torch.no_grad():
            def hook(module, input, output):
                self.intermediate_output = output
            handle = self.model.visual.transformer.resblocks[-2].register_forward_hook(hook)
            x = self.model.encode_image(input)
            handle.remove()
            # print(x.shape)
            return self.intermediate_output