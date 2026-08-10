import os
import glob
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

# Import utility scripts
from U2Net import U2NET
from u2net_data_loader import RescaleT, ToTensorLab, SalObjDataset