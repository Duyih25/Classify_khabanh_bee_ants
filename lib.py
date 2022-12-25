import torch
import torch.nn as nn
import torchvision
import numpy as np

import os
import zipfile

import urllib.request as ur

import glob
import torch.optim as optim
from PIL import Image
import torch.utils.data as data
from torchvision import transforms, models
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

torch.manual_seed(25)
np.random.seed(25)
random.seed(25)

cudnn.benchmark = False
cudnn.deterministic = True
