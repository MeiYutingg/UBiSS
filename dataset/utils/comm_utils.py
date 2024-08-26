import io
import json

import numpy as np
import torch
import math
import PIL.Image
import matplotlib.pyplot as plt
import wandb

from torchvision.transforms import ToTensor

def test_collate(input):
        return input[0]

