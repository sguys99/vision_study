import torch
from einops import rearrange
from PIL import Image
import numpy as np

IMAGE_SIZE = 224


def image_path_to_array(path: str) -> np.ndarray:
    image = Image.open(path)
    image = np.array(image)

    # image = torch.tensor(image, dtype = torch.float) / 255.0
    # image = rearrange(image, "h w c -> c h w")
    
    return image
