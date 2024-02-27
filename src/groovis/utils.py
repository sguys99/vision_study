import torch
from einops import rearrange
from PIL import Image
import numpy as np

IMAGE_SIZE = 224

def image_path_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)

    image = torch.tensor(image, dtype = torch.float) / 255.0
    image = rearrange(image, "h w c -> c h w")
    return image

image = image_path_to_tensor("tests/images/dog.webp")