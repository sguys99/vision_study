import torch
from einops import reduce
from PIL import Image
import numpy as np

from groovis.utils import image_path_to_tensor

torch.set_printoptions(linewidth=120)

image = image_path_to_tensor("tests/images/dog.webp")

PATCH_SIZE = 28 # 그냥 조정

representation = reduce(
    image, "c (h ph) (w pw) ->h w", "mean", ph=PATCH_SIZE, pw=PATCH_SIZE)

global_average = representation.mean()

print(representation > global_average)