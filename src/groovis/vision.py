from PIL import Image
from einops import reduce
import torch


PATCH_SIZE = 16


def vision(image: torch.Tensor) -> torch.Tensor:
    return reduce(
        image, "c (h ph) (w pw) ->h w", "mean", ph=PATCH_SIZE, pw=PATCH_SIZE
    )