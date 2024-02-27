from PIL import Image
from einops import rearrange, reduce
import torch


PATCH_SIZE = 16


class Vision:
    def __init__(self) -> None:
        self.weight = torch.randn(1024, 3, 16, 16) / (3*16*16)
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        patches = rearrange(
            image, 
            "c (h ph) (w pw) -> h w c ph pw", 
            ph=PATCH_SIZE, 
            pw=PATCH_SIZE
        )
        
        representation = torch.einsum(
            "i j c h w, d c h w ->i j d", patches, self.weight
            )
        
        representation = reduce(representation, "h w d -> d", "mean")
        
        return representation