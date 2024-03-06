from copy import deepcopy
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from groovis import Vision
from groovis.data import Animals


STEP_SIZE = 0.01

animals = Animals()
dataloader = DataLoader(animals, batch_size=3)

vision = Vision()
vision_explore = deepcopy(vision)

images: list[torch.Tensor]

best_quality = -1e9

for i in range(1000):
    for images in dataloader:
        images_1, images_2 = images
        
        vision_explore.weight += STEP_SIZE * torch.randn_like(
            vision_explore.weight
            )
        
        representations_1 = vision_explore(images_1)
        representations_2 = vision_explore(images_2)
        
        quality = quality_fn(representations_1, representations_2)

        if quality > best_quality:
            vision = deepcopy(vision_explore)
            print(f"Found better vision at iteration {i}, quality={quality: .10f}")
            best_quality = quality
            torch.save(vision, "build/vision.pth")
        else:
            vision_explore = deepcopy(vision) # 롤백