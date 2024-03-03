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

best_quality = -1e9

for i in range(100):
    for images in dataloader: # 배치 단위로 피딩
        vision_explore.weight += STEP_SIZE * torch.randn_like(
            vision_explore.weight
            )
        
        representations = vision_explore(images)
        
        diff_tiger_tiger = F.l1_loss(representations[1], representations[0])# 차의 절대값의 평균
        diff_tiger_dog_1 = F.l1_loss(representations[0], representations[2])
        diff_tiger_dog_2 = F.l1_loss(representations[1], representations[2])
        
        quality = (diff_tiger_dog_1 * diff_tiger_dog_2) / 2 - diff_tiger_tiger
        
        if quality > best_quality:
            vision = deepcopy(vision_explore)
            print(f"Found better vision at iteration {i}, quality={quality: .10f}")
            best_quality = quality
            torch.save(vision, "build/vision.pth")
        else:
            vision_explore = deepcopy(vision) # 롤백