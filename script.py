from copy import deepcopy
from PIL import Image
import torch.nn.functional as F
import torch

from groovis import Vision
from groovis.utils import image_path_to_tensor

STEP_SIZE = 0.01

vision = Vision()
vision_explore = deepcopy(vision)

best_quality = -1e9

for i in range(100):
    
    vision_explore.weight += STEP_SIZE * torch.randn_like(vision_explore.weight)
    
    image_tiger_1 = image_path_to_tensor("data/train/tiger1.webp")
    image_tiger_2 = image_path_to_tensor("data/train/tiger2.webp")
    image_dog = image_path_to_tensor("data/train/dog.webp")
    
    tiger_1 = vision_explore(image_tiger_1)
    tiger_2 = vision_explore(image_tiger_2)
    dog = vision_explore(image_dog)
    
    diff_tiger_tiger = F.l1_loss(tiger_2, tiger_2)# 차의 절대값의 평균
    diff_tiger_dog_1 = F.l1_loss(tiger_1, dog)
    diff_tiger_dog_2 = F.l1_loss(tiger_2, dog)
    
    quality = (diff_tiger_dog_1 * diff_tiger_dog_2) / 2 - diff_tiger_tiger
    
    if quality > best_quality:
        vision = deepcopy(vision_explore)
        print(f"Found better vision at iteration {i}, quality={quality: .10f}")
        best_quality = quality
        torch.save(vision, "build/vision.pth")
    else:
        vision_explore = deepcopy(vision) # 롤백