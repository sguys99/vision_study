import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from groovis.utils import image_path_to_array


# 디펄트 어그멘테이션
SIMCLR_AUG = A.Compose(
    [
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.08, 1),
            ratio=(0.75, 1.3333333333),
            always_apply=True,
        ),
        A.HorizontalFlip(p=1),
        A.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
            p=0.8
            ),
        A.ToGray(p=0.2),
        A.GaussianBlur(
            blur_limit=(21, 23),
            sigma_limit=(0.1, 2),
            always_apply=True
        ),
        ToTensorV2(always_apply=True) # 파이토치 텐서로 바꿔주기
        ]
    )


class Animals(Dataset):
    def __init__(self, transforms: A.Compose = SIMCLR_AUG):
        self.paths = [
            "data/train/tiger1.webp",
            "data/train/tiger2.webp",
            "data/train/dog.webp"
        ]
        
        self.transofroms = transforms
    
    def __getitem__(self, index):
        image = image_path_to_array(self.paths[index])
        augmented_image = self.transofroms(image=image)["image"]
        
        return augmented_image / 255.0  # 텐서임
    
    def __len__(self):
        return len(self.paths)