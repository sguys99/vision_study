from torch.utils.data import Dataset
from groovis.utils import image_path_to_tensor


class Animals(Dataset):
    def __init__(self):
        self.paths = [
            "data/train/tiger1.webp",
            "data/train/tiger2.webp",
            "data/train/dog.webp"
        ]
    
    def __getitem__(self, index):
        return image_path_to_tensor(self.paths[index])
    
    def __len__(self):
        return len(self.paths)
