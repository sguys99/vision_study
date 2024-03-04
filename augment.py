import albumentations as A
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.savefig("augmented.png", bbox_inches="tight", pad_inches=0)


image = np.array(Image.open("data/train/dog.webp"))

transform = A.HorizontalFlip(p=1)
augmented_image = transform(image=image)["image"]

visualize(augmented_image)
