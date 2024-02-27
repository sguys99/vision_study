import torch

image = torch.rand(14, 14, 3, 16, 16)

weight = torch.rand(1024, 3, 16, 16) / (3*16*16)

mixed = torch.einsum("i j c h w, d c h w ->i j d", image, weight)

for i in range(14):
    for j in range(14):
        print(mixed[i][j])