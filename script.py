from einops import rearrange
import torch


x = torch.randn(2, 4)

y = rearrange(x, "group element -> element group")
y2 = rearrange(x, "group element -> (group element)")