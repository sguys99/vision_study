import torch
import torch.nn.functional as F


def quality_fn(
    representations_1: torch.Tensor, representations_2: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate the quality of given pairs of representation
    """
    
    representations = combine_representations(
        representations_1, representations_2
    )
    
    differences = compare_representations(representations)
    
    quality = evaluate_difference(differences)
    
    return quality


def combine_representations(representations_1, representations_2):
    B, D = representations_1.shape
    
    representations = torch.empty(2*B, D, dtype=torch.float)
    
    for i in range(B): # 대응되는 표상이 위치하도록...
        representations[2*i] = representations_1[i]
        representations[2*i + 1] = representations_2[i]
        
    return representations


def compare_representations(representations):  # 모든 경우의 수를 비교해야함
    N, D = representations.shape  # N * 2 = batch_size
    
    differences = torch.empty(N, N, dtype = torch.float)
    
    for i in range(N):
        for j in range(N):
            differences[i][j] = F.l1_loss(
                representations[i], representations[j]
            )
            
    return differences


def evaluate_difference(differences):
    pass

