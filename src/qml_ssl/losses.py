import torch
import torch.nn.functional as F

def alignment_loss(embeddings1, embeddings2):
    # Alignment loss over the batch
    return torch.mean(torch.norm(embeddings1 - embeddings2, dim=1) ** 2)

def uniformity_loss(embeddings, t=2):
    # Normalize embeddings to lie on the unit hypersphere
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2) ** 2
    # Apply the uniformity loss formula
    uniformity = torch.mean(torch.exp(-t * distances))
    return uniformity