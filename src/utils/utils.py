import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import kornia
import torchvision.transforms as T

def contrastive_loss_with_margins(embeddings, labels, pos_margin=0.25, neg_margin=1.5):
    """
    Custom contrastive loss function with positive and negative margins.
    
    Args:
        embeddings (Tensor): Embedding vectors.
        labels (Tensor): Corresponding labels.
        pos_margin (float): Margin for positive pairs.
        neg_margin (float): Margin for negative pairs.

    Returns:
        Tensor: Calculated loss.
    """
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    positive_loss = (1 - labels) * F.relu(distance_matrix - pos_margin).pow(2)
    negative_loss = labels * F.relu(neg_margin - distance_matrix).pow(2)
    combined_loss = 0.5 * (positive_loss + negative_loss)
    mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    combined_loss = combined_loss.masked_fill_(mask, 0)
    loss = combined_loss.mean()
    return loss

def get_preprocessing(preprocess):
    """
    Get preprocessing transformation based on the specified type.

    Args:
        preprocess (str): Type of preprocessing.

    Returns:
        Callable: Preprocessing transformation.
    """
    if preprocess == "RandAffine":
        return kornia.augmentation.RandomAffine(degrees=(-40, 40), translate=0.25, scale=[0.5, 1.5], shear=45)
    elif preprocess == "RandAug":
        return T.RandAugment()
    return None

def pca_proj(embeddings, labels, seed=1):
    """
    Perform PCA projection and plot the results.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
        seed (int): Random seed for reproducibility.
    """
    proj = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette=sns.color_palette("tab10")).set(title="PCA")
    plt.show()

def tsne_proj(embeddings, labels, seed=1):
    """
    Perform t-SNE projection and plot the results.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
        seed (int): Random seed for reproducibility.
    """
    proj = TSNE(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette=sns.color_palette("tab10")).set(title="T-SNE")
    plt.show()

def generate_embeddings(model, data_loader):
    """
    Generate embeddings for the given data using the provided model.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): Data loader for the dataset.

    Returns:
        tuple: Embeddings and labels as numpy arrays.
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = model.encoder(x)
            emb = model.head(x)
            embeddings.append(emb)
            labels.append(y)
    
    embeddings = torch.cat(embeddings).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    return embeddings, labels

# def swap_test_circuit(embedding1, embedding2):
#     """
#     Defines a quantum circuit for the SWAP test to compare two embeddings.
    
#     Args:
#         embedding1 (np.ndarray): First embedding vector.
#         embedding2 (np.ndarray): Second embedding vector.

#     Returns:
#         Callable: Quantum circuit function.
#     """
#     num_qubits_per_embedding = int(np.ceil(np.log2(len(embedding1))))
#     total_num_qubits = 2 * num_qubits_per_embedding + 1  # +1 for the ancilla qubit

#     dev = qml.device('default.qubit', wires=total_num_qubits)
    
#     @qml.qnode(dev)
#     def circuit(embedding1, embedding2):
#         qml.Hadamard(wires=0)
#         qml.AmplitudeEmbedding(features=embedding1, wires=range(1, num_qubits_per_embedding + 1), normalize=True)
#         qml.AmplitudeEmbedding(features=embedding2, wires=range(num_qubits_per_embedding + 1, 2 * num_qubits_per_embedding + 1), normalize=True)
        
#         for i in range(1, num_qubits_per_embedding + 1):
#             qml.CSWAP(wires=[0, i, i + num_qubits_per_embedding])
#         qml.Hadamard(wires=0)
        
#         return qml.expval(qml.PauliZ(0))
    
#     return circuit(embedding1, embedding2)
