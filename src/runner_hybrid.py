import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split, Subset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from models.hybrid.hybrid_contrastive import MNISTQSupContrast
from utils.utils import generate_embeddings, pca_proj, tsne_proj

import matplotlib.pyplot as plt

def load_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        tuple: Training and validation data loaders.
    """
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root="../data/", train=True, download=True, transform=transform)

    filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in [0, 1, 5]]
    filtered_dataset = Subset(dataset, filtered_indices)

    train_size = int(0.9 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=12)

    return train_loader, val_loader

# Initialize model
def main():
    """
    Main function to train the model and generate embeddings.
    """
    train_loader, val_loader = load_data()

    model = MNISTQSupContrast(activ_type="relu", pool_type="max", head_output=2, lr=1e-3, n_qlayers=1)

    # Plot embeddings before training
    embeddings, labels = generate_embeddings(model, val_loader)
    pca_proj(embeddings, labels)
    tsne_proj(embeddings, labels)

    # Training the model
    logger = CSVLogger(save_dir="logs/", name="MNISTContrast", version=0)
    trainer = Trainer(max_epochs=10, logger=logger, gpus=0)
    trainer.fit(model, train_loader, val_loader)

    # Plot embeddings after training
    embeddings, labels = generate_embeddings(model, val_loader)
    pca_proj(embeddings, labels)
    tsne_proj(embeddings, labels)

    # Plot training and validation loss
    metrics_df = pd.read_csv(f"{logger.log_dir}/metrics.csv")
    train_loss_epoch = metrics_df['train_loss'].dropna().reset_index(drop=True)
    val_loss_epoch = metrics_df['val_loss'].dropna().reset_index(drop=True)
    min_length = min(len(train_loss_epoch), len(val_loss_epoch))
    train_loss_epoch = train_loss_epoch[:min_length]
    val_loss_epoch = val_loss_epoch[:min_length]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_epoch, label='Train Loss')
    plt.plot(val_loss_epoch, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Quantum encoding and SWAP test
    # img1, label1 = val_loader.dataset[0]
    # img2, label2 = val_loader.dataset[1]
    # img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)

    # model.eval()
    # with torch.no_grad():
    #     embedding1 = model.forward(img1).flatten().numpy()
    #     embedding2 = model.forward(img2).flatten().numpy()

    # fidelity = swap_test_circuit(embedding1, embedding2)
    # print("Fidelity (result of SWAP test):", fidelity)

    # Fidelity scores for a batch of samples
    # reference_image, reference_label = val_loader.dataset[0]
    # reference_embedding = model.forward(reference_image.unsqueeze(0)).detach().numpy().flatten()

    # embeddings = []
    # labels = []

    # for i in range(100):
    #     image, label = val_loader.dataset[i]
    #     embedding = model.forward(image.unsqueeze(0)).detach().numpy().flatten()
    #     embeddings.append(embedding)
    #     labels.append(label)

    # fidelity_scores = []
    # for embedding, label in zip(embeddings, labels):
    #     fidelity = swap_test_circuit(reference_embedding, embedding)
    #     fidelity_scores.append((fidelity.item(), str(label)))

    # df = pd.DataFrame(fidelity_scores, columns=['Fidelity', 'Label'])

    # plt.figure(figsize=(10, 6))
    # sns.kdeplot(data=df, x='Fidelity', hue='Label', common_norm=False, fill=True)
    # plt.title('Fidelity Score Distributions by Label')
    # plt.xlabel('Fidelity Score')
    # plt.ylabel('Density')
    # plt.show()

if __name__ == "__main__":
    main()
