import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from models.classical.models import MNISTSupContrast
from utils.utils import generate_embeddings, vmf_kde_on_circle, pca_proj, tsne_proj
from utils.data_mnist import load_mnist_data

import matplotlib.pyplot as plt

# Initialize model
def main():
    """
    Main function to train the model and generate embeddings.
    """
    classes = (3, 6, 9)
    reduced_dim = 10
    dataset_size = (1000, 300)

    mnist_data = load_mnist_data(classes=classes, reduced_dim = reduced_dim, dataset_size=dataset_size, data_dir="../data/")
    
    def create_data_loader(data, labels, batch_size=64, shuffle=True, num_workers=4):
        dataset = TensorDataset(data, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader
    
    train_loader = create_data_loader(mnist_data["train_data"], mnist_data["train_labels"])
    val_loader = create_data_loader(mnist_data["test_data"], mnist_data["test_labels"])

    model = MNISTSupContrast(activ_type="relu", pool_type="max", head_output=2, lr=1e-3)

    # Plot embeddings before training
    embeddings, labels = generate_embeddings(model, val_loader)
    # pca_proj(embeddings, labels)
    # tsne_proj(embeddings, labels)
    vmf_kde_on_circle(embeddings, labels)

    # Training the model
    logger = CSVLogger(save_dir="logs/", name="MNISTContrast", version=0)
    trainer = Trainer(max_epochs=20, logger=logger, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_loader, val_loader)

    # Plot embeddings after training
    embeddings, labels = generate_embeddings(model, val_loader)
    # pca_proj(embeddings, labels)
    # tsne_proj(embeddings, labels)
    vmf_kde_on_circle(embeddings, labels)

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
