
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

plt.rcParams.update({'font.size': 14})

def plot_metrics_from_csv(metrics_file, metrics={'val_loss', 'val_acc', 'val_auc'}):
    df = pd.read_csv(metrics_file)

    required_columns = metrics
    if not required_columns.issubset(df.columns):
        raise ValueError("The CSV file does not contain the required metrics.")

    df = df.sort_values('epoch')

    df = df.fillna(method='ffill')

    epochs = df['epoch']
    val_loss = df['val_loss']
    val_acc = df['val_acc']
    val_auc = df['val_auc']

    plt.figure(figsize=(5*len(metrics), 5))

    plt.subplot(1, len(metrics), 1)
    plt.plot(epochs, val_loss, marker='o', linestyle='-', color='b', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_acc, marker='o', linestyle='-', color='r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_auc, marker='o', linestyle='-', color='g', label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def reduce_resolution(image):
    image = np.array(image)
    
    assert image.shape[0] % 5 == 0 and image.shape[1] % 5 == 0, "Image dimensions should be divisible by 5"
    
    # Reshape the image into 25x25x5x5
    reduced_image = image.reshape(25, 5, 25, 5).mean(axis=(1, 3))
    
    return reduced_image

def reduce_resolution_batch(images):
    return np.array([reduce_resolution(img) for img in images])

def visualize_image(image, label):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im = axs[0].imshow(image[:, :, 0], cmap='binary')
    axs[0].title.set_text(f'Class {label} - Channel 0')

    im = axs[1].imshow(image[:, :, 1],  cmap='binary')
    axs[1].title.set_text(f'Class {label} - Channel 1')

    im = axs[2].imshow(image[:, :, 2],  cmap='binary')
    axs[2].title.set_text(f'Class {label} - Channel 2')

def visualize_average_images(x_data, y_data, num=-1, use_lognorm=False):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    norm = LogNorm() if use_lognorm else None
    # Calculate average images for each class and channel
    avg_images = {}
    for class_label in [0, 1]:
        avg_images[class_label] = []
        class_data = x_data[y_data == class_label]
        for channel in range(3):
            # print(len(class_data))
            avg_image = np.average(class_data[:num, :, :, channel], 0)
            avg_images[class_label].append(avg_image)
    
    # Plot for class 0
    im = axs[0, 0].imshow(avg_images[0][0], norm = norm, cmap='binary')
    axs[0, 0].title.set_text('Class 0 - Channel 0')
    fig.colorbar(im, ax=axs[0, 0])

    im = axs[0, 1].imshow(avg_images[0][1], norm = norm, cmap='binary')
    axs[0, 1].title.set_text('Class 0 - Channel 1')
    fig.colorbar(im, ax=axs[0, 1])

    # im = axs[0, 2].imshow(reduce_resolution(avg_images[0][2]), norm=LogNorm(), cmap='binary')
    im = axs[0, 2].imshow(avg_images[1][2], norm = norm, cmap='binary')
    axs[0, 2].title.set_text('Class 0 - Channel 2')
    fig.colorbar(im, ax=axs[0, 2])

    # Plot for class 1
    im = axs[1, 0].imshow(avg_images[1][0], norm = norm, cmap='binary')
    axs[1, 0].title.set_text('Class 1 - Channel 0')
    fig.colorbar(im, ax=axs[1, 0])

    im = axs[1, 1].imshow(avg_images[1][1], norm = norm, cmap='binary')
    axs[1, 1].title.set_text('Class 1 - Channel 1')
    fig.colorbar(im, ax=axs[1, 1])
    # print(avg_images[1][2].shape)
    im = axs[1, 2].imshow(reduce_resolution(avg_images[1][2]), norm = norm, cmap='binary')
    axs[1, 2].title.set_text('Class 1 - Channel 2')
    fig.colorbar(im, ax=axs[1, 2])

    fig.tight_layout()
    plt.show()


def visualize_diff_average_images(x_data, y_data, num=-1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # Calculate average images for each class and channel
    avg_images = {}
    for class_label in [0, 1]:
        avg_images[class_label] = []
        class_data = x_data[y_data == class_label]
        for channel in range(3):
            # print(len(class_data))
            avg_image = np.average(class_data[:num, :, :, channel], 0)
            avg_images[class_label].append(avg_image)

    im = axs[0].imshow(avg_images[1][0]/avg_images[0][0], norm=LogNorm(), cmap='binary')
    axs[0].title.set_text('Class 1 / Class 0 - Channel 0')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(avg_images[1][1]/avg_images[0][1], norm=LogNorm(), cmap='binary')
    axs[1].title.set_text('Class 1 / Class 0 - Channel 1')
    fig.colorbar(im, ax=axs[1])

    im = axs[2].imshow(avg_images[1][2]/avg_images[0][2], norm=LogNorm(), cmap='binary')
    axs[2].title.set_text('Class 1 / Class 0 - Channel 2')
    fig.colorbar(im, ax=axs[2])

    fig.tight_layout()
    plt.show()
    
