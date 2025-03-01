import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_autoencoder(ae_model, dataloader, device='cuda'):
    """
    Plots TSNE for:
      1) The latent vectors from ae_model.encoder(...)
      2) The raw image domain (flattened)
    using the same approach as your original plot_tsne code.
    """
    ae_model.eval()

    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            latent_vector = ae_model.encoder(images)  

            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())

    images = np.concatenate(images_list, axis=0)        
    labels = np.concatenate(labels_list, axis=0)        
    latent_vectors = np.concatenate(latent_list, axis=0) 

    # 1) TSNE on latent space
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:,0], latent_tsne[:,1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Autoencoder Latent Space')
    plt.show()

    images_flat = images.reshape(images.shape[0], -1) 
    tsne_image = TSNE(n_components=2, random_state=42)
    image_tsne = tsne_image.fit_transform(images_flat)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:,0], image_tsne[:,1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Raw Image Space')
    plt.show()

def plot_tsne_classifier(encoder_classifier, dataloader, device='cuda'):
    """
    Extract latent embeddings via encoder_classifier.encoder(...),
    then do TSNE on them + TSNE on raw images.
    """
    encoder_classifier.eval()

    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            latent_vector = encoder_classifier.encoder(images)

            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)

    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(latent_tsne[:,0], latent_tsne[:,1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Classifier (Encoder) Latent Space')
    plt.show()

    images_flat = images.reshape(images.shape[0], -1)
    tsne_image = TSNE(n_components=2, random_state=42)
    image_tsne = tsne_image.fit_transform(images_flat)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(image_tsne[:,0], image_tsne[:,1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Raw Image Space')
    plt.show()


def plot_tsne_supcon(supcon_model, dataloader, device='cuda'):
    """
    For the MNISTSupCon model, forward(x) returns embeddings directly.
    Plot TSNE on those embeddings + raw images.
    """
    supcon_model.eval()

    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            z = supcon_model(images)  

            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(z.cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)

    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(latent_tsne[:,0], latent_tsne[:,1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of SupCon Latent Space')
    plt.show()

    images_flat = images.reshape(images.shape[0], -1)
    tsne_image = TSNE(n_components=2, random_state=42)
    image_tsne = tsne_image.fit_transform(images_flat)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(image_tsne[:,0], image_tsne[:,1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Raw Image Space')
    plt.show()

