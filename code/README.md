# My Deep Learning Project

This project explores different methods like autoencoders, supervised learning, and contrastive learning on MNIST and CIFAR-10 datasets.

## How to Run

Choose a dataset (`mnist` or `cifar`) and a method (`autoencoder`, `supervised`, or `contrastive`):

### Example:
python main.py --dataset cifar --method autoencoder

### Optional Arguments:
- `--batch-size`: Batch size (default: 64)  
- `--latent-dim`: Latent vector size (default: 128)  
- `--data-path`: Dataset location (default: `/datasets/cv_datasets/data`)  
- `--seed`: Random seed (default: 42)  
- `--device`: `cuda` or `cpu` (auto-detected)