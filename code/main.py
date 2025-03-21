import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import random
import argparse
import sys
sys.path.append('..')

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'],
                    help='Dataset to use: "mnist" or "cifar"')
    parser.add_argument('--method', type=str, default='autoencoder',
                    choices=['autoencoder','supervised','contrastive'],
                    help="Which approach to run: autoencoder, supervised or contrastive")
    return parser.parse_args()
    

if __name__ == "__main__":

    args = get_args()
    freeze_seeds(args.seed)

    print(f"Selected dataset: {args.dataset}")
    print(f"Selected method: {args.method}")
    print(f"Running on device: {args.device}")

    if args.dataset == 'mnist':            
        if args.method == 'autoencoder':
            print("Running MNIST Autoencoder...")
            from scripts.mnist_autoencoder import run
            run(args)
        elif args.method == 'supervised':
            print("Running MNIST Supervised...")
            from scripts.mnist_supervised import run
            run(args)
        elif args.method == 'contrastive':
            print("Running MNIST Contrastive...")
            from scripts.mnist_contrastive import run
            run(args)    
        else:
            raise ValueError(f"Unknown method: {args.method}")                                  
    else:  # args.dataset == 'cifar'
        if args.method == 'autoencoder':
            print("Running CIFAR Autoencoder...")
            from scripts.cifar_autoencoder import run
            run(args)
        elif args.method == 'supervised':
            print("Running CIFAR Supervised...")
            from scripts.cifar_supervised import run
            run(args)
        elif args.method == 'contrastive':
            print("Running CIFAR Contrastive...")
            from scripts.cifar_contrastive import run
            run(args)    
        else:
            raise ValueError(f"Unknown method: {args.method}")