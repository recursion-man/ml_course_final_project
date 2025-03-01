import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
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
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--method', type=str, default='1.2.1',
                    choices=['autoencoder','supervised','contrastive'],
                    help="Which approach to run: autoencoder, supervised or contrastive")
    return parser.parse_args()
    

if __name__ == "__main__":

    args = get_args()
    freeze_seeds(args.seed)

    if args.mnist:            
        if args.method == 'autoencoder':
            from scripts.mnist_autoencoder import run
            run(args)
        elif args.method == 'supervised':
            from scripts.mnist_supervised import run
            run(args)
        elif args.method == 'contrastive':
            from scripts.mnist_contrastive import run
            run(args)    
        else:
            raise ValueError(f"Unknown method: {args.method}")                                  
    else:
        if args.method == 'autoencoder':
            from scripts.cifar_autoencoder import run
            run(args)
        elif args.method == 'supervised':
            from scripts.cifar_supervised import run
            run(args)
        elif args.method == 'contrastive':
            from scripts.cifar_contrastive import run
            run(args)    
        else:
            raise ValueError(f"Unknown method: {args.method}") 