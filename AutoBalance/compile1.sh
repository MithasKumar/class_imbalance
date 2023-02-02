#!/bin/sh

python loss_search.py --config configs/cifar10/dyly_no_init/1.yaml
python retraining.py --config results/cifar10/dyly_no_init/config.yaml
#python3 plot_embeddings_tsne.py