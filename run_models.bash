#!/bin/bash

python train.py --exp_id s2nr-mnist-sgd-normal --mode normal --optim SGD
#python train.py --exp_id s2nr-mnist-adam-normal --mode normal --optim Adam

python train.py --exp_id s2nr-mnist-sgd-face --mode face --optim SGD
#python train.py --exp_id s2nr-mnist-adam-face --mode face --optim Adam

python train.py --exp_id s2nr-mnist-sgd-vertex --mode vertex --optim SGD
#python train.py --exp_id s2nr-mnist-adam-vertex --mode vertex --optim Adam

python train.py --exp_id s2nr-mnist-sgd-regular --mode regular --optim SGD
#python train.py --exp_id s2nr-mnist-adam-regular --mode regular --optim Adam


