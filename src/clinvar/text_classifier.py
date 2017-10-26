"""
A small multiclass text classifier
from PyTorch
"""

import sys
import os
import argparse
import time
import random
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import logging

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--train", type=str, default='./ptb/ptb.train.txt', help="training file")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--unroll_size", type=int, default=35)
argparser.add_argument("--max_epoch", type=int, default=300)
argparser.add_argument("--d", type=int, default=910)
argparser.add_argument("--dropout", type=float, default=0.3,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--rnn_dropout", type=float, default=0.2,
                       help="dropout of RNN layers")
argparser.add_argument("--depth", type=int, default=2)
argparser.add_argument("--lr", type=float, default=1.0)
argparser.add_argument("--lr_decay", type=float, default=0.98)
argparser.add_argument("--lr_decay_epoch", type=int, default=175)
argparser.add_argument("--weight_decay", type=float, default=1e-5)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./sandbox')
argparser.add_argument("--seed", type=int, default=123)

args = argparser.parse_args()
print (args)

if __name__ == '__main__':

    with open('./data/clinvar/simple_allele_associations.json', 'rb') as f:
        associations = json.load(f)