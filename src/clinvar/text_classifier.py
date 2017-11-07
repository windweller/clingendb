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
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data

import logging

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--data_dir", type=str, default='./data', help="training file")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--unroll_size", type=int, default=35)
argparser.add_argument("--max_epoch", type=int, default=40)
argparser.add_argument("--d", type=int, default=910)
argparser.add_argument("--dropout", type=float, default=0.3,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--rnn_dropout", type=float, default=0.2,
                       help="dropout of RNN layers")
argparser.add_argument("--depth", type=int, default=2)
argparser.add_argument("--lr", type=float, default=1.0)
# argparser.add_argument("--lr_decay", type=float, default=0.98)
# argparser.add_argument("--lr_decay_epoch", type=int, default=175)
# argparser.add_argument("--weight_decay", type=float, default=1e-5)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./sandbox')
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--gpu", type=int, default=-1)

args = argparser.parse_args()
print (args)

"""
Seeding
"""
torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

class Model(nn.Module):
    def __init__(self, vocab, emb_dim=100, hidden_size=256, depth=1, nclasses=5):
        super(Model, self).__init__()
        self.drop = nn.Dropout(0.2)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_size,
            depth,
            dropout=0.2,
            bidirectional=False)
        d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        # self.embed.weight.requires_grad = False  # no training on word embedding?

    def forward(self, input):
        embed_input = self.embed(input)
        output, hidden = self.encoder(embed_input)
        output = output[-1]

        output = self.drop(output)
        return self.out(output)

def get_multiclass_accuracy(preds, labels):
    # this can't be preds, must be output!!!
    label_cat = range(len(labels))
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of the correct labels
        idx_of_cat = labels == la
        cat_preds = preds[idx_of_cat]
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def cumulate_multiclass_accuracy(total_accu, labels_accu):
    for k, v in labels_accu.iteritems():
        total_accu[k].extend(v)


def get_mean_multiclass_accuracy(self, total_accu):
    for k, v in total_accu.iteritems():
        total_accu[k] = np.mean(total_accu[k])

def eval_model(model, valid_iter):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0
    total_labels_accu = None
    for data in valid_iter:
        x, y = data.Text, data.Description
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.data[0] * x.size(1)
        preds = output.data.max(1)[1]  # already taking max...I think, max returns a tuple
        correct += preds.eq(y.data).cpu().sum()
        cnt += y.numel()

        # compute multiclass
        labels_accu = get_multiclass_accuracy(output.data.cpu().numpy(), y.data.cpu().numpy())
        if total_labels_accu is None:
            total_labels_accu = labels_accu
        else:
            cumulate_multiclass_accuracy(total_labels_accu, labels_accu)

        if cnt >= 10198:
            break

    multiclass_accu_msg = ''
    for k, v in total_labels_accu.iteritems():
        multiclass_accu_msg += labels[k] + ": " + str(v) + " "

    logging.info(multiclass_accu_msg)

    model.train()
    return correct / cnt


def train_module(model, optimizer,
                 train_iter, valid_iter, test_iter, max_epoch):
    model.train()
    cnt = 0
    criterion = nn.CrossEntropyLoss()

    exp_cost = None
    end_of_epoch = True # False  # set true because we want immediate feedback...
    iter = 0
    best_valid = 0.
    epoch = 1

    for data in train_iter:
        iter += 1

        model.zero_grad()
        x, y = data.Text, data.Description

        cnt += y.numel()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

        optimizer.step()

        if cnt >= 101979:
            epoch += 1
            end_of_epoch = True
            cnt = 0

        if not exp_cost:
            exp_cost = loss.data[0]
        else:
            exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

        if iter % 100 == 0:
            logging.info("iter {} lr={} train_loss={} \n".format(iter, optimizer.param_groups[0]['lr'],
                                                                 loss.data[0]))

        if end_of_epoch:
            valid_accu = eval_model(model, valid_iter)
            sys.stdout.write("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
                epoch,
                optimizer.param_groups[0]['lr'],
                loss.data[0],
                valid_accu
            ))

            if valid_accu > best_valid:
                best_valid = valid_accu
                test_accu = eval_model(model, test_iter)
                sys.stdout.write("test_acc: {:.6f}\n".format(
                    test_accu
                ))
            sys.stdout.write("\n")
            end_of_epoch = False

        if epoch == max_epoch:
            break


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


if __name__ == '__main__':
    import spacy

    spacy_en = spacy.load('en')

    with open('../../data/clinvar/text_classification_db_labels.json', 'r') as f:
        labels = json.load(f)

    # map labels to list
    label_list = [None] * len(labels)
    for k, v in labels.items():
        label_list[v] = k

    labels = label_list
    print("available labels: ")
    print(labels)

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train, val, test = data.TabularDataset.splits(
        path='../../data/clinvar/', train='text_classification_db_train.tsv',
        validation='text_classification_db_valid.tsv', test='text_classification_db_test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Description', LABEL)])

    TEXT.build_vocab(train, vectors="glove.6B.100d")

    # data is on GPU already!
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),
        batch_sizes=(32, 256, 256), device=args.gpu)

    vocab = TEXT.vocab

    # so now all you need to do is to create an iterator
    print("processed")

    model = Model(vocab, nclasses=len(labels))
    if torch.cuda.is_available():
        model.cuda()

    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr=0.001)
    # optimizer = optim.SGD(
    #     filter(need_grad, model.parameters()),
    #     lr=0.01)

    train_module(model, optimizer, train_iter, val_iter, test_iter,
                 max_epoch=40)
