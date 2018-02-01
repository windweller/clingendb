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

from torch.nn.utils.rnn import pad_packed_sequence as unpack

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data
from sklearn.metrics import f1_score
from util import MultiLabelField, ReversibleField

from sklearn import metrics

import logging

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--dataset", type=str, default='major', help="major|multi, majority label or multi label")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--unroll_size", type=int, default=35)
argparser.add_argument("--max_epoch", type=int, default=10)
argparser.add_argument("--d", type=int, default=256)
argparser.add_argument("--emb_dim", type=int, default=100)
argparser.add_argument("--dropout", type=float, default=0.3,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--depth", type=int, default=1)
argparser.add_argument("--lr", type=float, default=1.0)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./exp')
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--gpu", type=int, default=-1)
argparser.add_argument("--attn", action="store_true", help="by using attention to generate some interpretation")
argparser.add_argument("--emb_update", action="store_true", help="update embedding")
argparser.add_argument("--rand_unk", action="store_true", help="randomly initialize unk")

args = argparser.parse_args()

VERY_NEGATIVE_NUMBER = -1e30

"""
Seeding
"""
torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

"""
Logging
"""
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(args.run_dir):
    os.makedirs(args.run_dir)
file_handler = logging.FileHandler("{0}/log.txt".format(args.run_dir))
logging.getLogger().addHandler(file_handler)

logger.info(args)


def move_to_cuda(th_var):
    if torch.cuda.is_available():
        return th_var.cuda(args.gpu)
    else:
        return th_var


class Model(nn.Module):
    def __init__(self, vocab, emb_dim=100, hidden_size=256, depth=1, nclasses=5,
                 scaled_dot_attn=False, temp_max_pool=False):
        super(Model, self).__init__()
        self.scaled_dot_attn = scaled_dot_attn
        self.nclasses = nclasses
        self.temp_max_pool = temp_max_pool
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(args.dropout)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_size,
            depth,
            dropout=args.dropout,
            bidirectional=False)  # ha...not even bidirectional
        d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)  # nclasses
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if args.emb_update else False

        if self.scaled_dot_attn:
            logging.info("adding scaled dot attention matrix")
            self.key_w = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def create_mask(self, lengths):
        # lengths would be a python list here, not a Tensor
        # [max_len, batch_size]
        masks = np.ones([max(lengths), len(lengths)], dtype='float32')
        for i, l in enumerate(lengths):
            masks[l:, i] = 0.
        return torch.from_numpy(masks)

    def exp_mask(self, val, mask):
        """Give very negative number to unmasked elements in val.
        For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
        Typically, this effectively masks in exponential space (e.g. softmax)
        Args:
            val: values to be masked
            mask: masking boolean tensor, same shape as tensor
            name: name for output tensor
        Returns:
            Same shape as val, where some elements are very small (exponentially zero)
        """
        exp_mask = Variable((1 - mask) * VERY_NEGATIVE_NUMBER, requires_grad=False)
        return val + move_to_cuda(exp_mask)

    def forward(self, input, lengths=None):
        embed_input = self.embed(input)

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input
        hidden = torch.squeeze(hidden[0]) # hidden states, the 2nd is cell states

        if lengths is not None:
            output = unpack(output)[0]

        if self.temp_max_pool:
            hidden = torch.max(output, 0)[0].squeeze(0)
        elif self.scaled_dot_attn:
            # add scaled dot product attention

            # enc_output = output[-1]
            enc_output = torch.squeeze(hidden)
            # (batch_size, hidden_state)
            keys = torch.mm(enc_output, self.key_w) / np.sqrt(self.hidden_size)

            # (time, batch_size, hidden_state) * (1, batch_size, hidden_state)
            keys = torch.sum(keys.view(1, -1, self.hidden_size) * output, 2)

            batch_mask = self.create_mask(lengths)

            # masked_keys = keys * Variable(move_to_cuda(batch_mask))  # taking masked parts to be 0

            exp_maksed_keys = self.exp_mask(keys, batch_mask)

            # (time, batch_size) -> (batch_size, time)
            keys = torch.nn.Softmax()(torch.transpose(exp_maksed_keys, 0, 1))  # boradcast?
            keys_view = torch.transpose(keys, 0, 1).contiguous().view(output.size()[0], output.size()[1], 1)
            # (time, batch_size, 1) * (time, batch_size, hidden_state)
            output = torch.sum(output * keys_view, 0)

            return self.out(output), keys  # (batch_size, time)

        return self.out(hidden)


to_prob = nn.Sigmoid()


def preds_to_sparse_matrix(indices, batch_size, label_size):
    # this is for preds
    # indices will be a list: [[0, 0, 0], [0, 0, 1], ...]
    labels = np.zeros((batch_size, label_size))
    for b, l in indices:
        labels[b, l] = 1.
    return labels

def output_to_preds(output):
    return (output > 0.5)

def sparse_one_hot_mat_to_indices(preds):
    return preds.nonzero()

def condense_preds(indicies, batch_size):
    # can condense both preds and y
    a = [[] for _ in range(batch_size)]
    for b, l in indicies:
        a[b].append(str(l))
    condensed_preds = []
    for labels in a:
        condensed_preds.append("-".join(labels))
    assert len(condensed_preds) == len(a)

    return condensed_preds


def eval_model(model, valid_iter, save_pred=False):
    # when test_final is true, we save predictions
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0

    all_preds = []
    all_condensed_preds = []
    all_y_labels = []
    all_condensed_ys = []
    all_orig_texts = []
    all_keys = []

    for data in valid_iter:
        (x, x_lengths), y = data.Text, data.Description
        if args.attn:
            output, keys = model(x, x_lengths)
            all_keys.extend(keys.data.cpu().numpy().tolist())
        else:
            output = model(x)
        loss = criterion(output, y)
        total_loss += loss.data[0] * x.size(1)

        batch_size = x.size(1)

        # preds = output.data.max(1)[1]  # already taking max...I think, max returns a tuple
        preds = output_to_preds(output)
        preds_indices = sparse_one_hot_mat_to_indices(preds)

        sparse_preds = preds_to_sparse_matrix(preds_indices.data.cpu().numpy(), batch_size, model.nclasses)

        all_preds.append(sparse_preds)
        all_y_labels.append(y.data.cpu().numpy())

        # we rebuild preds for [1-2-5, 10-10-1]
        # and turn y into [1-2-5, 10-10-1] as well

        # correct += preds.eq(y.data).cpu().sum()

        correct += metrics.accuracy_score(y.data.cpu().numpy(), sparse_preds)
        cnt += 1  # accuracy is average already, so we just count number of batches

        orig_text = TEXT.reverse(x.data)
        all_orig_texts.extend(orig_text)

        if save_pred:
            y_indices = sparse_one_hot_mat_to_indices(y)
            condensed_preds = condense_preds(preds_indices.data.cpu().numpy().tolist(), batch_size)
            condensed_ys = condense_preds(y_indices.data.cpu().numpy().tolist(), batch_size)

            all_condensed_preds.extend(condensed_preds)
            all_condensed_ys.extend(condensed_ys)

    multiclass_f1_msg = 'Multiclass F1 - '

    preds = np.vstack(all_preds)
    ys = np.vstack(all_y_labels)

    # both should be giant sparse label matrices
    f1_by_label = metrics.f1_score(ys, preds, average=None)
    for i, f1_value in enumerate(f1_by_label.tolist()):
        multiclass_f1_msg += labels[i] + ": " + str(f1_value) + " "

    logging.info(multiclass_f1_msg)

    logging.info("\n" + metrics.classification_report(ys, preds))

    if save_pred:
        assert len(all_condensed_ys) == len(all_condensed_preds) == len(all_orig_texts)

        import csv
        # we store things out, hopefully they are in correct order
        with open(pjoin(args.run_dir, 'confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in zip(all_condensed_preds, all_condensed_ys, all_orig_texts):
                writer.writerow({'preds': pair[0], 'labels': pair[1], 'text': pair[2]})
        with open(pjoin(args.run_dir, 'attn_map.json'), 'wb') as f:
            json.dump([all_condensed_preds, all_condensed_ys, all_orig_texts, all_keys], f)
        with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
            json.dump(label_list, f)

    model.train()
    return correct / cnt


def train_module(model, optimizer,
                 train_iter, valid_iter, test_iter, max_epoch):
    model.train()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    exp_cost = None
    end_of_epoch = True  # False  # set true because we want immediate feedback...

    best_valid = 0.
    epoch = 1

    for n in range(max_epoch):
        iter = 0
        for data in train_iter:
            iter += 1

            model.zero_grad()
            (x, x_lengths), y = data.Text, data.Description

            if args.attn:
                output, keys = model(x, x_lengths)
            else:
                output = model(x)

            loss = criterion(output, y)  # y, output should be the same shape
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = loss.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

            if iter % 100 == 0:
                logging.info("iter {} lr={} train_loss={} exp_cost={} \n".format(iter,
                                                                                 optimizer.param_groups[0]['lr'],
                                                                                 loss.data[0],
                                                                                 exp_cost))

        valid_accu = eval_model(model, valid_iter)
        sys.stdout.write("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
            epoch,
            optimizer.param_groups[0]['lr'],
            loss.data[0],
            valid_accu
        ))

        epoch += 1

        if valid_accu > best_valid:
            best_valid = valid_accu

        sys.stdout.write("\n")


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def init_emb(vocab, init="randn", num_special_toks=2):
    # we can try randn or glorot
    # mode="unk"|"all", all means initialize everything
    emb_vectors = vocab.vectors
    sweep_range = len(vocab)
    running_norm = 0.
    num_non_zero = 0
    total_words = 0
    for i in range(num_special_toks, sweep_range):
        if len(emb_vectors[i, :].nonzero()) == 0:
            # std = 0.5 is based on the norm of average GloVE word vectors
            if init == "randn":
                torch.nn.init.normal(emb_vectors[i], mean=0, std=0.5)
        else:
            num_non_zero += 1
            running_norm += torch.norm(emb_vectors[i])
        total_words += 1
    logger.info("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
        running_norm / num_non_zero, num_non_zero, total_words))


if __name__ == '__main__':
    import spacy

    # spacy_en = spacy.load('en')
    spacy_en = spacy.load('en_core_web_sm')

    labels = range(0, 18)  # 1 to 18 but now we processed to be 0 to 17 :)

    # with open('../../data/clinvar/text_classification_db_labels.json', 'r') as f:
    #     labels = json.load(f)

    labels = {}
    for true_label in range(1, 19):
        labels[str(true_label)] = true_label - 1  # actual label we see

    # # map labels to list
    label_list = [None] * len(labels)
    for k, v in labels.items():
        label_list[v] = k

    labels = label_list
    logger.info("available labels: ")
    logger.info(labels)

    TEXT = ReversibleField(sequential=True, tokenize=tokenizer, include_lengths=True, lower=False)

    LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=18, tensor_type=torch.FloatTensor)

    if args.dataset == 'major':
        train, val, test = data.TabularDataset.splits(
            path='../../data/csu/', train='maj_label_train.tsv',
            validation='maj_label_valid.tsv',
            test='maj_label_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])
    elif args.dataset == 'multi':
        train, val, test = data.TabularDataset.splits(
            path='../../data/csu/', train='multi_label_train.tsv',
            validation='multi_label_valid.tsv',
            test='multi_label_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])
    elif args.dataset == 'multi_no_des':
        train, val, test = data.TabularDataset.splits(
            path='../../data/csu/', train='multi_label_no_des_train.tsv',
            validation='multi_label_no_des_valid.tsv',
            test='multi_label_no_des_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])

    if args.emb_dim == 100:
        TEXT.build_vocab(train, vectors="glove.6B.100d")
    elif args.emb_dim == 200:
        TEXT.build_vocab(train, vectors="glove.6B.200d")
    elif args.emb_dim == 300:
        TEXT.build_vocab(train, vectors="glove.6B.300d")
    else:
        TEXT.build_vocab(train, vectors="glove.6B.100d")

    # do repeat=False
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
        batch_sizes=(32, 256, 256), device=args.gpu,
        sort_within_batch=True, repeat=False)  # stop infinite runs
    # if not labeling sort=False, then you are sorting through valid and test

    vocab = TEXT.vocab

    if args.rand_unk:
        init_emb(vocab, init="randn")

    # so now all you need to do is to create an iterator

    model = Model(vocab, nclasses=len(labels), emb_dim=args.emb_dim,
                  scaled_dot_attn=args.attn, hidden_size=args.d,
                  depth=args.depth)
    if torch.cuda.is_available():
        model.cuda(args.gpu)

    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr=0.001)

    train_module(model, optimizer, train_iter, val_iter, test_iter,
                 max_epoch=args.max_epoch)

    test_accu, test_sparsity_coherence_cost = eval_model(model, test_iter, save_pred=True)
    logger.info("final test accu: {}, test sparsity cost: {}".format(test_accu, test_sparsity_coherence_cost))
