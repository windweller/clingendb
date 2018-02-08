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

import logging

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--dataset", type=str, default='merged', help="merged|sub_sum, merged is the better one")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--unroll_size", type=int, default=35)
argparser.add_argument("--max_epoch", type=int, default=5)
argparser.add_argument("--d", type=int, default=910)
argparser.add_argument("--dropout", type=float, default=0.3,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--rnn_dropout", type=float, default=0.2,
                       help="dropout of RNN layers")
argparser.add_argument("--depth", type=int, default=1)
argparser.add_argument("--lr", type=float, default=1.0)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./exp')
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--gpu", type=int, default=-1)
argparser.add_argument("--rand_unk", action="store_true", help="randomly initialize unk")
argparser.add_argument("--emb_update", action="store_true", help="update embedding")

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
        return th_var.cuda()
    else:
        return th_var


class Model(nn.Module):
    def __init__(self, vocab, emb_dim=100, hidden_size=256, depth=1, nclasses=5):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.drop = nn.Dropout(0.2)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_size,
            depth,
            dropout=0.2,
            bidirectional=False)  # ha...not even bidirectional
        d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)  # include bias, to prevent bias assignment
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if args.emb_update else False

    def forward(self, input, lengths=None):
        output_vecs = self.get_vectors(input, lengths)
        return self.get_logits(output_vecs)

    def exp_mask(self, val, mask, no_var=False):
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
        if not no_var:
            exp_mask = Variable((1 - mask) * VERY_NEGATIVE_NUMBER, requires_grad=False)
            return val + move_to_cuda(exp_mask)
        else:
            exp_mask = (1 - mask) * VERY_NEGATIVE_NUMBER
            return val + exp_mask  # no longer on cuda

    def create_mask(self, lengths):
        # lengths would be a python list here, not a Tensor
        # [max_len, batch_size]
        masks = np.ones([max(lengths), len(lengths)], dtype='float32')
        for i, l in enumerate(lengths):
            masks[l:, i] = 0.
        return torch.from_numpy(masks)

    def get_vectors(self, input, lengths=None):
        embed_input = self.embed(input)

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output)[0]

        # MUST apply negative mapping, so max pooling will not take padding elements
        batch_mask = self.create_mask(lengths)  # (time, batch_size)
        batch_mask = batch_mask.view(-1, len(lengths), 1)  # not sure if here broadcasting is right
        output = self.exp_mask(output, batch_mask)  # now pads will never be chosen...

        return output

    def get_logits(self, output_vec):
        output = torch.max(output_vec, 0)[0].squeeze(0)
        return self.out(output)

    def get_softmax_weight(self):
        return self.out.weight

    def get_weight_map(self, sent_vec, normalize='local'):
        # normalize: 'local' | 'global'
        # global norm, local select will call this function twice...
        # Warning: old method, do not use

        # sent_vec: (batch_size, vec_dim)
        s_weight = self.get_softmax_weight()
        # multiply: sent_vec * s_weight
        # (batch_size, vec_dim) * (vec_dim, label_size)

        # so we reshape (this is verified)
        # (batch_size, 1, vec_dim) * (1, label_size, vec_dim)
        # result: (batch_size, label_size, vec_dim)
        weight_map = sent_vec.view(-1, 1, self.hidden_size) * s_weight.view(1, self.nclasses, self.hidden_size)

        # normalize along label_size
        norm_dim = 1 if normalize == 'local' else 2
        n_weight_map = nn.Softmax(dim=norm_dim)(weight_map)

        # we return for every single time step that's used, a label distribution
        # then in visualization we can cut it out, meddle however we want
        return n_weight_map

    def get_time_contrib_map(self, output):

        seq_len, batch_size, _ = output.size()

        sent_vec, indices = torch.max(output, 0)

        s_weight = self.get_softmax_weight()
        # s_weight: (hidden_size, n_classes)
        # sent_vec: (batch_size, hidden_size)

        weight_map = sent_vec.view(-1, self.hidden_size, 1) * s_weight.view(1, self.hidden_size, self.nclasses)
        # (batch_size, vec_dim, label_size)
        # note this shape is DIFFERENT from the other method
        # weight_map is un-normalized

        # this by default gets FloatTensor
        assignment_dist = torch.zeros(batch_size, seq_len, self.nclasses)

        for b in range(batch_size):
            # (we just directly sum the influence, if the timestep is already in there!)
            time_steps = set()
            for d in range(self.hidden_size):
                time_step = indices.data.cpu()[b, d]
                if time_step in time_steps:
                    # sum of history happens here
                    assignment_dist[b, time_step, :] += weight_map.data.cpu()[b, d, :]
                else:
                    assignment_dist[b, time_step, :] = weight_map.data.cpu()[b, d, :]

        # return (batch, time, label_size), not all time steps are filled up, many are 0
        return assignment_dist

    def directional_norm(self, contrib_map):
        # in fact this is just unitize
        # use non-masked contrib-map
        b, t, l = contrib_map.size()

        # (batch_size, time, label-size)
        normed_denom = torch.norm(contrib_map, p=1, dim=1).view(b, 1, l)  # unitize
        # we should get one summed up value per time dim

        normed_contrib_map = contrib_map / normed_denom

        return normed_contrib_map

    def get_tensor_credit_assignment(self, contrib_map):
        # also sum of history...
        # assign by-label credits, no conflicts
        # return [batch_size, time_step, label_size]

        normed_contrib_map = self.directional_norm(contrib_map)  # global unitized, now directional!

        return normed_contrib_map

    def get_tensor_label_attr(self, contrib_map):

        # we reuse the exp_mask function, but need to have custom mask
        mask = (contrib_map != 0.).type(
            torch.FloatTensor)  # surprisingly your exp_mask function asks mask to indicate what's there...
        zero_mask = 1 - mask

        # here, we choose to do local and global normalization, so we return 2 contrib_maps
        # this is only used for global normalization
        masked_contrib_map = self.exp_mask(contrib_map, mask, no_var=True)  # no need to put mask as variable
        # because contrib_map itself is not variable

        # then do local normalization, and multiply by zero_mask to cancel weights from 0 time steps
        # emmm, this is correct

        # (batch, time, label_size)
        global_normed_contrib_map = nn.Softmax(dim=1)(Variable(masked_contrib_map)).data
        local_normed_contrib_map = nn.Softmax(dim=2)(Variable(masked_contrib_map)).data # local masks still matter
        local_normed_contrib_map *= zero_mask  # to make sure [0,0,0] will not be [0.3, 0.3, 0.3]
        # masked contrib map gives correct weight assignment

        return global_normed_contrib_map, local_normed_contrib_map

    def get_visualization_tensor_max_assignment(self, output):
        # NOTE: this returns a tensor, not a dictionary

        # indices = get_label_attribution()
        # return [batch_size, time_step, label_distribution]
        # for time_step with no assignment, we do [0, 0, ..., 0]
        # this method takes a while...not optimized for GPU

        # n_weight_map: (batch_size, label_size, vec_dim)
        seq_len, batch_size, _ = output.size()
        assignment_dist = torch.zeros(batch_size, seq_len, self.nclasses)

        sent_vec, indices = torch.max(output, 0)
        n_weight_map = self.get_weight_map(sent_vec)

        # we record how many times each dimension voted to DIFFERENT candidate/label
        records = torch.zeros(batch_size, seq_len)
        # records how often we actually reassign when the label changed
        reassign_records = torch.zeros(batch_size, seq_len)
        # votes distribution: indices (how many votes per word has)
        votes_dist = torch.zeros(batch_size, seq_len)

        # not sure if there's Torch tensor-op that will solve this
        for b in range(batch_size):
            # this is per-example, we record used time_steps
            time_steps = set()
            for d in range(self.hidden_size):
                # get the time_step of each dim
                time_step = indices.data.cpu()[b, d]
                # add to votes distributions
                votes_dist[b, time_step] += 1
                if time_step not in time_steps:
                    assignment_dist[b, time_step, :] = n_weight_map.data.cpu()[b, :, d]
                    time_steps.add(time_step)
                else:
                    # this part is supposed to compare new assignment to prev one
                    # and find the most influential:
                    # compare the max proportion of the two assignments, and go with
                    # whichever one is larger...
                    old_assignment = assignment_dist[b, time_step, :]
                    new_assignment = n_weight_map.data.cpu()[b, :, d]

                    # assignment is a vector (1-dim)
                    old_max_contribution, old_label_index = torch.max(old_assignment, 0)
                    new_max_contribution, new_label_index = torch.max(new_assignment, 0)

                    if (new_max_contribution > old_max_contribution).numpy():
                        assignment_dist[b, time_step, :] = n_weight_map.data.cpu()[b, :, d]

                    # we investigate if this time step voted for a different candidate
                    if (old_label_index != new_label_index).numpy():
                        records[b, time_step] += 1  # because it voted for a different label
                        if (new_max_contribution > old_max_contribution).numpy():
                            reassign_records[b, time_step] += 1

        return assignment_dist, records, reassign_records, votes_dist


def get_multiclass_recall(preds, y_label):
    # preds: (label_size), y_label; (label_size)
    label_cat = range(len(label_list))
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of the correct labels
        idx_of_cat = y_label == la
        cat_preds = preds[idx_of_cat]
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def get_multiclass_prec(preds, y_label):
    label_cat = range(len(label_list))
    labels_accu = {}

    for la in label_cat:
        # for each label, we get the index of predictions
        idx_of_cat = preds == la
        cat_preds = y_label[idx_of_cat]  # ground truth
        if cat_preds.size != 0:
            accu = np.mean(cat_preds == la)
            labels_accu[la] = [accu]
        else:
            labels_accu[la] = []

    return labels_accu


def cumulate_multiclass_accuracy(total_accu, labels_accu):
    for k, v in labels_accu.iteritems():
        total_accu[k].extend(v)


def get_mean_multiclass_accuracy(total_accu):
    new_dict = {}
    for k, v in total_accu.iteritems():
        new_dict[k] = np.mean(total_accu[k])
    return new_dict


def eval_model(model, valid_iter, save_pred=False):
    # when test_final is true, we save predictions
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0
    total_labels_recall = None

    total_labels_prec = None
    all_preds = []
    all_y_labels = []
    all_orig_texts = []
    # all_text_vis = []

    all_credit_assign = []
    all_la_global = []
    all_la_local = []

    # maximum amount of records would be 256...
    # all_records = []  # we record how many times each dimension voted to DIFFERENT candidate/label
    # all_reassign_records = []
    # all_votes_dist = []

    for data in valid_iter:
        (x, x_lengths), y = data.Text, data.Description

        output_vecs = model.get_vectors(x, x_lengths)
        output = model.get_logits(output_vecs)

        # (batch_size, time_step, label_dist)
        # label_assignment_tensor, records, reassign_records, votes_dist = model.get_visualization_tensor_max_assignment(
        #     output_vecs)
        # all_text_vis.extend(label_assignment_tensor.numpy().tolist())
        # all_records.extend(records.numpy().tolist())
        # all_reassign_records.extend(reassign_records.numpy().tolist())
        # all_votes_dist.extend(votes_dist.numpy().tolist())

        if save_pred:
            contrib_map = model.get_time_contrib_map(output_vecs)
            credit_assign = model.get_tensor_credit_assignment(contrib_map)
            global_map, local_map = model.get_tensor_label_attr(contrib_map)

            all_credit_assign.extend(credit_assign.numpy().tolist())
            all_la_global.extend(global_map.numpy().tolist())
            all_la_local.extend(local_map.numpy().tolist())

        loss = criterion(output, y)
        total_loss += loss.data[0] * x.size(1)
        preds = output.data.max(1)[1]  # already taking max...I think, max returns a tuple
        correct += preds.eq(y.data).cpu().sum()
        cnt += y.numel()

        orig_text = TEXT.reverse(x.data)
        all_orig_texts.extend(orig_text)

        if save_pred:
            all_preds.extend(preds.cpu().numpy().tolist())
            all_y_labels.extend(y.data.cpu().numpy().tolist())

        # compute multiclass
        labels_recall = get_multiclass_recall(preds.cpu().numpy(), y.data.cpu().numpy())
        labels_prec = get_multiclass_prec(preds.cpu().numpy(), y.data.cpu().numpy())
        if total_labels_recall is None:
            total_labels_recall = labels_recall
            total_labels_prec = labels_prec
        else:
            cumulate_multiclass_accuracy(total_labels_recall, labels_recall)
            cumulate_multiclass_accuracy(total_labels_prec, labels_prec)

            # valid and tests can stop themselves

    multiclass_recall_msg = 'Multiclass Recall - '
    mean_multi_recall = get_mean_multiclass_accuracy(total_labels_recall)

    for k, v in mean_multi_recall.iteritems():
        multiclass_recall_msg += labels[k] + ": " + str(v) + " "

    multiclass_prec_msg = 'Multiclass Precision - '
    mean_multi_prec = get_mean_multiclass_accuracy(total_labels_prec)

    for k, v in mean_multi_prec.iteritems():
        multiclass_prec_msg += labels[k] + ": " + str(v) + " "

    logging.info(multiclass_recall_msg)
    logging.info(multiclass_prec_msg)

    if save_pred:
        import csv
        with open(pjoin(args.run_dir, 'confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels', 'text', 'credit_assign', 'la_global', 'la_local']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in zip(all_preds, all_y_labels, all_orig_texts, all_credit_assign, all_la_global, all_la_local):
                writer.writerow({'preds': pair[0], 'labels': pair[1], 'text': pair[2], 'credit_assign': pair[3],
                                 'la_global': pair[4], 'la_local': pair[5]})

        with open(pjoin(args.run_dir, 'label_vis_map.json'), 'wb') as f:
            json.dump([all_preds, all_y_labels, all_orig_texts, all_credit_assign, all_la_global, all_la_local], f)

        with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
            json.dump(label_list, f)

    model.train()
    return correct / cnt


def train_module(model, optimizer,
                 train_iter, valid_iter, test_iter, max_epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()

    exp_cost = None
    end_of_epoch = True  # False  # set true because we want immediate feedback...
    iter = 0
    best_valid = 0.
    epoch = 1

    for n in range(max_epoch):
        for data in train_iter:
            iter += 1

            model.zero_grad()
            (x, x_lengths), y = data.Text, data.Description

            output = model(x, x_lengths)

            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = loss.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

            if iter % 100 == 0:
                logging.info("iter {} lr={} train_loss={} exp_cost={} \n".format(iter, optimizer.param_groups[0]['lr'],
                                                                                 loss.data[0], exp_cost))

        valid_accu = eval_model(model, valid_iter)
        sys.stdout.write("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
            epoch,
            optimizer.param_groups[0]['lr'],
            loss.data[0],
            valid_accu
        ))

        if valid_accu > best_valid:
            best_valid = valid_accu

        sys.stdout.write("\n")
        epoch += 1


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

    with open('../../data/clinvar/text_classification_db_labels.json', 'r') as f:
        labels = json.load(f)

    # map labels to list
    label_list = [None] * len(labels)
    for k, v in labels.items():
        label_list[v] = k

    labels = label_list
    logger.info("available labels: ")
    logger.info(labels)

    TEXT = data.ReversibleField(sequential=True, lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    if args.dataset == 'merged':
        train, val, test = data.TabularDataset.splits(
            path='../../data/clinvar/', train='merged_text_classification_db_train.tsv',
            validation='merged_text_classification_db_valid.tsv',
            test='merged_text_classification_db_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])
    else:
        train, val, test = data.TabularDataset.splits(
            path='../../data/clinvar/', train='text_classification_db_train.tsv',
            validation='text_classification_db_valid.tsv', test='text_classification_db_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])

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

    model = Model(vocab, nclasses=len(labels))
    if torch.cuda.is_available():
        model.cuda(args.gpu)

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
                 max_epoch=args.max_epoch)

    test_accu = eval_model(model, test_iter, save_pred=True)
    logger.info("final test accu: {}".format(test_accu))
