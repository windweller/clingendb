"""
Based on text_hierarchy
"""

"""
Leverage hierarchical loss
"""

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

from sklearn import metrics

from util import MultiLabelField, ReversibleField, BCEWithLogitsLoss, MultiMarginHierarchyLoss

import logging
import itertools

import sys
import json

from collections import defaultdict

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--dataset", type=str, default='multi_top_snomed_adjusted_no_des',
                       help="multi_top_snomed_no_des|multi_top_snomed_adjusted_no_des, merged is the better one")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--emb_dim", type=int, default=100)
argparser.add_argument("--max_epoch", type=int, default=5)
argparser.add_argument("--d", type=int, default=512)
argparser.add_argument("--dropout", type=float, default=0.2,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--depth", type=int, default=1)
argparser.add_argument("--lr", type=float, default=1.0)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./exp')
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--gpu", type=int, default=-1)
argparser.add_argument("--load_model", action="store_true", help="load model according to run_dir position")
argparser.add_argument("--rand_unk", action="store_true", help="randomly initialize unk")
argparser.add_argument("--emb_update", action="store_true", help="update embedding")
argparser.add_argument("--mc_dropout", action="store_true", help="use variational dropout at inference time")
argparser.add_argument("--dice_loss", action="store_true", help="use Dice loss to solve class imbalance, currently only implemented without extra penalty")
argparser.add_argument("--bidir", action="store_true", help="use bidirectional ")

argparser.add_argument("--reject", action="store_true", help="learn to reject")
argparser.add_argument("--reject_output", action="store_true", help="learn to reject by logit")
argparser.add_argument("--reject_hidden", action="store_true", help="learn to reject by hidden rep")
argparser.add_argument("--reject_delay", type=int, default=0, help="delay optimizing for rejection loss after certain epoch")
argparser.add_argument("--reject_anneal", type=float, default=1., help="at end of each epoch, we halve the rejection rate")
argparser.add_argument("--gamma", type=float, default=5., help="default rejection cost")

argparser.add_argument("--l2_penalty_softmax", type=float, default=0., help="add L2 penalty on softmax weight matrices")
argparser.add_argument("--l2_str", type=float, default=0, help="a scalar that reduces strength")  # 1e-3

argparser.add_argument("--prototype", action="store_true", help="use hierarchical loss")
argparser.add_argument("--softmax_hier", action="store_true", help="use hierarchical loss")
argparser.add_argument("--softmax_partial", action="store_true", help="use hierarchical loss")
argparser.add_argument("--max_margin", action="store_true", help="use hierarchical loss")

argparser.add_argument("--proto_str", type=float, default=1e-3, help="a scalar that reduces strength")
argparser.add_argument("--proto_out_str", type=float, default=1e-3, help="a scalar that reduces strength")
argparser.add_argument("--proto_cos", action="store_true", help="use cosine distance instead of dot product")
argparser.add_argument("--proto_maxout", action="store_true", help="maximize between-group distance")
argparser.add_argument("--proto_maxin", action="store_true", help="maximize in-group distance")
argparser.add_argument("--softmax_str", type=float, default=1e-2,
                       help="a scalar controls penalty for not falling in the group")
argparser.add_argument("--softmax_reward", type=float, default=0.1,
                       help="a scalar controls penalty for not falling in the group")
argparser.add_argument("--softmax_hier_prod_prob", action="store_true", help="use hierarchical loss")
argparser.add_argument("--softmax_hier_sum_logit", action="store_true", help="use hierarchical loss")
argparser.add_argument("--max_margin_neighbor", type=float, default=0.5,
                       help="maximum should be 1., ")

# 1. Softmax loss penalty is basically duplicating the "label"..and count loss twice
# 2. Prototype loss penalty is adding a closeness constraints (very natrual)
# 3. Max-margin loss penalty is margin-based

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

cos_sim = nn.CosineSimilarity(dim=0)


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = dice_coeff(probs, targets)
        score = 1 - score.sum() / num
        return score


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
        self.drop = nn.Dropout(args.dropout)  # embedding dropout
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_size,
            depth,
            dropout=args.dropout,
            bidirectional=args.bidir)  # ha...not even bidirectional
        d_out = hidden_size if not args.bidir else hidden_size * 2
        self.out = nn.Linear(d_out, nclasses)  # include bias, to prevent bias assignment
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if args.emb_update else False

        if args.reject:
            # we add more parameters
            # takes in logits or probs as input, and output whether to drop or not
            # reject based on the whole batch

            # (batch_size, label_size)
            reject_dim = nclasses if args.reject_output else d_out

            self.reject_model = nn.Sequential(
                nn.Linear(reject_dim, 1),
                nn.Sigmoid()
            )

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

        # this is new, not working ,remove it
        # embed_input = self.drop(embed_input)

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

    def directional_norm(self, contrib_map):
        # in fact this is just unitize
        # use non-masked contrib-map
        b, t, l = contrib_map.size()

        # (batch_size, time, label-size)
        normed_denom = torch.norm(contrib_map, p=1, dim=1).view(b, 1, l)  # unitize
        # we should get one summed up value per time dim

        normed_contrib_map = contrib_map / normed_denom

        return normed_contrib_map

def preds_to_sparse_matrix(indices, batch_size, label_size):
    # this is for preds
    # indices will be a list: [[0, 0, 0], [0, 0, 1], ...]
    labels = np.zeros((batch_size, label_size))
    for b, l in indices:
        labels[b, l] = 1.
    return labels


def output_to_prob(output):
    return torch.sigmoid(output)


def output_to_preds(output):
    return (torch.sigmoid(output) > 0.5)


def sparse_one_hot_mat_to_indices(preds):
    return preds.nonzero()


def condense_preds(indices, batch_size):
    # can condense both preds and y
    a = [[] for _ in range(batch_size)]
    for b, l in indices:
        a[b].append(str(l))
    condensed_preds = []
    for labels in a:
        condensed_preds.append("-".join(labels))
    assert len(condensed_preds) == len(a)

    return condensed_preds


# I think this is good now.
def generate_meta_y(indices, meta_label_size, batch_size):
    a = np.array([[0.] * meta_label_size for _ in range(batch_size)], dtype=np.float32)
    matched = defaultdict(set)
    for b, l in indices:
        if b not in matched:
            a[b, meta_label_mapping[str(l)]] = 1.
            matched[b].add(meta_label_mapping[str(l)])
        elif meta_label_mapping[str(l)] not in matched[b]:
            a[b, meta_label_mapping[str(l)]] = 1.
            matched[b].add(meta_label_mapping[str(l)])

    assert np.sum(a <= 1) == a.size

    return a

# TODO: this might be wrong. Because if prototype constraints work, this should work as well
def spread_by_meta_y(y, indices):
    # indices are still those where y labels exist
    matched = defaultdict(set)
    snomed_label = set()
    for b, l in indices:
        meta_label = meta_label_mapping[str(l)]
        snomed_label.add(l)
        if meta_label not in matched[b]:
            neighbors = neighbor_maps[str(l)]

            # this is preventing a top-label can have > 1 probability
            neighbors = [n for n in neighbors if n not in snomed_label]

            if len(neighbors) > 0:
                y[:, neighbors] = args.softmax_reward  # 0.1 # might be += instead of =
                matched[b].add(meta_label)  # in this batched example, this meta label is flagged
                # in an alternative setting, we can let it add :) as long as they are below 1

    y = torch.clamp(y, max=1., min=0.)
    return y


def eval_model(model, valid_iter, save_pred=False, save_viz=False):
    # when test_final is true, we save predictions
    if not args.mc_dropout:
        model.eval()

    criterion = BCEWithLogitsLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0

    all_scores = []  # probability to measure uncertainty
    all_preds = []
    all_y_labels = []
    all_print_y_labels = []
    all_orig_texts = []
    all_text_vis = []

    all_condensed_preds = []
    all_condensed_ys = []

    all_meta_preds = []
    all_meta_y_labels = []

    all_uncertainty = []

    # all_credit_assign = []
    # all_la_global = []
    # all_la_local = []

    iter = 0
    for data in valid_iter:
        (x, x_lengths), y = data.Text, data.Description
        iter += 1

        if iter % 5 == 0 and save_pred:
            logger.info("at iteration {}".format(iter))

        if not args.mc_dropout:
            output_vecs = model.get_vectors(x, x_lengths)
            final_rep = torch.max(output_vecs, 0)[0].squeeze(0)
            output = model.get_logits(output_vecs)  # this is logits, not sent to sigmoid yet!
        else:
            # [(batch_size, class_scores) * 10]
            # probs = []
            outputs = []
            # run 10 times per batch to get mc estimation
            for _ in xrange(10):
                output_vecs = model.get_vectors(x, x_lengths)
                output = model.get_logits(output_vecs)
                outputs += [output.data.cpu().numpy()]
                # probs += [output_to_prob(output).data.cpu().numpy()]  # remember this is batched!!!!

            output_mean = np.mean(outputs, axis=0)
            # predictive_mean = np.mean(probs, axis=0)
            # predictive_variance = np.var(probs, axis=0)
            predictive_variance = np.var(outputs, axis=0)

            # Note that we are NOT computing tau; this value is rather small
            # tau = l ** 2 * (1 - model.p) / (2 * N * model.weight_decay)
            # predictive_variance += tau ** -1

        batch_size = x.size(1)

        loss = criterion(output, y)
        total_loss += loss.data[0] * x.size(1)

        if not args.mc_dropout:
            # this is the dropping out y and output part
            if args.reject:
                if args.reject_output:
                    reject_scores = torch.squeeze(model.reject_model(output.detach()))
                elif args.reject_hidden:
                    reject_scores = torch.squeeze(model.reject_model(final_rep.detach()))
                    # so what we do here if to take out things

                # (batch_size) (prob)
                drop_choices = torch.bernoulli(reject_scores).data.cpu().numpy()  # 1 means drop, 0 means keep
                drop_rate = np.sum(drop_choices)

                drop_choices_list = drop_choices.tolist()
                assert len(drop_choices_list) == batch_size

                new_y = []
                new_output = []
                for i, d_c in enumerate(drop_choices_list):
                    if d_c == 0.:
                        new_y.append(y[i,:])
                        new_output.append(output[i,:])

                if len(new_y) == 0:
                    logging.info("rejected all examples")
                    return

                y = torch.stack(new_y, dim=0)
                output = torch.stack(new_output, dim=0)

                logging.info("number of examples dropped on eval set is {}".format(drop_rate))
                logging.info("drop chances of each example is: {}".format(reject_scores.data.cpu().numpy().tolist()))

            scores = output_to_prob(output).data.cpu().numpy()
            preds = output_to_preds(output)
        else:
            pt_output_mean = torch.from_numpy(output_mean)
            scores = output_to_prob(pt_output_mean).cpu().numpy()
            preds = output_to_preds(pt_output_mean)

        preds_indices = sparse_one_hot_mat_to_indices(preds)

        if not args.mc_dropout:
            sparse_preds = preds_to_sparse_matrix(preds_indices.data.cpu().numpy(), batch_size, model.nclasses)
        else:
            sparse_preds = preds_to_sparse_matrix(preds_indices.numpy(), batch_size, model.nclasses)

        all_scores.extend(scores.tolist())
        all_print_y_labels.extend(y.data.cpu().numpy().tolist())
        all_preds.append(sparse_preds)
        all_y_labels.append(y.data.cpu().numpy())

        if args.mc_dropout:
            all_uncertainty.extend(predictive_variance.tolist())

        # Generate meta-y labels and meta-level predictions
        # meta-y label is:
        # meta_y = generate_meta_y(y_indices.data.cpu().numpy().tolist(), meta_label_size, batch_size)
        # meta-level prediction needs additional code

        correct += metrics.accuracy_score(y.data.cpu().numpy(), sparse_preds)
        cnt += 1

        orig_text = TEXT.reverse(x.data)
        all_orig_texts.extend(orig_text)

        if save_pred:
            y_indices = sparse_one_hot_mat_to_indices(y)
            if not args.mc_dropout:
                condensed_preds = condense_preds(preds_indices.data.cpu().numpy().tolist(), batch_size)
            else:
                condensed_preds = condense_preds(preds_indices.numpy().tolist(), batch_size)
            condensed_ys = condense_preds(y_indices.data.cpu().numpy().tolist(), batch_size)

            all_condensed_preds.extend(condensed_preds)
            all_condensed_ys.extend(condensed_ys)

    preds = np.vstack(all_preds)
    ys = np.vstack(all_y_labels)

    # both are only true if it's for test, this saves sysout
    logging.info("\n" + metrics.classification_report(ys, preds))

    if save_pred:
        # So the format for each entry is: y = [], pred = [], for all labels
        # we also need

        with open(pjoin(args.run_dir, 'label_vis_map.json'), 'wb') as f:
            # used to be : all_condensed_preds, all_condensed_ys, all_scores, all_print_y_labels
            json.dump([all_condensed_preds, all_condensed_ys, all_scores, all_uncertainty, all_print_y_labels, all_orig_texts], f)

        with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
            json.dump(labels, f)

        # save model
        torch.save(model, pjoin(args.run_dir, 'model.pickle'))

    if save_viz:
        with open(pjoin(args.run_dir, 'label_vis_map.json'), 'wb') as f:
            json.dump([all_condensed_preds, all_condensed_ys, all_orig_texts, all_text_vis], f)

        with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
            json.dump(labels, f)

    return correct / cnt

def eval_adobe(model, adobe_iter, save_pred=False, save_viz=False):
    # when test_final is true, we save predictions
    if not args.mc_dropout:
        model.eval()

    criterion = BCEWithLogitsLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0

    all_scores = []  # probability to measure uncertainty
    all_preds = []
    all_y_labels = []
    all_print_y_labels = []
    all_orig_texts = []
    all_text_vis = []

    all_condensed_preds = []
    all_condensed_ys = []

    all_meta_preds = []
    all_meta_y_labels = []

    all_uncertainty = []

    # all_credit_assign = []
    # all_la_global = []
    # all_la_local = []

    logger.info("Evaluating on Adobe dataset")

    iter = 0
    for data in adobe_iter:
        (x, x_lengths), y = data.Text, data.Description
        iter += 1

        if iter % 5 == 0 and save_pred:
            logger.info("at iteration {}".format(iter))

        if not args.mc_dropout:
            output_vecs = model.get_vectors(x, x_lengths)
            final_rep = torch.max(output_vecs, 0)[0].squeeze(0)
            output = model.get_logits(output_vecs)  # this is logits, not sent to sigmoid yet!
        else:
            # [(batch_size, class_scores) * 10]
            outputs = []
            # run 10 times per batch to get mc estimation
            for _ in xrange(10):
                output_vecs = model.get_vectors(x, x_lengths)
                output = model.get_logits(output_vecs)
                outputs += [output.data.cpu().numpy()]
                # probs += [output_to_prob(output).data.cpu().numpy()]  # remember this is batched!!!!

            output_mean = np.mean(outputs, axis=0)
            # predictive_mean = np.mean(probs, axis=0)
            # predictive_variance = np.var(probs, axis=0)
            predictive_variance = np.var(outputs, axis=0)

            # Note that we are NOT computing tau; this value is rather small
            # tau = l ** 2 * (1 - model.p) / (2 * N * model.weight_decay)
            # predictive_variance += tau ** -1

            # if save_pred:
            # credit_assign = model.get_tensor_credit_assignment(output_vecs)
            # global_map, local_map = model.get_tensor_label_attr(output_vecs)

            # all_credit_assign.extend(credit_assign.numpy().tolist())
            # all_la_global.extend(global_map.numpy().tolist())
            # all_la_local.extend(local_map.numpy().tolist())

        batch_size = x.size(1)

        loss = criterion(output, y)
        total_loss += loss.data[0] * x.size(1)

        if not args.mc_dropout:
            if args.reject:
                if args.reject_output:
                    reject_scores = torch.squeeze(model.reject_model(output.detach()))
                elif args.reject_hidden:
                    reject_scores = torch.squeeze(model.reject_model(final_rep.detach()))
                    # so what we do here if to take out things

                # (batch_size) (prob)
                drop_choices = torch.bernoulli(reject_scores).data.cpu().numpy()  # 1 means drop, 0 means keep
                drop_rate = np.mean(drop_choices)

                drop_choices_list = drop_choices.tolist()
                assert len(drop_choices_list) == batch_size

                new_y = []
                new_output = []
                for i, d_c in enumerate(drop_choices_list):
                    if d_c == 0.:
                        new_y.append(y[i,:])
                        new_output.append(output[i,:])
                y = torch.stack(new_y, dim=0)
                output = torch.stack(new_output, dim=0)

                logging.info("drop rate on eval set is {}".format(drop_rate))

            scores = output_to_prob(output).data.cpu().numpy()
            preds = output_to_preds(output)
        else:
            pt_output_mean = torch.from_numpy(output_mean)
            scores = output_to_prob(pt_output_mean).cpu().numpy()
            preds = output_to_preds(pt_output_mean)

        preds_indices = sparse_one_hot_mat_to_indices(preds)

        if not args.mc_dropout:
            sparse_preds = preds_to_sparse_matrix(preds_indices.data.cpu().numpy(), batch_size, model.nclasses)
        else:
            sparse_preds = preds_to_sparse_matrix(preds_indices.numpy(), batch_size, model.nclasses)

        all_scores.extend(scores.tolist())
        all_print_y_labels.extend(y.data.cpu().numpy().tolist())
        all_preds.append(sparse_preds)
        all_y_labels.append(y.data.cpu().numpy())

        if args.mc_dropout:
            all_uncertainty.extend(predictive_variance.tolist())

        # Generate meta-y labels and meta-level predictions
        # meta-y label is:
        # meta_y = generate_meta_y(y_indices.data.cpu().numpy().tolist(), meta_label_size, batch_size)
        # meta-level prediction needs additional code

        # TODO: this is possibly incorrect?...not that we are using accuracy...
        correct += metrics.accuracy_score(y.data.cpu().numpy(), sparse_preds)
        cnt += 1

        orig_text = TEXT.reverse(x.data)
        all_orig_texts.extend(orig_text)

        if save_pred:
            y_indices = sparse_one_hot_mat_to_indices(y)
            if not args.mc_dropout:
                condensed_preds = condense_preds(preds_indices.data.cpu().numpy().tolist(), batch_size)
            else:
                condensed_preds = condense_preds(preds_indices.numpy().tolist(), batch_size)
            condensed_ys = condense_preds(y_indices.data.cpu().numpy().tolist(), batch_size)

            all_condensed_preds.extend(condensed_preds)
            all_condensed_ys.extend(condensed_ys)

    preds = np.vstack(all_preds)
    ys = np.vstack(all_y_labels)

    # both are only true if it's for test, this saves sysout
    logging.info("\n" + metrics.classification_report(ys, preds))

    if save_pred:
        # So the format for each entry is: y = [], pred = [], for all labels
        # we also need
        with open(pjoin(args.run_dir, 'adobe_label_vis_map.json'), 'wb') as f:
            # used to be : all_condensed_preds, all_condensed_ys, all_scores, all_print_y_labels
            json.dump([all_condensed_preds, all_condensed_ys, all_scores, all_uncertainty, all_print_y_labels, all_orig_texts], f)

    if save_viz:
        with open(pjoin(args.run_dir, 'adobe_label_vis_map.json'), 'wb') as f:
            json.dump([all_condensed_preds, all_condensed_ys, all_orig_texts, all_text_vis], f)

    return correct / cnt


prob_threshold = nn.Threshold(1e-5, 0)


def train_module(model, optimizer,
                 train_iter, valid_iter, max_epoch):
    model.train()
    criterion = BCEWithLogitsLoss(reduce=False)

    exp_cost = None
    end_of_epoch = True  # False  # set true because we want immediate feedback...

    best_valid = 0.
    epoch = 1

    training_rejectiong_rate = []

    softmax_weight = model.get_softmax_weight()

    for n in range(max_epoch):
        iter = 0
        for data in train_iter:
            iter += 1

            model.zero_grad()
            (x, x_lengths), y = data.Text, data.Description

            if args.reject:
                output_vec = model.get_vectors(x, x_lengths)  # this is just logit (before calling sigmoid)
                final_rep = torch.max(output_vec, 0)[0].squeeze(0)
                output = model.get_logits(output_vec)
            else:
                output = model(x, x_lengths)

            if args.prototype:
                loss = criterion(output, y)
                hierarchy_inner_penalty = 0.
                hierarchy_outer_penalty = 0.
                proto_count = 0.
                # prototype constraints are pair-wise dot product (cosine similarities)
                for meta_i in range(meta_label_size):
                    grouped_indices = label_grouping[str(meta_i)]
                    for pair_a, pair_b in itertools.combinations(grouped_indices, 2):
                        # compute dot product
                        if args.proto_cos:
                            # dist = 1 - cos_sim
                            hierarchy_inner_penalty += 1 - cos_sim(softmax_weight[:, pair_a], softmax_weight[:, pair_b])
                        else:
                            hierarchy_inner_penalty += torch.dot(softmax_weight[:, pair_a], softmax_weight[:, pair_b])
                        proto_count += 1
                hierarchy_inner_penalty = hierarchy_inner_penalty / proto_count  # average

                if args.proto_maxout:
                    # we only compute it when it has max_out flag
                    for label_j in range(label_size):
                        non_neighbor_indices = non_neighbor_maps[str(label_j)]
                        outer_vec = torch.sum(softmax_weight[:, non_neighbor_indices], dim=1) / len(
                            non_neighbor_indices)
                        hierarchy_outer_penalty += torch.dot(softmax_weight[:, label_j], outer_vec)

                if args.proto_maxin:
                    # maximize inner distance as well
                    loss = loss.mean() + hierarchy_inner_penalty * args.proto_str
                else:
                    # multiply a scalar, and maximize this value
                    loss = loss.mean() - hierarchy_inner_penalty * args.proto_str + hierarchy_outer_penalty * args.proto_out_str

                # add L2 penalty on prototype vectors
                # loss += softmax_weight.norm(2, dim=0).sum() * args.generate

                # instead, we do TensorFlow convention
                loss += softmax_weight.pow(2).sum() / 2 * args.l2_penalty_softmax

                loss.backward()
            elif args.softmax_hier:
                # compute loss for the higher level
                # (Batch, n_classes)

                # this is now logit, not
                if args.softmax_hier_prod_prob:
                    snomed_values = output_to_prob(output)
                elif args.softmax_hier_sum_logit:  # this should be our default approach
                    snomed_values = torch.max(output, move_to_cuda(Variable(torch.zeros(1))))  # max(x, 0)
                else:
                    raise Exception("Must flag softmax_hier_sum_prob or softmax_hier_sum_logit")

                batch_size = x.size(1)
                meta_probs = []

                # sum these prob into meta group
                for i in range(meta_label_size):
                    if args.softmax_hier_sum_logit:
                        meta_probs.append(snomed_values[:, label_grouping[str(i)]].sum(1))
                    elif args.softmax_hier_prod_prob:
                        # 1 - (1 - p_1)(...)(1 - p_n)
                        meta_prob = (1 - snomed_values[:, label_grouping[str(i)]]).prod(1)
                        # threshold at 1e-5
                        meta_probs.append(prob_threshold(meta_prob))  # we don't want really small probability

                meta_probs = torch.stack(meta_probs, dim=1)

                assert meta_probs.size(1) == meta_label_size

                # clamp everything to be between 0 and 1 for prob
                if args.softmax_hier_prod_prob:
                    meta_probs = torch.clamp(meta_probs, min=0., max=1.)

                # generate meta-label
                y_indices = sparse_one_hot_mat_to_indices(y)
                meta_y = generate_meta_y(y_indices.data.cpu().numpy().tolist(), meta_label_size, batch_size)
                meta_y = move_to_cuda(Variable(torch.from_numpy(meta_y)))

                loss = criterion(output, y).mean()  # original loss
                meta_loss = criterion(meta_probs, meta_y).mean()  # hierarchy loss
                loss += meta_loss * args.softmax_str
            else:
                loss = criterion(output, y)

                per_example_loss = loss.mean(dim=1).data.cpu().numpy().tolist()

                if epoch > args.reject_delay:
                    # detach the input because we don't want loss backprop into the representation
                    if args.reject_output:
                        s = torch.squeeze(model.reject_model(output.detach()))  # (batch_size)

                        # per example loss; average across all labels
                        loss = (1 - s) * loss.mean(dim=1) + s * args.gamma
                        # collect average rejection size
                        training_rejectiong_rate.append(s.mean().data[0])
                    elif args.reject_hidden:
                        s = torch.squeeze(model.reject_model(final_rep.detach()))  # (batch_size)

                        # per example loss; average across all labels
                        loss = (1 - s) * loss.mean(dim=1) + s * args.gamma
                        # collect average rejection size
                        training_rejectiong_rate.append(s.mean().data[0])
                else:
                    loss = loss.mean(dim=1)

            loss = loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = loss.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

            if iter % 100 == 0:
                avg_rej_rate = sum(training_rejectiong_rate) / float(len(training_rejectiong_rate))

                logging.info("iter {} lr={} train_loss={} exp_cost={} rej={} \n".format(iter, optimizer.param_groups[0]['lr'],
                                                                                 loss.data[0], exp_cost, avg_rej_rate))
                logging.info("per-example rej: {}".format(s.data.cpu().numpy().tolist()))
                logging.info("per-example loss: {}".format(per_example_loss))

        if epoch > args.reject_delay:
            args.gamma = args.reject_anneal * args.gamma
            logging.info("anneal gamma to {}".format(args.gamma))

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

    with open('../../data/csu/snomed_label_to_meta_neighbors.json', 'rb') as f:
        neighbor_maps = json.load(f)
        # {0: [12, 34, 13]}
        # turn this into neighbor matrices.
        # Remember in this way,
        # if a label in a group appears together, we are over-adding. From 1 to 1.2

        # be careful of row/column
        neighbor_mat_np = np.zeros((len(neighbor_maps), len(neighbor_maps)), dtype="float32")
        for nei_i in range(len(neighbor_maps)):
            ns = neighbor_maps[str(nei_i)]
            neighbor_mat_np[nei_i][ns] = 0.01  # args.softmax_str

        neighbor_mat = move_to_cuda(Variable(torch.from_numpy(neighbor_mat_np), requires_grad=False))

    with open('../../data/csu/snomed_label_to_meta_non_neighbors.json', 'rb') as f:
        non_neighbor_maps = json.load(f)

    with open('../../data/csu/snomed_label_to_meta_grouping.json', 'rb') as f:
        label_grouping = json.load(f)
        # {0: [12, 34, 13],
        meta_label_size = len(label_grouping)

    with open('../../data/csu/snomed_label_to_meta_map.json', 'rb') as f:
        meta_label_mapping = json.load(f)
        # {42: 14} maps snomed_indexed_label -> meta_labels

    with open('../../data/csu/snomed_labels_to_name.json', 'r') as f:
        labels = json.load(f)

    logger.info("available labels are: ")
    logger.info(labels)

    TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False)

    label_size = 42  # 18 if args.dataset != "multi_top_snomed_no_des" else 42

    LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=label_size, tensor_type=torch.FloatTensor)

    # load in adobe
    adobe_test = data.TabularDataset(path='../../data/csu/adobe_snomed_multi_label_no_des_test.tsv',
                                     format='tsv',
                                     fields=[('Text', TEXT), ('Description', LABEL)])

    if args.dataset == 'multi_top_snomed_no_des':
        train, val, test = data.TabularDataset.splits(
            path='../../data/csu/', train='snomed_multi_label_no_des_train.tsv',
            validation='snomed_multi_label_no_des_valid.tsv',
            test='snomed_multi_label_no_des_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])
    elif args.dataset == 'multi_top_snomed_adjusted_no_des':
        train, val, test = data.TabularDataset.splits(
            path='../../data/csu/', train='snomed_adjusted_multi_label_no_des_train.tsv',
            validation='snomed_adjusted_multi_label_no_des_valid.tsv',
            test='snomed_adjusted_multi_label_no_des_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])

    # actually, this is the first point of improvement: load in clinical embedding instead!!!
    TEXT.build_vocab(train, vectors="glove.6B.{}d".format(args.emb_dim))

    # do repeat=False
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
        batch_sizes=(32, 128, 128), device=args.gpu,
        sort_within_batch=True, repeat=False)  # stop infinite runs
    # if not labeling sort=False, then you are sorting through valid and test

    adobe_test_iter = data.Iterator(adobe_test, 128, sort_key=lambda x: len(x.Text),
                                    device=args.gpu, train=False, repeat=False, sort_within_batch=True)

    vocab = TEXT.vocab

    if args.rand_unk:
        init_emb(vocab, init="randn")

    if args.load_model:
        model = torch.load(pjoin(args.run_dir, 'model.pickle'))
    else:
        model = Model(vocab, nclasses=len(labels), emb_dim=args.emb_dim,
                      hidden_size=args.d, depth=args.depth)

    if torch.cuda.is_available():
        model.cuda(args.gpu)

    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))

    if not args.load_model:
        need_grad = lambda x: x.requires_grad
        optimizer = optim.Adam(
            filter(need_grad, model.parameters()),
            lr=0.001, weight_decay=args.l2_str)

        train_module(model, optimizer, train_iter, val_iter,
                     max_epoch=args.max_epoch)

    test_accu = eval_model(model, test_iter, save_pred=True, save_viz=False)
    logger.info("final test accu: {}".format(test_accu))

    adobe_accu = eval_adobe(model, adobe_test_iter, save_pred=True, save_viz=False)
    logger.info("final adobe accu: {}".format(adobe_accu))