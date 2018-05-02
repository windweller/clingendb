"""
This file has all variations and modifications on the model
that other files have explored
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

reload(sys)
sys.setdefaultencoding('utf-8')

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--dataset", type=str, default='snomed_refined_multi_label_no_des',
                       help="multi_top_snomed_no_des|snomed_refined_multi_label_no_des")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--emb_dim", type=int, default=100)
argparser.add_argument("--max_epoch", type=int, default=5)
argparser.add_argument("--d", type=int, default=512)
argparser.add_argument("--dropout", type=float, default=0.3,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--depth", type=int, default=1)
argparser.add_argument("--lr", type=float, default=1.0)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./exp')
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--gpu", type=int, default=-1)
argparser.add_argument("--rand_unk", action="store_true", help="randomly initialize unk")
argparser.add_argument("--emb_update", action="store_true", help="update embedding")
argparser.add_argument("--multi_attn", action="store_true", help="create task-specific representations")

argparser.add_argument("--skim", action="store_true", help="a skimming model")
argparser.add_argument("--skim_interval", type=int, default=5, help="how many words to skim and group together")
argparser.add_argument("--local_sum", action="store_true", help="sum over an interval")
argparser.add_argument("--local_trans", action="store_true", help="apply weight matrix instead")
argparser.add_argument("--local_trans_nonlinear", action="store_true",
                       help="apply weight matrix + nonlinear instead, practically a CNN")

argparser.add_argument("--cluster", action="store_true", help="use hierarchical loss")
argparser.add_argument("--sigma_M", type=float, default=0, help="L2 norms of the weight vectors")  # 1e-3
argparser.add_argument("--sigma_B", type=float, default=0, help="the distance between cluster vectors")  # 1e-3
argparser.add_argument("--sigma_W", type=float, default=0, help="the distance within cluster vectors")  # 1e-3

argparser.add_argument("--softmax_hier", action="store_true", help="use hierarchical loss, we use prod product instead")
argparser.add_argument("--beta", type=float, default=1e-2,
                       help="a scalar controls penalty for not falling in the group")

argparser.add_argument("--penal_keys", action="store_true",
                       help="apply closeness penalty for keys instead of classifier weights")
argparser.add_argument("--proto_cos", action="store_true", help="use cosine distance instead of dot product")

argparser.add_argument("--reject", action="store_true",
                       help="learn a simple linear model to reject")
# essentially we are rejecting (or converting aggregate prob) by summing or averaging them
# not the same as our confidence calculation
argparser.add_argument("--gamma", type=float, default=0.1,
                       help="rejection cost, should be < 0.5 for size_average=True, "
                            "should be < 0.5 * m for size_average=False")

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

        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if args.emb_update else False

        self.emb_dim = emb_dim

        # add skim_interval
        if args.local_trans:
            self.trans_w = nn.Parameter(torch.randn(args.skim_interval * emb_dim, emb_dim))

        if args.local_trans_nonlinear:
            self.trans_w = nn.Parameter(torch.randn(args.skim_interval * emb_dim, emb_dim))
            self.trans_bias = nn.Parameter(torch.randn(emb_dim))

        if args.multi_attn:
            # prepare keys
            logger.info("adding attention matrix")
            self.task_queries = nn.Parameter(torch.randn(hidden_size, nclasses))
            # self.out = nn.Linear(self.hidden_size, self.nclasses)  # include bias, to prevent bias assignment

            # we could have just use one vector...here we use multiple...
            self.out_proj = nn.Parameter(torch.randn(1, self.nclasses, self.hidden_size))
            self.normalize = torch.nn.Softmax(dim=0)
        else:
            self.out = nn.Linear(d_out, nclasses)  # include bias, to prevent bias assignment

        if args.reject:
            # we add more parameters
            # takes in logits or probs as input, and output whether to drop or not
            # reject based on the whole batch
            self.reject_model = nn.Sequential(
                nn.Linear(nclasses, 1),
                nn.Sigmoid()
            )

    # this is the main function used
    def forward(self, input, lengths=None):
        output_vecs = self.get_vectors(input, lengths)
        return self.get_logits(output_vecs, lengths)

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

        if lengths is not None:
            lengths = lengths.view(-1).tolist()

        if args.skim:
            seq_len, batch_size, _ = embed_input.size()
            # then we actually skim it here!!!
            if args.local_sum:
                # just add...
                skimmed_input_vec = []
                for t in range(0, seq_len, args.skim_interval):
                    # skimmed = torch.max(output_vec[t:t+args.skim_interval, :, :], 0)[0].squeeze(0)
                    if t + args.skim_interval < seq_len:
                        skimmed = torch.sum(embed_input[t:t + args.skim_interval, :, :], dim=0)
                    else:
                        skimmed = torch.sum(embed_input[t:seq_len, :, :], dim=0)
                    skimmed_input_vec.append(skimmed)
                skimmed_input_vec = torch.stack(skimmed_input_vec, dim=0)
                embed_input = skimmed_input_vec
                if lengths is not None:
                    # list(int)
                    # this should be the reduced lengths
                    lengths = [len(range(0, l, args.skim_interval)) for l in lengths]
            elif args.local_trans:
                skimmed_input_vec = []
                for t in range(0, seq_len, args.skim_interval):
                    # range(t, t+args.skim_interval) will not go out of range
                    # even when t+args.skim_interval = seq_len
                    if t + args.skim_interval <= seq_len:
                        skimmed_vecs = [embed_input[i, :, :] for i in range(t, t + args.skim_interval)]
                        skimmed_vecs = torch.cat(skimmed_vecs, dim=1)  # (batch_size, hidden * time_interval)
                        skimmed = torch.matmul(skimmed_vecs, self.trans_w)
                        # skimmed = torch.sum(embed_input[t:t + args.skim_interval, :, :], dim=0)
                    else:
                        # we multiply with only part of the matrix
                        residuals = seq_len % args.skim_interval
                        skimmed_vecs = [embed_input[i, :, :] for i in range(t, seq_len)]
                        skimmed_vecs = torch.cat(skimmed_vecs, dim=1)
                        # only multiply matrix as needed
                        skimmed = torch.matmul(skimmed_vecs, self.trans_w[0:self.emb_dim * residuals, :])
                    skimmed_input_vec.append(skimmed)
                skimmed_input_vec = torch.stack(skimmed_input_vec, dim=0)
                embed_input = skimmed_input_vec
            elif args.local_trans_nonlinear:
                skimmed_input_vec = []
                for t in range(0, seq_len, args.skim_interval):
                    # range(t, t+args.skim_interval) will not go out of range
                    # even when t+args.skim_interval = seq_len
                    if t + args.skim_interval <= seq_len:
                        skimmed_vecs = [embed_input[i, :, :] for i in range(t, t + args.skim_interval)]
                        skimmed_vecs = torch.cat(skimmed_vecs, dim=1)  # (batch_size, hidden * time_interval)
                        skimmed = torch.tanh(torch.matmul(skimmed_vecs, self.trans_w) + self.trans_bias)
                        # skimmed = torch.sum(embed_input[t:t + args.skim_interval, :, :], dim=0)
                    else:
                        # we multiply with only part of the matrix
                        residuals = seq_len % args.skim_interval
                        skimmed_vecs = [embed_input[i, :, :] for i in range(t, seq_len)]
                        skimmed_vecs = torch.cat(skimmed_vecs, dim=1)
                        # only multiply matrix as needed
                        skimmed = torch.tanh(
                            torch.matmul(skimmed_vecs, self.trans_w[0:self.emb_dim * residuals, :]) + self.trans_bias)

                    skimmed_input_vec.append(skimmed)
                skimmed_input_vec = torch.stack(skimmed_input_vec, dim=0)
                embed_input = skimmed_input_vec
                if lengths is not None:
                    # list(int)
                    # this should be the reduced lengths
                    lengths = [len(range(0, l, args.skim_interval)) for l in lengths]

        packed_emb = embed_input
        if lengths is not None:
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output)[0]

        if args.multi_attn:
            return output  # without masking

        # MUST apply negative mapping, so max pooling will not take padding elements
        batch_mask = self.create_mask(lengths)  # (time, batch_size)
        batch_mask = batch_mask.view(-1, len(lengths), 1)  # not sure if here broadcasting is right
        output = self.exp_mask(output, batch_mask)  # now pads will never be chosen...

        return output

    def get_logits(self, output_vec, lengths):
        if not args.multi_attn:
            output = torch.max(output_vec, 0)[0].squeeze(0)
            return self.out(output)
        else:
            # output_vec: (seq_len, batch_size, hid_dim) x task_queries: (hid_dim, attn_heads)
            # (seq_len, batch_size, attn_heads)

            seq_len, batch_size, _ = output_vec.size()

            keys = torch.matmul(output_vec, self.task_queries)

            # (seq_len, batch_size)
            # batch_mask = self.create_mask(lengths)
            # exp_mask = Variable((1 - batch_mask) * VERY_NEGATIVE_NUMBER, requires_grad=False)
            # keys += move_to_cuda(exp_mask).view(seq_len, batch_size, 1) # (seq_len, batch_size, 1)

            # TODO: this is not perhaps perfect...but should work to a good degree
            # exp_mask = (1 - (output_vec.data == 0.).float()) * VERY_NEGATIVE_NUMBER
            # exp_mask = Variable(exp_mask, requires_grad=False).sum(2)  # this is on CUDA anyway
            # keys += exp_mask.view(skimmed_len, batch_size, 1)

            # (seq_len, batch_size, label_size)
            keys = self.normalize(keys)  # softmax normalization with attention

            # This way is more space-saving, albeit slower
            # output_vec: (seq_len, batch_size, hid_dim) x keys: (seq_len, batch_size, task_numbers)
            task_specific_list = []
            for t_n in xrange(self.nclasses):
                # (seq_len, batch_size, hid_dim) x (seq_len, batch_size, 1)
                # sum over 0
                # (batch_size, hid_dim)
                task_specific_list.append(
                    torch.squeeze(torch.sum(output_vec * keys[:, :, t_n].contiguous().view(seq_len, batch_size, 1), 0)))

            # now it's (batch_size, task_numbers, hid_dim)
            task_specific_mix = torch.stack(task_specific_list, dim=1)
            output = torch.sum(self.out_proj * task_specific_mix, 2)

            # (batch_size, hid_dim * task_numbers)
            # task_specific_mix = torch.cat(task_specific_list, dim=1)

            # output = self.out(task_specific_mix)

            return output

    def get_softmax_weight(self):
        if not args.multi_attn:
            return self.out.weight
        else:
            # I guess the key prototype vectors
            # and final vectors are the same...
            if args.penal_keys:
                return self.task_queries
            else:
                # so dim is (hidden_size, n_class) after operation
                return torch.squeeze(self.out_proj).t

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

    def get_time_contrib_map(self, sent_vec, indices, batch_size, seq_len):
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

    def get_tensor_credit_assignment(self, output):
        # also sum of history...
        # assign by-label credits, no conflicts
        # return [batch_size, time_step, label_size]

        seq_len, batch_size, _ = output.size()

        sent_vec, indices = torch.max(output, 0)
        contrib_map = self.get_time_contrib_map(sent_vec, indices, batch_size, seq_len)
        normed_contrib_map = self.directional_norm(contrib_map)  # global unitized, now directional!

        return normed_contrib_map

    def get_tensor_label_attr(self, output):
        seq_len, batch_size, _ = output.size()

        sent_vec, indices = torch.max(output, 0)
        contrib_map = self.get_time_contrib_map(sent_vec, indices, batch_size, seq_len)

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
        global_normed_contrib_map = torch.nn.Softmax(dim=1)(masked_contrib_map)
        local_normed_contrib_map = torch.nn.Softmax(dim=2)(masked_contrib_map)  # local masks still matter
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

        # not sure if there's Torch tensor-op that will solve this
        for b in range(batch_size):
            # this is per-example, we record used time_steps
            time_steps = set()
            for d in range(self.hidden_size):
                # get the time_step of each dim
                time_step = indices.data.cpu()[b, d]
                # add to votes distributions
                # votes_dist[b, time_step] += 1
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
                        # if (old_label_index != new_label_index).numpy():
                        #     records[b, time_step] += 1  # because it voted for a different label
                        #     if (new_max_contribution > old_max_contribution).numpy():
                        #         reassign_records[b, time_step] += 1

        return assignment_dist  # , records, reassign_records, votes_dist


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


# def generate_meta_y(indices, meta_label_size, batch_size):
#     a = np.array([[0.] * meta_label_size for _ in range(batch_size)], dtype=np.float32)
#     matched = defaultdict(set)  # not necessary right?? since we only make it 1.
#     for b, l in indices:
#         if b not in matched:
#             a[b, meta_label_mapping[str(l)]] = 1.
#             matched[b].add(meta_label_mapping[str(l)])
#         elif meta_label_mapping[str(l)] not in matched[b]:
#             a[b, meta_label_mapping[str(l)]] = 1.
#             matched[b].add(meta_label_mapping[str(l)])
#
#     assert np.sum(a <= 1) == a.size
#
#     return a

def generate_meta_y(indices, meta_label_size, batch_size):
    a = np.array([[0.] * meta_label_size for _ in range(batch_size)], dtype=np.float32)
    for b, l in indices:
        if l in label_id_to_meta_group:
            a[b, label_id_to_meta_group[l]] = 1.

    # will generate meta_y=(0, 0, ..., 0) if no match
    assert np.sum(a <= 1) == a.size
    return a


def eval_model(model, valid_iter, save_pred=False, save_viz=False):
    # when test_final is true, we save predictions
    model.eval()
    criterion = BCEWithLogitsLoss(reduce=False)
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

    # we save something else for rejected examples
    # in order to look at them closely
    reject_scores = []
    reject_preds = []
    reject_y_labels = []
    reject_print_y_labels = []
    reject_orig_texts = []
    reject_condensed_preds = []
    reject_condensed_ys = []

    iter = 0
    for data in valid_iter:
        (x, x_lengths), y = data.Text, data.Description
        iter += 1

        if iter % 5 == 0 and save_pred:
            logger.info("at iteration {}".format(iter))

        output_vecs = model.get_vectors(x, x_lengths)
        output = model.get_logits(output_vecs, x_lengths)  # this is logits, not sent to sigmoid yet!

        if save_viz:
            # (batch_size, time_step, label_dist)
            label_assignment_tensor = model.get_visualization_tensor_max_assignment(output_vecs)
            all_text_vis.extend(label_assignment_tensor.numpy().tolist())

            # if save_pred:
            # credit_assign = model.get_tensor_credit_assignment(output_vecs)
            # global_map, local_map = model.get_tensor_label_attr(output_vecs)

            # all_credit_assign.extend(credit_assign.numpy().tolist())
            # all_la_global.extend(global_map.numpy().tolist())
            # all_la_local.extend(local_map.numpy().tolist())

        batch_size = x.size(1)

        loss = criterion(output, y)

        if args.reject:
            reject_probs = model.reject_model(output)  # [batch_size]
            loss = (1 - reject_probs) * loss + reject_probs * args.gamma

        loss = loss.mean()

        total_loss += loss.data[0] * x.size(1)

        if args.reject:
            # when we reject, in validation; we actually take out
            # output: [batch_size, label_size]
            non_rejects = reject_probs < 0.5
            rejects = reject_probs >= 0.5
            rejected_output = output[rejects.detach(), :]  # ones we reject
            rejected_batch_size = rejects.sum().cpu().data[0]

            if save_pred:
                rejected_x = x[:, rejects.detach(), :]
                rejected_y = y[rejects.detach(), :]

                reject_scores = output_to_prob(rejected_output).data.cpu().numpy()
                reject_preds = output_to_preds(rejected_output)
                reject_preds_indices = sparse_one_hot_mat_to_indices(reject_preds)

                reject_sparse_preds = preds_to_sparse_matrix(reject_preds_indices.data.cpu().numpy(),
                                                             rejected_batch_size, model.nclasses)

                reject_scores.extend(reject_scores.tolist())
                reject_print_y_labels.extend(rejected_y.data.cpu().numpy().tolist())
                reject_preds.append(reject_sparse_preds)
                reject_y_labels.append(rejected_y.data.cpu().numpy())

                reject_orig_text = TEXT.reverse(rejected_x.data)
                reject_orig_texts.extend(reject_orig_text)

                reject_y_indices = sparse_one_hot_mat_to_indices(rejected_y)
                reject_condensed_pred = condense_preds(reject_preds_indices.data.cpu().numpy().tolist(), rejected_batch_size)
                reject_condensed_y = condense_preds(reject_y_indices.data.cpu().numpy().tolist(), rejected_batch_size)

                reject_condensed_preds.extend(reject_condensed_pred)
                reject_condensed_ys.extend(reject_condensed_y)

            # clear out for the rest
            output = output[non_rejects.detach(), :]  # selecting ones that are not rejecting
            x = x[:, non_rejects.detach(), :]  # (time_seq, batch_size, dim)
            y = y[non_rejects.detach(), :]  # (batch_size, label_size)
            batch_size = batch_size - rejected_batch_size  # int, new non-rejected batch size

        scores = output_to_prob(output).data.cpu().numpy()
        preds = output_to_preds(output)
        preds_indices = sparse_one_hot_mat_to_indices(preds)

        sparse_preds = preds_to_sparse_matrix(preds_indices.data.cpu().numpy(), batch_size, model.nclasses)

        if save_pred:
            all_scores.extend(scores.tolist())
            all_print_y_labels.extend(y.data.cpu().numpy().tolist())
            all_preds.append(sparse_preds)
            all_y_labels.append(y.data.cpu().numpy())

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

    logger.info(multiclass_f1_msg)

    # both are only true if it's for test, this saves sysout
    logger.info("\n" + metrics.classification_report(ys, preds))

    if save_pred:
        # So the format for each entry is: y = [], pred = [], for all labels
        # we also need
        import csv
        with open(pjoin(args.run_dir, 'confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in zip(all_condensed_preds, all_condensed_ys, all_orig_texts):
                writer.writerow({'preds': pair[0], 'labels': pair[1], 'text': pair[2]})

        if args.reject:
            with open(pjoin(args.run_dir, 'label_vis_map_reject.json'), 'wb') as f:
                json.dump([reject_condensed_preds, reject_condensed_ys, reject_scores, reject_print_y_labels, reject_orig_texts], f)

        with open(pjoin(args.run_dir, 'label_vis_map.json'), 'wb') as f:
            json.dump([all_condensed_preds, all_condensed_ys, all_scores, all_print_y_labels, all_orig_texts], f)

    if save_viz:
        import csv
        with open(pjoin(args.run_dir, 'confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in zip(all_condensed_preds, all_condensed_ys, all_orig_texts):
                writer.writerow({'preds': pair[0], 'labels': pair[1], 'text': pair[2]})

        with open(pjoin(args.run_dir, 'label_vis_map.json'), 'wb') as f:
            json.dump([all_condensed_preds, all_condensed_ys, all_orig_texts, all_text_vis], f)

    return correct / cnt


# ignore probability smaller than 1e-5 0.00001 <-> 0.01%
prob_threshold = nn.Threshold(1e-5, 0)

# average of each example
criterion = BCEWithLogitsLoss(reduce=False)
meta_criterion = nn.BCELoss()


def train_module(model, optimizer,
                 train_iter, valid_iter, max_epoch):
    model.train()

    exp_cost = None
    end_of_epoch = True  # False  # set true because we want immediate feedback...
    training_rejectiong_rate = []

    best_valid = 0.
    epoch = 1
    no_meta_label = 0.

    # (d_out, 1)
    softmax_weight = model.get_softmax_weight()

    for n in range(max_epoch):
        iter = 0
        for data in train_iter:
            iter += 1

            model.zero_grad()
            (x, x_lengths), y = data.Text, data.Description

            output = model(x, x_lengths)  # this is just logit (before calling sigmoid)

            batch_size = x.size(1)

            if args.cluster:
                loss = criterion(output, y)

                # three penalty terms:
                # Omega_mean, Omega_between, Omega_within
                # Omega_mean

                w_bar = softmax_weight.sum(1) / label_size  # w_bar

                omega_mean = softmax_weight.pow(2).sum()
                omega_between = 0.
                omega_within = 0.

                for c in xrange(len(meta_category_groups)):
                    m_c = len(meta_category_groups[c])
                    w_c_bar = softmax_weight[:, meta_category_groups[c]].sum(1) / m_c
                    omega_between += m_c * (w_c_bar - w_bar).pow(2).sum()
                    for i in meta_category_groups[c]:
                        omega_within += (softmax_weight[:, i] - w_c_bar).pow(2).sum()

                # singleton groups need to be penalized as well in this setting
                for s in xrange(len(singleton_groups)):
                    w_c_bar = singleton_groups[s]
                    omega_between += (w_c_bar - w_bar).pow(2).sum()

            elif args.softmax_hier:
                # compute loss for the meta-category level

                # (Batch, n_classes)
                snomed_values = output_to_prob(output)

                # generate meta-label
                y_indices = sparse_one_hot_mat_to_indices(y)
                meta_y = generate_meta_y(y_indices.data.cpu().numpy().tolist(), meta_label_size, batch_size)

                # no meta-label present (very rare, since it's a batch)
                if meta_y.sum() == 0.:
                    no_meta_label += 1
                    loss = criterion(output, y)
                else:
                    # meta-lbel present
                    meta_y = move_to_cuda(Variable(torch.from_numpy(meta_y)))

                    meta_probs = []  # list of list of probability
                    # sum these prob into meta group
                    for i in range(meta_label_size):
                        # 1 - (1 - p_1)(...)(1 - p_n)
                        meta_prob = (1 - snomed_values[:, meta_category_groups[i]]).prod(1)
                        # threshold at 1e-5
                        meta_probs.append(prob_threshold(meta_prob))  # we don't want really small probability

                    meta_probs = torch.stack(meta_probs, dim=1)

                    assert meta_probs.size(1) == meta_label_size

                    loss = criterion(output, y)  # original loss
                    meta_loss = criterion(meta_probs, meta_y)  # hierarchy loss
                    loss += meta_loss * args.beta
            else:
                loss = criterion(output, y)

            if args.reject:
                s = torch.squeeze(model.reject_model(output))  # (batch_size)
                # per example loss
                loss = (1 - s) * loss + s * args.gamma
                # collect average rejection size
                training_rejectiong_rate.append(s.mean().data[0])

            if args.cluster:
                # because here we do loss.mean()
                # omega_mean is already applied to every example: n * ||w||^2
                # omega_between, omega_within are only computed once
                # should be scaled down with batch_size; otherwise can be too dominating
                # the rest
                loss = loss.mean() + omega_mean * args.sigma_M + (omega_between * args.sigma_B \
                                                                  + omega_within * args.sigma_W) / batch_size
            else:
                loss = loss.mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = loss.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

            if iter % 100 == 0:
                if len(training_rejectiong_rate) > 0:
                    avg_rej_rate = sum(training_rejectiong_rate) / float(len(training_rejectiong_rate))
                else:
                    avg_rej_rate = 0.
                logger.info("iter {} lr={} train_loss={} exp_cost={} rej={} no_meta_label={} \n".format(iter,
                                                                                                        optimizer.param_groups[
                                                                                                            0]['lr'],
                                                                                                        loss.data[0],
                                                                                                        exp_cost,
                                                                                                        avg_rej_rate,
                                                                                                        no_meta_label))

        valid_accu = eval_model(model, valid_iter)
        logger.info("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
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

    meta_category_groups = [
        [0, 17],
        [1, 2, 9, 10],
        [3, 20],
        [4, 5, 7, 12, 16, 18, 23, 33],
        [6, 24],
        [11, 31],
        [29, 30],
        [13, 14, 17, 19],
        [15, 26],
    ]
    meta_label_size = len(meta_category_groups)

    task_weight_indices = [item for items in meta_category_groups for item in items]

    # useful for cluster penalty
    singleton_groups = [32, 34, 35, 8, 21, 22, 25, 27, 28]

    # label id maps to meta group
    label_id_to_meta_group = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 3, 9: 1, 10: 1, 11: 5, 12: 3, 13: 7, 14: 7,
                              15: 8, 16: 3, 17: 7,
                              18: 3, 19: 7, 20: 2, 23: 3, 24: 4, 26: 8, 29: 6, 30: 6, 31: 5, 33: 3}

    with open('../../data/csu/snomed_refined_labels_to_name.json', 'r') as f:
        labels = json.load(f)

    logger.info("available labels are: ")
    logger.info(labels)

    TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False)

    label_size = 36  # used to be 42 and 18

    LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=label_size, tensor_type=torch.FloatTensor)

    if args.dataset == 'multi_top_snomed_no_des':
        train, val, test = data.TabularDataset.splits(
            path='../../data/csu/', train='snomed_multi_label_no_des_train.tsv',
            validation='snomed_multi_label_no_des_valid.tsv',
            test='snomed_multi_label_no_des_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])
    elif args.dataset == 'snomed_refined_multi_label_no_des':
        train, val, test = data.TabularDataset.splits(
            path='../../data/csu/', train='snomed_refined_multi_label_no_des_train.tsv',
            validation='snomed_refined_multi_label_no_des_valid.tsv',
            test='snomed_refined_multi_label_no_des_test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])

    # actually, this is the first point of improvement: load in clinical embedding instead!!!
    TEXT.build_vocab(train, vectors="glove.6B.{}d".format(args.emb_dim))

    # do repeat=False
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
        batch_sizes=(32, 256, 256), device=args.gpu,
        sort_within_batch=True, repeat=False)  # stop infinite runs
    # if not labeling sort=False, then you are sorting through valid and test

    vocab = TEXT.vocab

    if args.rand_unk:
        init_emb(vocab, init="randn")

    model = Model(vocab, nclasses=len(labels), emb_dim=args.emb_dim,
                  hidden_size=args.d, depth=args.depth)

    # print model information
    logger.info(model)

    if torch.cuda.is_available():
        model.cuda(args.gpu)

    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr=0.001)  # weight_decay=0.

    train_module(model, optimizer, train_iter, val_iter,
                 max_epoch=args.max_epoch)

    test_accu = eval_model(model, test_iter, save_pred=True, save_viz=False)
    logger.info("final test accu: {}".format(test_accu))
