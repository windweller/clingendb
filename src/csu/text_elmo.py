"""
This is a Python 3 file

ELMO uses real text...
"""

import logging
import itertools

import sys
import json
import os
import argparse

from collections import defaultdict

from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn import metrics

from torch.nn import Module

import nltk
from allennlp.commands.elmo import ElmoEmbedder
from typing import Generator, List, Tuple

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
argparser.add_argument("--multi_attn", action="store_true", help="create task-specific representations")
argparser.add_argument("--skim", action="store_true", help="a skimming model")
argparser.add_argument("--skim_interval", type=int, default=5, help="how many words to skim and group together")
argparser.add_argument("--local_sum", action="store_true", help="sum over an interval")

argparser.add_argument("--l2_penalty_softmax", type=float, default=0., help="add L2 penalty on softmax weight matrices")
argparser.add_argument("--l2_str", type=float, default=0, help="a scalar that reduces strength")  # 1e-3

argparser.add_argument("--prototype", action="store_true", help="use hierarchical loss")
argparser.add_argument("--softmax_hier", action="store_true", help="use hierarchical loss")
argparser.add_argument("--max_margin", action="store_true", help="use hierarchical loss")
argparser.add_argument("--penal_keys", action="store_true",
                       help="apply closeness penalty for keys instead of classifier weights")

argparser.add_argument("--proto_str", type=float, default=1e-3, help="a scalar that reduces strength")
argparser.add_argument("--proto_out_str", type=float, default=1e-3, help="a scalar that reduces strength")
argparser.add_argument("--proto_cos", action="store_true", help="use cosine distance instead of dot product")
argparser.add_argument("--proto_maxout", action="store_true", help="maximize between-group distance")
argparser.add_argument("--proto_maxin", action="store_true", help="maximize in-group distance")
argparser.add_argument("--softmax_str", type=float, default=1e-2,
                       help="a scalar controls penalty for not falling in the group")
argparser.add_argument("--softmax_hier_prod_prob", action="store_true", help="use hierarchical loss, instead of sum, "
                                                                             "we do it correctly, product instead")
argparser.add_argument("--softmax_hier_sum_logit", action="store_true", help="use hierarchical loss")
argparser.add_argument("--max_margin_neighbor", type=float, default=0.5,
                       help="maximum should be 1., ")

args = argparser.parse_args()

VERY_NEGATIVE_NUMBER = -1e30

cos_sim = nn.CosineSimilarity(dim=0)

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


x_data = []
y_data = []

tokenizer = nltk.tokenize

# TODO: need to sentence tokenize first...then send each sentence into ELMo
# TODO: then concatenate those embeddings together...? But masking would be crazy...
def get_batch_iter(file, batch_size: int):
    global x_data
    global y_data

    if len(x_data) == 0 and len(y_data) == 0:
        with open(file, 'r') as f:
            for line in f:
                x, y = line.split('\t')
                x_words = tokenizer.word_tokenize(x)
                x_data.append(x_words)
                y_data.append(y.strip().split())
        logger.info("SpaCy preprocessing has finished!")

    for ii in range(0, len(x_data), batch_size):
        yield x_data[ii:ii + batch_size], y_data[ii:ii + batch_size]


"""
Util modules
"""


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    r"""Function that measures Binary Cross Entropy between target and output
    logits.
    See :class:`~torch.nn.BCEWithLogitsLoss` for details.
    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
    Examples::
         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.FloatTensor(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class BCEWithLogitsLoss(Module):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    The loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ t_n \cdot \log \sigma(x_n)
        + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],
    where :math:`N` is the batch size. If reduce is ``True``, then
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True
     Shape:
         - Input: :math:`(N, *)` where `*` means, any number of additional
           dimensions
         - Target: :math:`(N, *)`, same shape as the input
     Examples::
        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.FloatTensor(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, reduce=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            var = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return binary_cross_entropy_with_logits(input, target,
                                                    var,
                                                    self.size_average,
                                                    reduce=self.reduce)
        else:
            return binary_cross_entropy_with_logits(input, target,
                                                    size_average=self.size_average,
                                                    reduce=self.reduce)


def move_to_cuda(th_var):
    if torch.cuda.is_available():
        return th_var.cuda()
    else:
        return th_var


def y_to_tensor(y: List[List[str]]) -> torch.FloatTensor:
    y_tensor = torch.zeros([len(y), label_size])
    for i in range(len(y)):
        for l_b in y[i]:
            y_tensor[i, int(l_b)] = 1.
    return y_tensor


class Model(nn.Module):
    def __init__(self, elmo, hidden_size=512, depth=1, nclasses=5):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.drop = nn.Dropout(0.5)  # suggested by paper

        d_out = hidden_size

        self.emb_dim = 1024
        self.embed = elmo

        self.encoder = nn.LSTM(
            self.emb_dim,
            hidden_size,
            depth,
            dropout=0.5,  # dropout for ELMo, recommended
            bidirectional=False)  # ha...not even bidirectional

        self.is_cuda = torch.cuda.is_available()

        self.scalar_parameters = nn.ParameterList([nn.Parameter(torch.FloatTensor([0.0]))
                                                   for _ in range(3)])
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))

        self.mixture_size = 3

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

    def scalar_mix(self, tensors):
        if len(tensors) != self.mixture_size:
            raise ArithmeticError("{} tensors were passed, but the module was initialized to "
                                  "mix {} tensors.".format(len(tensors), self.mixture_size))

        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter
                                                                in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size=1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)

    # this is the main function used
    def forward(self, input: List[List[str]]):
        output_vecs, lengths = self.get_vectors(input)
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

    def get_vectors(self, input):
        # so sent_reps is actually a variable.
        # we could update ELMo if we want to :)
        # but currently it's not updated by Adam
        # so let's detach

        sent_reps, sent_len = self.embed.batch_to_embeddings(input)
        # (batch_size, 3=layer_num, Time, 1024)

        sent_reps = sent_reps.detach()  # so it's not backpropagating to ELMo

        # (batch_size, Time, 1024)
        sent = self.scalar_mix([sent_reps[:,0], sent_reps[:,1], sent_reps[:,2]])

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        if args.multi_attn:
            return sent_output  # without masking

        # MUST apply negative mapping, so max pooling will not take padding elements
        # batch_mask = self.create_mask(lengths)  # (time, batch_size)
        # batch_mask = batch_mask.view(-1, len(lengths), 1)  # not sure if here broadcasting is right
        # output = self.exp_mask(output, batch_mask)  # now pads will never be chosen...

        return sent_output, sent_len

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
            for t_n in range(self.nclasses):
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


def eval_model(model, valid_path, save_pred=False, save_viz=False):
    # when test_final is true, we save predictions
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

    iter = 0

    valid_iter = get_batch_iter(valid_path, 32)

    for data in valid_iter:

        iter += 1

        x, y_list = data
        y = y_to_tensor(y_list)

        # embed with ELMO
        # this is not sorted
        sents, sent_lengths = elmo.batch_to_embeddings(x)

        output = model(sents, sent_lengths)

        if iter % 5 == 0 and save_pred:
            logger.info("at iteration {}".format(iter))

        batch_size = len(x)

        loss = criterion(output, y)
        total_loss += loss.data[0] * batch_size

        scores = output_to_prob(output).data.cpu().numpy()
        preds = output_to_preds(output)
        preds_indices = sparse_one_hot_mat_to_indices(preds)

        sparse_preds = preds_to_sparse_matrix(preds_indices.data.cpu().numpy(), batch_size, model.nclasses)

        all_scores.extend(scores.tolist())
        all_print_y_labels.extend(y.data.cpu().numpy().tolist())
        all_preds.append(sparse_preds)
        all_y_labels.append(y.data.cpu().numpy())

        # TODO: this is possibly incorrect?...not that we are using accuracy...
        correct += metrics.accuracy_score(y.data.cpu().numpy(), sparse_preds)
        cnt += 1

        orig_text = x
        all_orig_texts.extend(orig_text)

        if save_pred:
            y_indices = sparse_one_hot_mat_to_indices(y)
            condensed_preds = condense_preds(preds_indices.data.cpu().numpy().tolist(), batch_size)
            condensed_ys = condense_preds(y_indices.data.cpu().numpy().tolist(), batch_size)

            all_condensed_preds.extend(condensed_preds)
            all_condensed_ys.extend(condensed_ys)

    preds = np.vstack(all_preds)
    ys = np.vstack(all_y_labels)

    # both are only true if it's for test, this saves sysout
    logger.info("\n" + metrics.classification_report(ys, preds))

    if save_pred:
        with open(pjoin(args.run_dir, 'label_vis_map.json'), 'wb') as f:
            json.dump([all_condensed_preds, all_condensed_ys, all_scores, all_print_y_labels, all_orig_texts], f)

        with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
            json.dump(labels, f)

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

        with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
            json.dump(labels, f)

    model.train()
    return correct / cnt


def eval_adobe(model, valid_path, save_pred=False, save_viz=False):
    # when test_final is true, we save predictions
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

    logger.info("Evaluating on Adobe dataset")

    iter = 0

    valid_iter = get_batch_iter(valid_path, 32)

    for data in valid_iter:

        iter += 1

        x, y_list = data
        y = y_to_tensor(y_list)

        # embed with ELMO
        # this is not sorted
        # sents, sent_lengths = elmo.batch_to_embeddings(x)

        output = model(x)

        if iter % 5 == 0 and save_pred:
            logger.info("at iteration {}".format(iter))

        batch_size = len(x)

        loss = criterion(output, y)
        total_loss += loss.data[0] * batch_size

        scores = output_to_prob(output).data.cpu().numpy()
        preds = output_to_preds(output)
        preds_indices = sparse_one_hot_mat_to_indices(preds)

        sparse_preds = preds_to_sparse_matrix(preds_indices.data.cpu().numpy(), batch_size, model.nclasses)

        all_scores.extend(scores.tolist())
        all_print_y_labels.extend(y.data.cpu().numpy().tolist())
        all_preds.append(sparse_preds)
        all_y_labels.append(y.data.cpu().numpy())

        # TODO: this is possibly incorrect?...not that we are using accuracy...
        correct += metrics.accuracy_score(y.data.cpu().numpy(), sparse_preds)
        cnt += 1

        orig_text = x
        all_orig_texts.extend(orig_text)

        if save_pred:
            y_indices = sparse_one_hot_mat_to_indices(y)
            condensed_preds = condense_preds(preds_indices.data.cpu().numpy().tolist(), batch_size)
            condensed_ys = condense_preds(y_indices.data.cpu().numpy().tolist(), batch_size)

            all_condensed_preds.extend(condensed_preds)
            all_condensed_ys.extend(condensed_ys)

    preds = np.vstack(all_preds)
    ys = np.vstack(all_y_labels)

    # both are only true if it's for test, this saves sysout
    logger.info("\n" + metrics.classification_report(ys, preds))

    if save_pred:
        # So the format for each entry is: y = [], pred = [], for all labels
        # we also need
        with open(pjoin(args.run_dir, 'adobe_label_vis_map.json'), 'wb') as f:
            # used to be : all_condensed_preds, all_condensed_ys, all_scores, all_print_y_labels
            json.dump([all_condensed_preds, all_condensed_ys, all_scores, all_print_y_labels, all_orig_texts], f)

    if save_viz:
        with open(pjoin(args.run_dir, 'adobe_label_vis_map.json'), 'wb') as f:
            json.dump([all_condensed_preds, all_condensed_ys, all_orig_texts, all_text_vis], f)

    model.train()
    return correct / cnt


# ignore probability smaller than 1e-5 0.00001 <-> 0.01%
prob_threshold = nn.Threshold(1e-5, 0)


def train_module(model, optimizer,
                 train_path, valid_path, max_epoch):
    model.train()
    criterion = BCEWithLogitsLoss(reduce=False)

    exp_cost = None
    end_of_epoch = True  # False  # set true because we want immediate feedback...

    best_valid = 0.
    epoch = 1

    softmax_weight = model.get_softmax_weight()

    for n in range(max_epoch):
        iter = 0
        train_iter = get_batch_iter(train_path, args.batch_size)

        for data in train_iter:
            iter += 1

            model.zero_grad()
            x, y_list = data
            y = y_to_tensor(y_list)

            # embed with ELMO
            # this is not sorted
            # sents, sent_lengths = elmo.batch_to_embeddings(x)

            output = model(x)  # this is just logit (before calling sigmoid)

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
                #   # (Batch, n_classes)

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

                # just here to safeguard any potential problem!
                if args.softmax_hier_sum_prob:
                    meta_probs = torch.clamp(meta_probs, min=0., max=1.)

                # generate meta-label
                y_indices = sparse_one_hot_mat_to_indices(y)
                meta_y = generate_meta_y(y_indices.data.cpu().numpy().tolist(), meta_label_size, batch_size)
                meta_y = move_to_cuda(Variable(torch.from_numpy(meta_y)))

                loss = criterion(output, y).mean()  # original loss
                meta_loss = criterion(meta_probs, meta_y).mean()  # hierarchy loss
                loss += meta_loss * args.softmax_str
                loss.backward()

            elif args.max_margin:
                pass
                # nn.MultiMarginLoss
                # margin_criterion(output, y)
            else:
                loss = criterion(output, y).mean()
                loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = loss.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

            if iter % 100 == 0:
                logger.info("iter {} lr={} train_loss={} exp_cost={} \n".format(iter, optimizer.param_groups[0]['lr'],
                                                                                loss.data[0], exp_cost))

        valid_accu = eval_model(model, valid_path)
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

    label_size = 36 if args.dataset == 'snomed_refined_multi_label_no_des' else 42
    # 18 if args.dataset != "multi_top_snomed_no_des" else 42

    if args.dataset == 'multi_top_snomed_no_des':
        root_path = '../../data/csu/'
        train_path = pjoin(root_path, 'snomed_multi_label_no_des_train.tsv')
        val_path = pjoin(root_path, 'snomed_multi_label_no_des_valid.tsv')
        test_path = pjoin(root_path, 'snomed_multi_label_no_des_test.tsv')
    elif args.dataset == 'snomed_refined_multi_label_no_des':
        root_path = '../../data/csu/'
        train_path = pjoin(root_path, 'snomed_refined_multi_label_no_des_train.tsv')
        val_path = pjoin(root_path, 'snomed_refined_multi_label_no_des_valid.tsv')
        test_path = pjoin(root_path, 'snomed_refined_multi_label_no_des_test.tsv')
    else:
        raise Exception("unknown dataset")

    adobe_path = pjoin(root_path, 'adobe_snomed_multi_label_no_des_test.tsv')

    # do repeat=False
    # train_iter = get_batch_iter(train_path, 32)
    # val_iter = get_batch_iter(val_path, 256)
    # test_iter = get_batch_iter(test_path, 256)

    # now we'll have to do sorting ourselves
    # ELMO returns things to us in the original order...we use InferSent code to help us sort...

    elmo = ElmoEmbedder(cuda_device=args.gpu)

    model = Model(elmo, nclasses=len(labels),
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
        lr=0.001, weight_decay=args.l2_str)

    train_module(model, optimizer, train_path, val_path,
                 max_epoch=args.max_epoch)

    test_accu = eval_model(model, test_path, save_pred=True, save_viz=False)
    logger.info("final test accu: {}".format(test_accu))

    adobe_accu = eval_adobe(model, adobe_path, save_pred=True, save_viz=False)
    logger.info("final adobe accu: {}".format(adobe_accu))
