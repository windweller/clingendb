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
# import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data
from util import BCEWithLogitsLoss

import logging

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--dataset", type=str, default='merged', help="merged|sub_sum, merged is the better one")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--max_epoch", type=int, default=5)
argparser.add_argument("--max_rl_epoch", type=int, default=5)
argparser.add_argument("--hidden_size", type=int, default=910)
argparser.add_argument("--num_classes", type=int, default=5)
argparser.add_argument("--dropout", type=float, default=0.2,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--depth", type=int, default=1)
argparser.add_argument("--lr", type=float, default=1.0)
argparser.add_argument("--sparsity", type=float, default=4e-4, help="{2e-4, 3e-4, 4e-4}")
argparser.add_argument("--coherent", type=float, default=2.0, help="paper did 2 * lambda1")
# argparser.add_argument("--lr_decay", type=float, default=0.98)
# argparser.add_argument("--lr_decay_epoch", type=int, default=175)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./exp')
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--gpu", type=int, default=-1)
argparser.add_argument("--emb_update", action="store_true", help="update embedding")
argparser.add_argument("--pretrain", action="store_true", help="pretrain encoder, and update embedding")
argparser.add_argument("--max_pool", action="store_true", help="use max-pooling")
argparser.add_argument("--rand_unk", action="store_true", help="randomly initialize unk")
argparser.add_argument("--update_gen_only", action="store_true",
                       help="During 2nd phase, only update generator, not encoder")
argparser.add_argument("--bidir", action="store_true", help="whether to use bidrectional LSTM or not")

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


def move_to_cpu(t_var):
    if torch.cuda.is_available():
        return t_var.cpu()
    else:
        return t_var


def to_numpy(t_var):
    cpu_t_var = move_to_cpu(t_var)
    return cpu_t_var.numpy()


def to_list(t_var):
    return to_numpy(move_to_cpu(t_var)).tolist()


class Encoder(nn.Module):
    def __init__(self, vocab, embed, emb_dim=100, hidden_size=256, depth=1, nclasses=5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.drop = nn.Dropout(args.dropout)
        self.nclasses = nclasses
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_size,
            depth,
            dropout=args.dropout,
            bidirectional=True)  # for rationale extraction, we don't want order to matter too much

        d_out = hidden_size * 2 if args.bidir else hidden_size
        self.out = nn.Linear(d_out, nclasses)
        self.embed = embed
        self.vocab = vocab

        self.cross_ent_loss_vec = nn.CrossEntropyLoss(reduce=False)

    def extract_input(self, inputs, lengths, z_mask, train=True):
        # only select input: [time_seq, batch_size, emb_dim]
        # according to mask z
        # return extracted_input, modified_length
        masks = Variable(move_to_cuda(self.create_mask(lengths)))

        # z_mask is a float Tensor on cuda, masks is also float on Cuda
        z_mask = z_mask * masks.view_as(z_mask)

        batch_size = z_mask.size(1)
        z_byte_mask = z_mask.type(torch.cuda.ByteTensor)
        lengths_list = torch.squeeze(torch.sum(z_mask, dim=0)).data.cpu().numpy().astype('int32')

        max_len = np.max(lengths_list)
        if train:
            new_embed = Variable(move_to_cuda(torch.zeros(int(max_len), batch_size, self.emb_dim)))
        else:
            # this is during evaluation
            new_embed = Variable(move_to_cuda(torch.zeros(int(max_len), batch_size)))

        # batch together again
        for i in xrange(batch_size):
            new_x = torch.masked_select(inputs[:, i], z_byte_mask[:, i]).view(-1, self.emb_dim)
            if len(new_x.size()) > 0:  # empty tensor we don't do anything, everything would be 0.
                new_embed[:lengths_list[i], i] = new_x
            else:
                # if it's empty, need to set the length to be 1
                lengths_list[i] = 1

        return new_embed, lengths_list

    def forward(self, inputs, lengths=None, z_mask=None):
        # input is a padded input with just indices
        # z_mask should not be backpropagatble. Pass in numpy array!

        embed_input = self.embed(inputs)

        if z_mask is not None:
            assert lengths is not None
            embed_input, ex_lengths = self.extract_input(embed_input, lengths, z_mask)

        packed_emb = embed_input

        if lengths is not None:
            # sort when z_mask
            if z_mask is not None:
                ex_lengths, idx_sort = np.sort(ex_lengths)[::-1], np.argsort(-ex_lengths)
                idx_unsort = np.argsort(idx_sort)

                idx_sort = move_to_cuda(torch.from_numpy(idx_sort))
                embed_input = embed_input.index_select(1, Variable(idx_sort))
            else:
                ex_lengths = lengths.view(-1).tolist()

            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, ex_lengths)

        output, hidden = self.lstm(packed_emb)  # embed_input

        if lengths is not None:
            output = nn.utils.rnn.pad_packed_sequence(output)[0]  # return to Tensor

        if z_mask is not None:
            # Un-sort by length
            # important!! Otherwise it won't match with y
            idx_unsort = move_to_cuda(torch.from_numpy(idx_unsort))
            output = output.index_select(1, Variable(idx_unsort))

        if args.max_pool:
            # TODO: can apply mask here...
            hidden = torch.max(output, 0)[0].squeeze(0)

        return output, hidden

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

    def encode(self, inputs, lengths, z_mask=None):
        # this is for pretraining, used by Model
        output, hidden = self.forward(inputs, lengths, z_mask)
        preds = self.out(hidden)

        return preds

    def get_encoder_loss(self, preds, y_labels):
        loss_vec = self.cross_ent_loss_vec(preds, y_labels)
        loss = torch.mean(loss_vec)
        return loss, loss_vec

    def get_loss(self, inputs, lengths, z_masks, y_labels):
        # run through the encoder itself, get loss, use as reward for generator training
        output, hidden = self.forward(inputs, lengths, z_masks)

        # compute the loss...return it
        preds = self.out(output)

        loss_vec = self.cross_ent_loss_vec(preds, y_labels)

        loss = torch.mean(loss_vec)

        # loss.backward() to assign gradient
        # loss.data is the actual loss
        return preds, loss, loss_vec


class SampleGenerator(nn.Module):
    def __init__(self, embed, emb_dim=100, hidden_size=256, depth=1):
        # SampleGenerator uses policy gradient to train
        # the generator
        super(SampleGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_size,
            depth,
            dropout=args.dropout,
            bidirectional=True)
        self.embed = embed

        d_out = hidden_size * 2 if args.bidir else hidden_size

        self.bce_loss = BCEWithLogitsLoss(reduce=False)  # this is per action, not averaged

        # we still use multiclass softmax loss
        self.output_layer = nn.Linear(in_features=d_out, out_features=1)
        self.to_prob = nn.Sigmoid()

    def sample(self, z_mask_prob_dist):
        # so z_mask is (seq_len, batch_size, 1)
        z_mask = torch.bernoulli(z_mask_prob_dist).detach()
        return z_mask

    def forward(self, inputs, lengths=None):
        # takes in x, output z_prob, z_mask
        # z_mask needs to be casted as ByteTensor
        # 1. run RNN through inputs
        # 2. transform outputs [time_seq, batch_size, dim] -> [time_seq, batch_size, 1], using sigmoid activation

        embed_input = self.embed(inputs)

        packed_emb = embed_input

        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.lstm(packed_emb)  # embed_input

        if lengths is not None:
            output = nn.utils.rnn.pad_packed_sequence(output)[0]  # return to Tensor

        z_mask_logits = self.output_layer(output)
        z_mask_prob_dist = self.to_prob(z_mask_logits)

        z_mask = self.sample(z_mask_prob_dist)

        return z_mask, z_mask_logits, z_mask_prob_dist

    def compute_sparsity_penalty(self, z_mask, sparsity=4e-4, coherent=2):
        # this is not considering differentiability
        # z_mask should be sampled results

        # based on the paper, search is {2e-4, 3e-4, 4e-4}, and lambda2 = 2 * lambda1
        coherent_factor = sparsity * coherent

        # float.Tensor
        z_sum = torch.sum(z_mask, dim=0)  # temporal dimension
        z_diff = torch.abs(z_mask[1:] - z_mask[:-1]).sum(dim=0)  # TODO: not checked...not understood

        sparsity_coherence_cost = torch.mean(z_sum) * sparsity + torch.mean(z_diff) * coherent_factor
        sparsity_coherence_cost_vec = z_sum * sparsity + z_diff * coherent_factor

        # vec version is used for optimization/policy gradient
        # we should print sparsity_coherence_cost out to actively monitor training situation
        return sparsity_coherence_cost, sparsity_coherence_cost_vec

    def get_loss(self, encoder_loss_vec, z_mask_logits, z_mask):
        # encoder_loss should be just a number, used as reward: loss.data
        # return a policy-gradient style loss here

        # need to be detached from previous graph
        encoder_loss_vec = encoder_loss_vec.detach()

        # z_mask_probs is still a PyTorch variable
        sparsity_cost, vec_sparsity_cost = self.compute_sparsity_penalty(z_mask, args.sparsity, args.coherent)

        logpz = self.bce_loss(z_mask_logits, z_mask)

        cost_vec = encoder_loss_vec + vec_sparsity_cost

        generator_cost = torch.mean(cost_vec)  # this is the reward

        # this is the differential objective, call .backward() on this
        cost_logpz = torch.mean(cost_vec * torch.sum(logpz, dim=0))

        return cost_logpz, generator_cost, sparsity_cost


class Model(nn.Module):
    def __init__(self, vocab, emb_dim=100, hidden_size=256, depth=1, nclasses=5):
        # this model generates rationale / interpretable parts when trained
        # this is the entire model, not just encoder
        super(Model, self).__init__()

        # build embedding layer (shared)
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if args.emb_update else False

        # build the encoder
        self.encoder = Encoder(vocab, self.embed, emb_dim, hidden_size, depth, nclasses)
        self.generator = SampleGenerator(self.embed, emb_dim, hidden_size, depth)

    def forward(self, inputs, lengths=None):
        # this should run through one trial, return loss/cost, all materials to training loop
        # if generator needs to sample multiple times, we write it here

        # run the entire model, train with loss/cost generated by model
        # all loss/costs are written into Encoder and Generator...so should be easy!

        # use generator to produce mask
        z_mask, z_mask_logits, z_mask_prob_dist = self.generator.forward(inputs, lengths)

        sparsity_coherence_cost, _ = self.generator.compute_sparsity_penalty(z_mask, args.sparsity, args.coherent)

        # encoder (pretrained or not) consumes the mask
        output = self.encoder.encode(inputs, lengths, z_mask)

        return output, z_mask, sparsity_coherence_cost

    def get_masked_input(self, inputs, lengths, z_mask):
        # in evaluation, after calling forward(), use this to get the actual extracted inputs
        # and save them
        trimmed_input, _ = self.encoder.extract_input(inputs, lengths, z_mask)
        return trimmed_input

    def get_loss(self, inputs, lengths=None, y_labels=None):
        # call during training
        z_mask, z_mask_logits, z_mask_prob_dist = self.generator.forward(inputs, lengths)

        # encoder (pretrained or not) consumes the mask
        preds = self.encoder.encode(inputs, lengths, z_mask)
        enc_loss, enc_loss_vec = self.encoder.get_encoder_loss(preds, y_labels)
        gen_cost_logpz, generator_cost, sparsity_cost = self.generator.get_loss(enc_loss_vec, z_mask_logits, z_mask)

        # gen_cost_logpz: policy gradient objective
        # generator_cost: encoder loss + sparsity cost
        # sparsity cost: coherent + sparsity

        return enc_loss, gen_cost_logpz, generator_cost, sparsity_cost


def eval_model(model, valid_iter, save_pred=False):
    # since we can't really "evaluate" the extract inputs,
    # we can only do encoder eval, basically
    model.eval()
    model.encoder.eval()
    model.generator.eval()

    correct = 0.0
    cnt = 0
    total_loss = 0.0
    total_labels_recall = None

    total_labels_prec = None
    all_preds = []
    all_y_labels = []
    all_orig_texts = []
    all_extracted_texts = []

    for data in valid_iter:
        (x, x_lengths), y = data.Text, data.Description

        # run the whole model forward mode
        output, z_mask, sparsity_coherence_cost = model.forward(x, x_lengths)
        loss, _ = model.encoder.get_encoder_loss(output, y)

        total_loss += loss.data[0] * x.size(1)  # because cross-ent by default is average
        preds = output.data.max(1)[1]  # already taking max...I think, max returns a tuple
        correct += preds.eq(y.data).cpu().sum()
        cnt += y.numel()

        # TODO: all these are a bit wrong...

        extracted_input, _ = model.encoder.extract_input(x, x_lengths, z_mask, train=False)

        extracted_text = TEXT.reverse(extracted_input.data)
        all_extracted_texts.extend(extracted_text)

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

    lengths_ratio = 0.
    for i in range(len(all_orig_texts)):
        lengths_ratio += len(all_extracted_texts[i]) / float(len(all_orig_texts[i]))
    lengths_ratio /= len(all_orig_texts)

    logger.info("average length ratio between extracted and original is: {}".format(lengths_ratio))

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
        # we store things out, hopefully they are in correct order
        with open(pjoin(args.run_dir, 'rationale_confusion_results_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels', 'text', 'extracted']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in zip(all_preds, all_y_labels, all_orig_texts, all_extracted_texts):
                writer.writerow({'preds': pair[0], 'labels': pair[1], 'text': pair[2], 'extracted': pair[3]})

        with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
            json.dump(label_list, f)

    return correct / cnt


def train_model(model, optimizer, train_iter, valid_iter, max_epoch):
    # run the entire model, train with loss/cost generated by model
    # all loss/costs are written into Encoder and Generator...so should be easy!

    model.train()
    model.encoder.train()
    model.generator.train()

    exp_cost = None
    best_valid = 0.
    epoch = 1

    for n in range(max_epoch):
        iter = 0
        for data in train_iter:
            iter += 1

            # don't know if this is necessary
            model.zero_grad()
            model.encoder.zero_grad()
            model.generator.zero_grad()

            (x, x_lengths), y = data.Text, data.Description
            enc_loss, gen_cost_logpz, generator_cost, sparsity_cost = model.get_loss(x, x_lengths, y)

            if not args.update_gen_only:
                enc_loss.backward() # retain_variables=True
            gen_cost_logpz.backward()

            torch.nn.utils.clip_grad_norm(model.generator.parameters(), args.clip_grad)
            torch.nn.utils.clip_grad_norm(model.encoder.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = gen_cost_logpz.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * gen_cost_logpz.data[0]

            if iter % 100 == 0:
                logging.info("iter {} lr={} gen_loss={} exp_cost={} enc_loss={} sparsity_cost={} \n".format(iter,
                                                                                                            optimizer.param_groups[
                                                                                                                0][
                                                                                                                'lr'],
                                                                                                            gen_cost_logpz.data[
                                                                                                                0],
                                                                                                            exp_cost,
                                                                                                            enc_loss.data[
                                                                                                                0],
                                                                                                            sparsity_cost.data[
                                                                                                                0]))

        valid_accu = eval_model(model, valid_iter)
        sys.stdout.write("epoch {} lr={:.6f} gen_train_loss={:.6f} valid_acc={:.6f}\n".format(
            epoch,
            optimizer.param_groups[0]['lr'],
            gen_cost_logpz.data[0],
            valid_accu
        ))

        epoch += 1
        if valid_accu > best_valid:
            best_valid = valid_accu

        sys.stdout.write("\n")


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


def eval_encoder(encoder, valid_iter, save_pred=False):
    encoder.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0
    total_labels_recall = None

    total_labels_prec = None
    all_preds = []
    all_y_labels = []
    all_orig_texts = []
    for data in valid_iter:
        (x, x_lengths), y = data.Text, data.Description

        output = encoder.encode(x, x_lengths)
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
        # we store things out, hopefully they are in correct order
        with open(pjoin(args.run_dir, 'pretrain_confusion_test.csv'), 'wb') as csvfile:
            fieldnames = ['preds', 'labels', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in zip(all_preds, all_y_labels, all_orig_texts):
                writer.writerow({'preds': pair[0], 'labels': pair[1], 'text': pair[2]})
        with open(pjoin(args.run_dir, 'pretrain_label_map.txt'), 'wb') as f:
            json.dump(label_list, f)

    return correct / cnt


def pretrain_encoder(encoder, optimizer,
                     train_iter, valid_iter, max_epoch):
    encoder.train()
    criterion = nn.CrossEntropyLoss()

    exp_cost = None
    end_of_epoch = True  # False  # set true because we want immediate feedback...
    best_valid = 0.
    epoch = 1

    for n in range(max_epoch):
        iter = 0
        for data in train_iter:
            iter += 1

            encoder.zero_grad()  # clear out the gradients
            (x, x_lengths), y = data.Text, data.Description

            output = encoder.encode(x, x_lengths)

            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = loss.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

            if iter % 100 == 0:
                logging.info("iter {} lr={} train_loss={} exp_cost={} \n".format(iter, optimizer.param_groups[0]['lr'],
                                                                                 loss.data[0], exp_cost))

        valid_accu = eval_encoder(encoder, valid_iter)
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


def init_emb(vocab, init="randn", num_special_toks=2, mode="unk"):
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
        batch_sizes=(args.batch_size, 256, 256), device=args.gpu,
        sort_within_batch=True, repeat=False)  # stop infinite runs

    vocab = TEXT.vocab

    if args.rand_unk:
        init_emb(vocab, init="randn")

    model = Model(vocab, nclasses=len(labels))
    encoder = model.encoder
    generator = model.generator

    if torch.cuda.is_available():
        model.cuda(args.gpu)
        encoder.cuda(args.gpu)
        generator.cuda(args.gpu)

    logger.info("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr=0.001)

    if args.pretrain:
        pretrain_encoder(encoder, optimizer, train_iter, val_iter,
                         max_epoch=args.max_epoch)
        test_accu = eval_encoder(encoder, test_iter, save_pred=True)
        logger.info("final test accu: {}".format(test_accu))

    logger.info("policy gradient training starts")

    # now we start training policy gradient
    train_model(model, optimizer, train_iter, val_iter, args.max_rl_epoch)
    test_accu = eval_model(model, test_iter, save_pred=True)
    logger.info("final encoder extracted test accu: {}".format(test_accu))
