import tensorflow as tf

import os
import argparse
import sys
from os.path import join as pjoin
from util import MultiLabelField, ReversibleField, BCEWithLogitsLoss, MultiMarginHierarchyLoss
from sklearn import metrics
import numpy as np

import logging
import itertools
import json
import time

from collections import defaultdict
from torchtext import data
import torch

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs

from tensorflow.contrib.rnn.python.ops.lstm_ops import LSTMBlockFusedCell

reload(sys)
sys.setdefaultencoding('utf-8')

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--dataset", type=str, default='multi_top_snomed_adjusted_no_des',
                       help="multi_top_snomed_no_des|multi_top_snomed_adjusted_no_des, merged is the better one")
argparser.add_argument("--batch_size", "--batch", type=int, default=32)
argparser.add_argument("--emb_dim", type=int, default=100)
argparser.add_argument("--max_epoch", type=int, default=5)
argparser.add_argument("--print_every", type=int, default=100)
argparser.add_argument("--d", type=int, default=512)
argparser.add_argument("--dropout", type=float, default=0.3,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--state_dropout", type=float, default=0.2,
                       help="dropout of word embeddings and softmax output")
argparser.add_argument("--depth", type=int, default=1)
argparser.add_argument("--lr", type=float, default=0.001)
argparser.add_argument("--lr_decay", type=float, default=0.5)
argparser.add_argument("--clip_grad", type=float, default=5)
argparser.add_argument("--run_dir", type=str, default='./exp')
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--rand_unk", action="store_true", help="randomly initialize unk")
argparser.add_argument("--emb_update", action="store_true", help="update embedding")
argparser.add_argument("--multi_attn", action="store_true", help="create task-specific representations")
argparser.add_argument("--shared_decoder", action="store_true", help="shared decoder/hidden state for classification")
argparser.add_argument("--l2_penalty_softmax", type=float, default=0., help="add L2 penalty on softmax weight matrices")
argparser.add_argument("--l2_str", type=float, default=0, help="a scalar that reduces strength")  # 1e-3

argparser.add_argument("--prototype", action="store_true", help="use hierarchical loss")
argparser.add_argument("--softmax_hier", action="store_true", help="use hierarchical loss")
argparser.add_argument("--max_margin", action="store_true", help="use hierarchical loss")

argparser.add_argument("--fast", action="store_true", help="use fast operations, but give up on state dropout")

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

"""
Seeding
"""
np.random.seed(args.seed)

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


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


def preds_to_sparse_matrix(indices, batch_size, label_size):
    # this is for preds
    # indices will be a tuple: (array([1, 1, 1, 2, 2, 3, 3, 3, 4]), array([0, 1, 4, 2, 3, 0, 3, 4, 4]))
    # this is numpy.nonzero result
    labels = np.zeros((batch_size, label_size))
    b_idx, l_idx = indices
    for b, l in zip(b_idx, l_idx):
        labels[b, l] = 1.
    return labels


def sparse_one_hot_mat_to_indices(preds):
    return preds.nonzero()


def output_to_preds(probs):
    return (probs > 0.5)


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


class Encoder(object):
    def __init__(self, size, num_layers):
        self.size = size
        self.num_layers = num_layers
        self.keep_prob = tf.placeholder(tf.float32)

        if not args.fast:
            self.state_keep_prob = tf.placeholder(tf.float32)

            cell = rnn_cell.BasicLSTMCell(self.size)
            state_is_tuple = True

            cell = DropoutWrapper(cell, input_keep_prob=self.keep_prob, state_keep_prob=self.state_keep_prob,
                                  seed=args.seed)
            self.encoder_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=state_is_tuple)

            cell_back = rnn_cell.BasicLSTMCell(self.size)
            state_is_tuple = True

            cell_back = DropoutWrapper(cell_back, input_keep_prob=self.keep_prob, state_keep_prob=self.state_keep_prob,
                                       seed=args.seed)
            self.encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_back] * num_layers,
                                                               state_is_tuple=state_is_tuple)
        else:
            logging.info("Using LSTMBlockFusedCell")
            cell = LSTMBlockFusedCell(self.size)
            state_is_tuple = True

            # cell = DropoutWrapper(cell, input_keep_prob=self.keep_prob, state_keep_prob=self.state_keep_prob,
            #                       seed=args.seed)
            # self.encoder_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=state_is_tuple)
            self.encoder_cell = cell

            cell_back = LSTMBlockFusedCell(self.size)
            # state_is_tuple = True
            #
            # cell_back = DropoutWrapper(cell_back, input_keep_prob=self.keep_prob, state_keep_prob=self.state_keep_prob,
            #                            seed=args.seed)
            # self.encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_back] * num_layers,
            #                                                    state_is_tuple=state_is_tuple)
            self.encoder_cell_bw = cell_back

    def encode(self, inputs, srclen, reuse=False, scope_name="", temp_max=False):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: (time_step, length, size), notice that input is "time-major"
                        instead of "batch-major".
        :param srclen: An int32/int64 vector, size `[batch_size]`
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with vs.variable_scope(scope_name + "Encoder", reuse=reuse):
            inp = inputs

            if args.fast:
                # apply at least input dropout since DropoutWrapper is not usable
                inp = tf.nn.dropout(inp, self.keep_prob)

            with vs.variable_scope("EncoderCell") as scope:
                (fw_out, bw_out), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    self.encoder_cell,
                    self.encoder_cell_bw, inp, srclen,
                    scope=scope, dtype=tf.float32, time_major=True)
                # (batch_size, T, hidden_size * 2)

                # (T, batch_size, hidden_size * 2)
                out = tf.concat([fw_out, bw_out], 2)

            # before we are using state_is_tuple=True, meaning we only chose top layer
            # now we choose both so layer 1 and layer 2 will have a difference
            # this is extracting the last hidden states

            # last layer [-1], hidden state [1]
            # this works with multilayer
            if temp_max:
                max_forward = tf.reduce_max(fw_out, axis=0)
                max_backward = tf.reduce_max(bw_out, axis=0)
                # (1, batch_size, hidden_size * 2)
                encoder_outputs = tf.concat([max_forward, max_backward], 1)
            else:
                encoder_outputs = tf.concat([output_state_fw[-1][1], output_state_bw[-1][1]], 0)

        return out, encoder_outputs


class Classifier(object):
    def __init__(self, encoder, vocab, is_training, optimizer='adam', emb_dim=100, hidden_size=256, depth=1,
                 nclasses=5,
                 multi_attn=False):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.multi_attn = multi_attn
        self.encoder = encoder
        glove_emb = vocab.vectors.numpy()

        # [time, batch]  # in sync with PyTorch
        self.seq = tf.placeholder(tf.int32, [None, None])
        self.seq_len = tf.placeholder(tf.int32, [None])

        # [batch, dim]
        self.labels = tf.placeholder(tf.float32, [None, None])

        self.keep_prob_config = 1.0 - args.dropout
        self.state_keep_prob_config = 1.0 - args.state_dropout
        self.learning_rate = tf.Variable(float(args.lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
        self.global_step = tf.Variable(0, trainable=False)

        with tf.device("/cpu:0"):
            embed = tf.Variable(glove_emb, dtype=tf.float32, name="glove")  # shape=[len(vocab), emb_dim]
            self.seq_inputs = tf.nn.embedding_lookup(embed, self.seq)

        self.build_graph()
        # both logits and label should be the same shape
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels))

        if is_training:
            # ==== set up training/updating procedure ====
            params = tf.trainable_variables()
            opt = get_optimizer(optimizer)(self.learning_rate)

            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_grad)
            self.gradient_norm = tf.global_norm(gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.max_epoch)

    def exp_mask(self, val, mask, name=None):
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
        if name is None:
            name = "exp_mask"
        return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)

    def build_graph(self):
        # time-major!
        seq_w_matrix, seq_c_vec = self.encoder.encode(self.seq_inputs, self.seq_len, temp_max=True)

        if not self.multi_attn:
            # normal classification
            # seq_c_vec: (batch_size, hidden_size)
            self.logits = rnn_cell_impl._linear([seq_c_vec], output_size=self.nclasses, bias=True)
        else:
            # seq_w_matrix: (T, batch_size, hidden_size)
            self.task_queries = tf.get_variable("taskQueries", shape=(self.hidden_size, self.nclasses), dtype=tf.float32)
            self.out_proj = tf.get_variable("outProj", shape=(1, self.nclasses, self.hidden_size), dtype=tf.float32)

            # define the process here
            # (seq_len, batch_size, hid_dim) x task_queries: (hid_dim, label_size)
            # (seq_len, batch_size, label_size)
            keys = tf.matmul(seq_w_matrix, self.task_queries)

            mask = seq_w_matrix == 0  # padded parts are outputted as 0.
            masked_keys = self.exp_mask(seq_w_matrix, mask)

            # softmax over seq_len
            keys_normalized = tf.nn.softmax(keys, dim=0)

            task_specific_list = []
            for t_n in xrange(self.nclasses):
                # (seq_len, batch_size, hid_dim) x (seq_len, batch_size, 1)
                # sum over 0
                # (batch_size, hid_dim)

                # sum over T
                task_specific_list.append(tf.reduce_sum(seq_w_matrix * tf.expand_dims(keys[:, :, t_n], 2), axis=0))

            # now it's (batch_size, label_size, hid_dim)
            task_specific_mix = tf.stack(task_specific_list, axis=1)
            # (batch_size, label_size)
            self.logits = tf.reduce_sum(task_specific_mix * self.out_proj, axis=2)

        self.probs = tf.nn.sigmoid(self.logits)




    def optimize(self, session, seq_tokens, seq_len, labels):
        input_feed = {}
        input_feed[self.seq] = seq_tokens
        input_feed[self.seq_len] = seq_len
        input_feed[self.labels] = labels

        input_feed[self.encoder.keep_prob] = self.keep_prob_config
        input_feed[self.encoder.state_keep_prob] = self.state_keep_prob_config

        # we get probability instead
        output_feed = [self.updates, self.probs, self.gradient_norm, self.loss, self.param_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3], outputs[4]

    def test(self, session, seq_tokens, seq_len, labels):
        input_feed = {}
        input_feed[self.seq] = seq_tokens
        input_feed[self.seq_len] = seq_len
        input_feed[self.labels] = labels

        input_feed[self.encoder.keep_prob] = 1.
        input_feed[self.encoder.state_keep_prob] = 1.

        output_feed = [self.loss, self.probs]

        outputs = session.run(output_feed, input_feed)

        return outputs[0], outputs[1]

    def validate(self, session, valid_iter, save_pred=False):

        all_scores, all_preds = [], []  # probability to measure uncertainty
        all_y_labels, all_print_y_labels = [], []
        all_condensed_preds, all_condensed_ys = [], []

        all_orig_texts = []

        batch_accu, cnt = 0., 0
        valid_cost = 0.

        for data in valid_iter:

            # PyTorch variables
            x = data.Text
            (seq, seq_lengths), y = data.Text, data.Description
            seq = seq.data.numpy()
            seq_lengths = seq_lengths.numpy()
            y = y.data.numpy()

            batch_size = y.shape[0]

            cost, probs = self.test(session, seq, seq_lengths, y)
            valid_cost += cost

            # gather accuracy/rec/prec/f1 etc.

            preds = output_to_preds(probs)
            preds_indices = sparse_one_hot_mat_to_indices(preds)  # numpy produces a tuple

            sparse_preds = preds_to_sparse_matrix(preds_indices, batch_size, self.nclasses)

            # TODO: this is incorrect...replace during publication
            batch_accu += metrics.accuracy_score(y, sparse_preds)
            cnt += 1

            all_scores.extend(probs.tolist())
            all_print_y_labels.extend(y.tolist())
            all_preds.append(sparse_preds)
            all_y_labels.append(y)

            orig_text = TEXT.reverse(x.data)
            all_orig_texts.extend(orig_text)

            if save_pred:
                y_indices = sparse_one_hot_mat_to_indices(y)
                condensed_preds = condense_preds(zip(preds_indices[0], preds_indices[1]), batch_size)
                condensed_ys = condense_preds(zip(y_indices[0], y_indices[1]), batch_size)

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
        logger.info("\n" + metrics.classification_report(ys, preds))

        if save_pred:
            # So the format for each entry is: y = [], pred = [], for all labels
            logging.info("Saving confusion matrix csv")
            import csv
            with open(pjoin(args.run_dir, 'confusion_test.csv'), 'wb') as csvfile:
                fieldnames = ['preds', 'labels', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for pair in zip(all_condensed_preds, all_condensed_ys, all_orig_texts):
                    writer.writerow({'preds': pair[0], 'labels': pair[1], 'text': pair[2]})

            with open(pjoin(args.run_dir, 'label_vis_map.json'), 'wb') as f:
                json.dump([all_condensed_preds, all_condensed_ys, all_scores, all_print_y_labels, all_orig_texts], f)

            with open(pjoin(args.run_dir, 'label_map.txt'), 'wb') as f:
                json.dump(labels, f)

        return valid_cost / cnt, batch_accu / cnt

    def train(self, session, train_iter, val_iter, test_iter, curr_epoch, num_epochs, save_train_dirs):

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        lr = args.lr
        epoch = curr_epoch
        best_epoch = 1
        previous_losses = []
        valid_accus = []
        exp_cost = None
        exp_norm = None

        correct = 0.0

        while num_epochs == 0 or epoch < num_epochs:
            epoch += 1
            current_step = 0

            ## Train
            epoch_tic = time.time()
            for data in train_iter:
                # Get a batch and make a step.
                tic = time.time()

                # PyTorch variables
                (seq, seq_lengths), y = data.Text, data.Description
                seq = seq.data.numpy()
                seq_lengths = seq_lengths.numpy()
                y = y.data.numpy()

                batch_size = y.shape[0]

                probs, grad_norm, cost, param_norm = self.optimize(session, seq, seq_lengths, y)

                preds = output_to_preds(probs)
                preds_indices = sparse_one_hot_mat_to_indices(preds)  # numpy produces a tuple

                sparse_preds = preds_to_sparse_matrix(preds_indices, batch_size, self.nclasses)

                batch_accu = metrics.accuracy_score(y, sparse_preds)

                toc = time.time()
                iter_time = toc - tic
                current_step += 1

                if not exp_cost:
                    exp_cost = cost
                    exp_norm = grad_norm
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * cost
                    exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

                if current_step % args.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, cost %f, exp_cost %f, accuracy %f, grad norm %f, param norm %f, batch time %f' %
                        (epoch, current_step, cost, exp_cost, batch_accu, grad_norm, param_norm, iter_time))

            epoch_toc = time.time()

            ## Checkpoint
            checkpoint_path = os.path.join(save_train_dirs, "csu.ckpt")

            ## Validate
            valid_cost, valid_accu = self.validate(session, val_iter)

            logging.info("Epoch %d Validation cost: %f validation accu: %f epoch time: %f" % (epoch, valid_cost,
                                                                                              valid_accu,
                                                                                              epoch_toc - epoch_tic))
            # only do accuracy
            if len(previous_losses) >= 1 and valid_accu < max(valid_accus):
                lr *= args.lr_decay
                logging.info("Annealing learning rate at epoch {} to {}".format(epoch, lr))
                session.run(self.learning_rate_decay_op)

                logging.info("validation cost trigger: restore model from epoch %d" % best_epoch)
                self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_losses.append(valid_cost)
                best_epoch = epoch
                self.saver.save(session, checkpoint_path, global_step=epoch)

            valid_accus.append(valid_accu)

        logging.info("restore model from best epoch %d" % best_epoch)
        self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))

        ## Test
        test_cost, test_accu = self.validate(session, test_iter, save_pred=True)
        logging.info("Final test cost: %f test accu: %f" % (test_cost, test_accu))

        sys.stdout.flush()


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
    TEXT.build_vocab(train, vectors="glove.6B.100d")

    # do repeat=False
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
        batch_sizes=(32, 256, 256), device=-1,  # on CPU instead
        sort_within_batch=True, repeat=False)  # stop infinite runs
    # if not labeling sort=False, then you are sorting through valid and test

    # this is the embedding, but we also have vocab list as well!
    vocab = TEXT.vocab

    # vocab.vectors is a Torch Tensor
    # we'll just call .numpy() to get numpy form
    if args.rand_unk:
        init_emb(vocab, init="randn")

    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=config_gpu) as session:
        tf.set_random_seed(args.seed)

        # initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)
        initializer = tf.uniform_unit_scaling_initializer(0.1, seed=args.seed)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            encoder = Encoder(size=args.d, num_layers=args.depth)
            classifier = Classifier(encoder, vocab, is_training=True, nclasses=len(labels),
                                    multi_attn=args.multi_attn,
                                    hidden_size=args.d)

        model_saver = tf.train.Saver(max_to_keep=args.max_epoch)

        tf.global_variables_initializer().run()
        classifier.train(session, train_iter, val_iter, test_iter, curr_epoch=0, num_epochs=args.max_epoch,
                         save_train_dirs=args.run_dir)
