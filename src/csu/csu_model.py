"""
Store modular components for Jupyter Notebook
"""
import json
import numpy as np
import os
import csv
import logging
import random
from sklearn import metrics
from os.path import join as pjoin

from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torchtext import data
from util import MultiLabelField, ReversibleField, BCEWithLogitsLoss, MultiMarginHierarchyLoss


class Config(dict):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.__dict__.update(**kwargs)

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))


# then we can make special class for different types of model
# each config is used to build a classifier and a trainer, so one for each
class LSTMBaseConfig(Config):
    def __init__(self, emb_dim=100, hidden_size=512, depth=1, label_size=42, bidir=False,
                 c=False, m=False, dropout=0.2, emb_update=True, clip_grad=5., seed=1234,
                 rand_unk=True, run_name="default", emb_corpus="gigaword",
                 **kwargs):
        # run_name: the folder for the trainer
        super(LSTMBaseConfig, self).__init__(emb_dim=emb_dim,
                                             hidden_size=hidden_size,
                                             depth=depth,
                                             label_size=label_size,
                                             bidir=bidir,
                                             c=c,
                                             m=m,
                                             dropout=dropout,
                                             emb_update=emb_update,
                                             clip_grad=clip_grad,
                                             seed=seed,
                                             rand_unk=rand_unk,
                                             run_name=run_name,
                                             emb_corpus=emb_corpus,
                                             **kwargs)


class LSTM_w_C_Config(LSTMBaseConfig):
    def __init__(self, sigma_M, sigma_B, sigma_W, **kwargs):
        super(LSTM_w_C_Config, self).__init__(sigma_M=sigma_M,
                                              sigma_B=sigma_B,
                                              sigma_W=sigma_W,
                                              c=True,
                                              **kwargs)


class LSTM_w_M_Config(LSTMBaseConfig):
    def __init__(self, beta, **kwargs):
        super(LSTM_w_M_Config, self).__init__(beta=beta, m=True, **kwargs)


class Classifier(nn.Module):
    def __init__(self, vocab, config):
        super(Classifier, self).__init__()
        self.config = config
        self.drop = nn.Dropout(config.dropout)  # embedding dropout
        self.encoder = nn.LSTM(
            config.emb_dim,
            config.hidden_size,
            config.depth,
            dropout=config.dropout,
            bidirectional=config.bidir)  # ha...not even bidirectional
        d_out = config.hidden_size if not config.bidir else config.hidden_size * 2
        self.out = nn.Linear(d_out, config.label_size)  # include bias, to prevent bias assignment
        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False

    def forward(self, input, lengths=None):
        output_vecs = self.get_vectors(input, lengths)
        return self.get_logits(output_vecs)

    def get_vectors(self, input, lengths=None):
        embed_input = self.embed(input)

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output)[0]

        # we ignored negative masking
        return output

    def get_logits(self, output_vec):
        output = torch.max(output_vec, 0)[0].squeeze(0)
        return self.out(output)

    def get_softmax_weight(self):
        return self.out.weight


# this dataset can also take in 5-class classification
class Dataset(object):
    def __init__(self, path='./data/csu/',
                 dataset_prefix='snomed_multi_label_no_des_',
                 test_data_name='adobe_abbr_matched_snomed_multi_label_no_des_test.tsv',
                 label_size=42):
        self.TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False)
        self.LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=label_size,
                                     tensor_type=torch.FloatTensor)

        # it's actually this step that will take 5 minutes
        self.train, self.val, self.test = data.TabularDataset.splits(
            path=path, train=dataset_prefix + 'train.tsv',
            validation=dataset_prefix + 'valid.tsv',
            test=dataset_prefix + 'test.tsv', format='tsv',
            fields=[('Text', self.TEXT), ('Description', self.LABEL)])

        self.external_test = data.TabularDataset(path=path + test_data_name,
                                                 format='tsv',
                                                 fields=[('Text', self.TEXT), ('Description', self.LABEL)])

        self.is_vocab_bulit = False
        self.iterators = []
        self.test_iterator = None

    def init_emb(self, vocab, init="randn", num_special_toks=2, silent=False):
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
        if not silent:
            print("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
                running_norm / num_non_zero, num_non_zero, total_words))  # directly printing into Jupyter Notebook

    def build_vocab(self, config, silent=False):
        if config.emb_corpus == 'common_crawl':
            self.TEXT.build_vocab(self.train, vectors="glove.840B.300d")
        else:
            self.TEXT.build_vocab(self.train, vectors="glove.6B.{}d".format(config.emb_dim))
        self.is_vocab_bulit = True
        self.vocab = self.TEXT.vocab
        if config.rand_unk:
            if not silent:
                print("initializing random vocabulary")
            self.init_emb(self.vocab, silent=silent)

    def get_iterators(self, device):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if len(self.iterators) > 0:
            return self.iterators  # return stored iterator

        # only get them after knowing the device (inside trainer or evaluator)
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (self.train, self.val, self.test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
            batch_sizes=(32, 128, 128), device=device,
            sort_within_batch=True, repeat=False)

        return train_iter, val_iter, test_iter

    def get_test_iterator(self, device):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if self.test_iterator is not None:
            return self.test_iterator

        external_test_iter = data.Iterator(self.external_test, 128, sort_key=lambda x: len(x.Text),
                                           device=device, train=False, repeat=False, sort_within_batch=True)
        return external_test_iter


# compute loss
class ClusterLoss(nn.Module):
    def __init__(self, config, cluster_path='./data/csu/snomed_label_to_meta_grouping.json'):
        super(ClusterLoss, self).__init__()

        with open(cluster_path, 'rb') as f:
            label_grouping = json.load(f)

        self.meta_category_groups = label_grouping.values()
        self.config = config

    def forward(self, softmax_weight, batch_size):
        w_bar = softmax_weight.sum(1) / self.config.label_size  # w_bar

        omega_mean = softmax_weight.pow(2).sum()
        omega_between = 0.
        omega_within = 0.

        for c in xrange(len(self.meta_category_groups)):
            m_c = len(self.meta_category_groups[c])
            w_c_bar = softmax_weight[:, self.meta_category_groups[c]].sum(1) / m_c
            omega_between += m_c * (w_c_bar - w_bar).pow(2).sum()
            for i in self.meta_category_groups[c]:
                # this value will be 0 for singleton group
                omega_within += (softmax_weight[:, i] - w_c_bar).pow(2).sum()

        aux_loss = omega_mean * self.config.sigma_M + (omega_between * self.config.sigma_B +
                                                       omega_within * self.config.sigma_W) / batch_size

        return aux_loss


class MetaLoss(nn.Module):
    def __init__(self, config, cluster_path='./data/csu/snomed_label_to_meta_grouping.json',
                 label_to_meta_map_path='./data/csu/snomed_label_to_meta_map.json'):
        super(MetaLoss, self).__init__()

        with open(cluster_path, 'rb') as f:
            self.label_grouping = json.load(f)

        with open(label_to_meta_map_path, 'rb') as f:
            self.meta_label_mapping = json.load(f)

        self.meta_label_size = len(self.label_grouping)
        self.config = config

        # your original classifier did this wrong...found a bug
        self.bce_loss = nn.BCELoss()  # this takes in probability (after sigmoid)

    # now that this becomes somewhat independent...maybe you can examine this more closely?
    def generate_meta_y(self, indices, meta_label_size, batch_size):
        a = np.array([[0.] * meta_label_size for _ in range(batch_size)], dtype=np.float32)
        matched = defaultdict(set)
        for b, l in indices:
            if b not in matched:
                a[b, self.meta_label_mapping[str(l)]] = 1.
                matched[b].add(self.meta_label_mapping[str(l)])
            elif self.meta_label_mapping[str(l)] not in matched[b]:
                a[b, self.meta_label_mapping[str(l)]] = 1.
                matched[b].add(self.meta_label_mapping[str(l)])
        assert np.sum(a <= 1) == a.size
        return a

    def forward(self, logits, true_y, device):
        batch_size = logits.size(0)
        y_hat = torch.sigmoid(logits)
        meta_probs = []
        for i in range(self.meta_label_size):
            # 1 - (1 - p_1)(...)(1 - p_n)
            meta_prob = (1 - y_hat[:, self.label_grouping[str(i)]]).prod(1)
            meta_probs.append(meta_prob)  # in this version we don't do threshold....(originally we did)

        meta_probs = torch.stack(meta_probs, dim=1)
        assert meta_probs.size(1) == self.meta_label_size

        # generate meta-label
        y_indices = true_y.nonzero()
        meta_y = self.generate_meta_y(y_indices.data.cpu().numpy().tolist(), self.meta_label_size,
                                      batch_size)
        meta_y = Variable(torch.from_numpy(meta_y)) if device == -1 else Variable(torch.from_numpy(meta_y)).cuda(device)

        meta_loss = self.bce_loss(meta_probs, meta_y) * self.config.beta
        return meta_loss


# maybe we should evaluate inside this
# currently each Trainer is tied to one GPU, so we don't have to worry about
# Each trainer is associated with a config and classifier actually...so should be associated with a log
# Experiment class will create a central folder, and it will have sub-folder for each trainer
# central folder will have an overall summary...(Experiment will also have ways to do 5 random seed exp)
class Trainer(object):
    def __init__(self, classifier, dataset, config, save_path, device, load=False,
                 **kwargs):
        # save_path: where to save log and model
        if load:
            self.classifier = torch.load(pjoin(save_path, 'model.pickle')).cuda(device)
        else:
            self.classifier = classifier.cuda(device)

        self.dataset = dataset
        self.device = device
        self.config = config
        self.save_path = save_path

        self.train_iter, self.val_iter, self.test_iter = self.dataset.get_iterators(device)
        self.external_test_iter = self.dataset.get_test_iterator(device)

        if config.m:
            self.aux_loss = MetaLoss(config, **kwargs)
        elif config.c:
            self.aux_loss = ClusterLoss(config, **kwargs)

        self.bce_logit_loss = BCEWithLogitsLoss(reduce=False)

        need_grad = lambda x: x.requires_grad
        self.optimizer = optim.Adam(
            filter(need_grad, classifier.parameters()),
            lr=0.001)  # obviously we could use config to control this

        # setting up logging
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
        file_handler = logging.FileHandler("{0}/log.txt".format(save_path))
        self.logger = logging.getLogger(save_path.split('/')[-1])  # so that no model is sharing logger
        self.logger.addHandler(file_handler)

        self.logger.info(config)

    def train(self, epochs=5, no_print=True):
        # train loop
        exp_cost = None
        for e in range(epochs):
            self.classifier.train()
            for iter, data in enumerate(self.train_iter):
                self.classifier.zero_grad()
                (x, x_lengths), y = data.Text, data.Description

                # output_vec = self.classifier.get_vectors(x, x_lengths)  # this is just logit (before calling sigmoid)
                # final_rep = torch.max(output_vec, 0)[0].squeeze(0)
                # logits = self.classifier.get_logits(output_vec)

                logits = self.classifier(x, x_lengths)

                batch_size = x.size(0)

                if self.config.c:
                    softmax_weight = self.classifier.get_softmax_weight()
                    aux_loss = self.aux_loss(softmax_weight, batch_size)
                elif self.config.m:
                    aux_loss = self.aux_loss(logits, y, self.device)
                else:
                    aux_loss = 0.

                loss = self.bce_logit_loss(logits, y).mean() + aux_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.config.clip_grad)
                self.optimizer.step()

                if not exp_cost:
                    exp_cost = loss.data[0]
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

                if iter % 100 == 0:
                    self.logger.info(
                        "iter {} lr={} train_loss={} exp_cost={} \n".format(iter, self.optimizer.param_groups[0]['lr'],
                                                                            loss.data[0], exp_cost))
            self.logger.info("enter validation...")
            valid_em, micro_tup, macro_tup = self.evaluate(is_test=False)
            self.logger.info("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
                e + 1, self.optimizer.param_groups[0]['lr'], loss.data[0], valid_em
            ))

        # save model
        torch.save(self.classifier, pjoin(self.save_path, 'model.pickle'))

    def test(self, silent=False, return_by_label_stats=False, return_instances=False):
        self.logger.info("compute test set performance...")
        return self.evaluate(is_test=True, silent=silent, return_by_label_stats=return_by_label_stats,
                             return_instances=return_instances)

    def evaluate(self, is_test=False, is_external=False, silent=False, return_by_label_stats=False,
                 return_instances=False):
        self.classifier.eval()
        data_iter = self.test_iter if is_test else self.val_iter  # evaluate on CSU
        data_iter = self.external_test_iter if is_external else data_iter  # evaluate on adobe

        all_preds, all_y_labels = [], []

        for iter, data in enumerate(data_iter):
            (x, x_lengths), y = data.Text, data.Description
            logits = self.classifier(x, x_lengths)

            preds = (torch.sigmoid(logits) > 0.5).data.cpu().numpy().astype(float)
            all_preds.append(preds)
            all_y_labels.append(y.data.cpu().numpy())

        preds = np.vstack(all_preds)
        ys = np.vstack(all_y_labels)

        if not silent:
            self.logger.info("\n" + metrics.classification_report(ys, preds, digits=3))  # write to file

        # this is actually the accurate exact match
        em = metrics.accuracy_score(ys, preds)
        p, r, f1, s = metrics.precision_recall_fscore_support(ys, preds, average=None)

        if return_by_label_stats:
            return p, r, f1, s
        elif return_instances:
            return ys, preds

        micro_p, micro_r, micro_f1 = np.average(p, weights=s), np.average(r, weights=s), np.average(f1, weights=s)

        # compute Macro-F1 here
        # if is_external:
        #     # include clinical finding
        #     macro_p, macro_r, macro_f1 = np.average(p[14:]), np.average(r[14:]), np.average(f1[14:])
        # else:
        #     # anything > 10
        #     macro_p, macro_r, macro_f1 = np.average(np.take(p, [12] + range(21, 42))), \
        #                                  np.average(np.take(r, [12] + range(21, 42))), \
        #                                  np.average(np.take(f1, [12] + range(21, 42)))

        # we switch to non-zero macro computing, this can figure out boost from rarest labels
        if is_external:
            # include clinical finding
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), np.average(r[r.nonzero()]), \
                                         np.average(f1[f1.nonzero()])
        else:
            # anything > 10
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), \
                                         np.average(r[r.nonzero()]), \
                                         np.average(f1[f1.nonzero()])

        return em, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)


# Experiment class can also be "handled" by Jupyter Notebook
# Usage guide:
# config also manages random seed. So it's possible to just swap in and out random seed from config
# to run an average, can write it into another function inside Experiment class called `repeat_execute()`
# also, currently once trainer is deleted, the classifier pointer would be lost...completely
class Experiment(object):
    def __init__(self, dataset, exp_save_path):
        """
        :param dataset: Dataset class
        :param exp_save_path: the overall saving folder
        """
        if not os.path.exists(exp_save_path):
            os.makedirs(exp_save_path)

        self.dataset = dataset
        self.exp_save_path = exp_save_path

        # we never want to overwrite this file
        if not os.path.exists(pjoin(exp_save_path, "all_runs_stats.csv")):
            with open(pjoin(self.exp_save_path, "all_runs_stats.csv"), 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['model', 'CSU EM', 'CSU micro-P', 'CSU micro-R', 'CSU micro-F1',
                                     'CSU macro-P', 'CSU macro-R', 'CSU macro-F1',
                                     'PP EM', 'PP micro-P', 'PP micro-R', 'PP micro-F1',
                                     'PP macro-P', 'PP macro-R', 'PP macro-F1'])

    def get_trainer(self, config, device, build_vocab=False, load=False, silent=True, **kwargs):
        # build each trainer and classifier by config; or reload classifier
        # **kwargs: additional commands for the two losses

        if build_vocab:
            self.dataset.build_vocab(config, silent)  # because we might try different word embedding size

        self.set_random_seed(config)

        classifier = Classifier(self.dataset.vocab, config)
        trainer_folder = config.run_name if config.run_name != 'default' else self.config_to_string(config)
        trainer = Trainer(classifier, self.dataset, config,
                          save_path=pjoin(self.exp_save_path, trainer_folder),
                          device=device, load=load, **kwargs)

        return trainer

    def set_random_seed(self, config):
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def config_to_string(self, config):
        # we compare config to baseline config, if values are modified, we produce it into string
        model_name = "mod"  # this will be the "baseline"
        base_config = LSTMBaseConfig()
        for k, new_v in config.items():
            if k in base_config.keys():
                old_v = base_config[k]
                if old_v != new_v:
                    model_name += "_{}_{}".format(k, new_v)
            else:
                model_name += "_{}_{}".format(k, new_v)

        return model_name.replace('.', '').replace('-', '_')  # for 1e-3 to 1e_3

    def record_meta_result(self, meta_results, append, config):
        # this records result one line at a time!
        mode = 'a' if append else 'w'
        model_str = self.config_to_string(config)

        csu_em, csu_micro_tup, csu_macro_tup, \
        pp_em, pp_micro_tup, pp_macro_tup = meta_results

        with open(pjoin(self.exp_save_path, "all_runs_stats.csv"), mode=mode) as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([model_str, csu_em, csu_micro_tup[0],
                                 csu_micro_tup[1], csu_micro_tup[2],
                                 csu_macro_tup[0], csu_macro_tup[1], csu_macro_tup[2],
                                 pp_em, pp_micro_tup[0], pp_micro_tup[1], pp_micro_tup[2],
                                 pp_macro_tup[0], pp_macro_tup[1], pp_macro_tup[2]])

    def execute(self, trainer, append=True):
        # the benefit of this function is it will record meta-result into a file...
        trainer.train()
        csu_em, csu_micro_tup, csu_macro_tup = trainer.test()
        trainer.logger.info("===== Evaluating on PP data =====")
        pp_em, pp_micro_tup, pp_macro_tup = trainer.evaluate(is_external=True)
        trainer.logger.info("PP accuracy = {}".format(pp_em))
        self.record_meta_result([csu_em, csu_micro_tup, csu_macro_tup,
                                 pp_em, pp_micro_tup, pp_macro_tup],
                                append=append, config=trainer.config)

    def delete_trainer(self, trainer):
        # move all parameters to cpu and then delete the pointer
        trainer.classifier.cpu()
        del trainer.classifier
        del trainer

    def re_execute(self, trainer, write_to_meta=False):
        # load in previous model in get_trainer(), and just get results, no recording
        csu_em, csu_micro_tup, csu_macro_tup = trainer.test(silent=True)
        trainer.logger.info("===== Evaluating on PP data =====")
        pp_em, pp_micro_tup, pp_macro_tup = trainer.evaluate(is_external=True, silent=True)
        if write_to_meta:
            self.record_meta_result([csu_em, csu_micro_tup, csu_macro_tup,
                                     pp_em, pp_micro_tup, pp_macro_tup],
                                    append=True, config=trainer.config)


# Important! Each time you use "get_iterators", must restore previous random state
# otherwise the sampling procedure will be different
def run_baseline(device):
    random.setstate(orig_state)
    lstm_base_c = LSTMBaseConfig(emb_corpus=emb_corpus)
    trainer = curr_exp.get_trainer(config=lstm_base_c, device=device, build_vocab=True)
    curr_exp.execute(trainer=trainer)


def run_bidir_baseline(device):
    random.setstate(orig_state)
    lstm_bidir_c = LSTMBaseConfig(bidir=True, emb_corpus=emb_corpus)
    trainer = curr_exp.get_trainer(config=lstm_bidir_c, device=device, build_vocab=True)
    curr_exp.execute(trainer=trainer)


def run_m_penalty(device, beta=1e-3, bidir=False):
    random.setstate(orig_state)
    config = LSTM_w_M_Config(beta, bidir=bidir, emb_corpus=emb_corpus)
    trainer = curr_exp.get_trainer(config=config, device=device, build_vocab=True)
    curr_exp.execute(trainer=trainer)


def run_c_penalty(device, sigma_M, sigma_B, sigma_W, bidir=False):
    random.setstate(orig_state)
    config = LSTM_w_C_Config(sigma_M, sigma_B, sigma_W, bidir=bidir, emb_corpus=emb_corpus)
    trainer = curr_exp.get_trainer(config=config, device=device, build_vocab=True)
    curr_exp.execute(trainer=trainer)


# TODO: maybe code in multi-run support (over many random seeds)
if __name__ == '__main__':
    # if we just call this file, it will set up an interactive console
    random.seed(1234)

    # we get the original random state, and simply reset during each run
    orig_state = random.getstate()

    action = raw_input("enter branches of default actions: active | baseline | meta | cluster \n")

    device_num = int(raw_input("enter the GPU device number \n"))
    assert -1 <= device_num <= 3, "GPU ID must be between -1 and 3"

    exp_name = raw_input("enter the experiment name, default is 'csu_new_exp', skip to use default: ")
    exp_name = 'csu_new_exp' if exp_name.strip() == '' else exp_name

    print("loading in dataset...will take 3-4 minutes...")
    dataset = Dataset()

    curr_exp = Experiment(dataset=dataset, exp_save_path='./{}/'.format(exp_name))
    emb_corpus = raw_input("enter embedding choice: gigaword | common_crawl \n")
    assert emb_corpus == 'gigaword' or emb_corpus == 'common_crawl'

    if action == 'active':
        import IPython; IPython.embed()
    elif action == 'baseline':
        # baseline LSTM
        run_baseline(device_num)
        run_bidir_baseline(device_num)
    elif action == 'meta':
        # baseline LSTM + M
        # run_m_penalty(device_num, beta=1e-3)
        # run_m_penalty(device_num, beta=1e-4)

        run_baseline(device_num)
        run_bidir_baseline(device_num)

        # baseline LSTM + M + bidir
        run_m_penalty(device_num, beta=1e-4, bidir=True)
        run_m_penalty(device_num, beta=1e-3, bidir=True)

        run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4, bidir=True)
        run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3, bidir=True)

        run_m_penalty(device_num, beta=1e-4)
        run_m_penalty(device_num, beta=1e-3)

        run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4)
        run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3)

    elif action == 'cluster':
        # baseline LSTM + C
        run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4)
        run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3)

        # baseline LSTM + C + bidir
        run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4, bidir=True)
        run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3, bidir=True)
    else:
        print("Non-identifiable action: {}".format(action))
