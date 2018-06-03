"""
Store modular components for Jupyter Notebook
"""

from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torchtext import data
from util import MultiLabelField, ReversibleField, BCEWithLogitsLoss, MultiMarginHierarchyLoss


class Config(object):
    def __init__(self, **kwargs):
        self.hparams = defaultdict(**kwargs)

    def __getitem__(self, key):
        return self.hparams[key]

    def __missing__(self, key):
        return False

    def __str__(self):
        return str(self.hparams)

    def __setitem__(self, key, value):
        self[key] = value

    def __getattr__(self, name):
        return self[name]


# then we can make special class for different types of model

class LSTMBaseConfig(Config):
    def __init__(self, emb_dim=100, hidden_size=512, depth=1, nclasses=5, bidir=False,
                 c=False, m=False, dropout=0.2, emb_update=True,
                 **kwargs):
        super(LSTMBaseConfig, self).__init__(emb_dim=emb_dim,
                                             hidden_size=hidden_size,
                                             depth=depth,
                                             nclasses=nclasses,
                                             bidir=bidir,
                                             c=c,
                                             m=m,
                                             dropout=dropout,
                                             emb_update=emb_update,
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
        self.out = nn.Linear(d_out, config.nclasses)  # include bias, to prevent bias assignment
        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False

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
        TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False)
        LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=label_size, tensor_type=torch.FloatTensor)

        self.train, self.val, self.test = data.TabularDataset.splits(
            path=path, train=dataset_prefix + 'train.tsv',
            validation=dataset_prefix + 'valid.tsv',
            test=dataset_prefix + 'test.tsv', format='tsv',
            fields=[('Text', TEXT), ('Description', LABEL)])

        self.external_test = data.TabularDataset(path=path + test_data_name,
                                                 format='tsv',
                                                 fields=[('Text', TEXT), ('Description', LABEL)])

    def get_iterators(self, device):
        # only get them after knowing the device (inside trainer or evaluator)
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (self.train, self.val, self.test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
            batch_sizes=(32, 128, 128), device=device,
            sort_within_batch=True, repeat=False)

        return train_iter, val_iter, test_iter

    def get_test_iterator(self, device):
        external_test_iter = data.Iterator(self.external_test, 128, sort_key=lambda x: len(x.Text),
                                           device=device, train=False, repeat=False, sort_within_batch=True)
        return external_test_iter


class Trainer(object):
    def __init__(self, classifier, dataset, device):
        pass
