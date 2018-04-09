import torch
from torchtext import datasets
from torchtext.data.field import Field
from torchtext import data
from torch import nn
import argparse
import sys
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from sklearn import metrics

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--rand_unk", action="store_true", help="randomly initialize unk")
argparser.add_argument("--emb_update", action="store_true", help="update embedding")
argparser.add_argument("--seed", type=int, default=123)
argparser.add_argument("--emb", type=int, default=50)
argparser.add_argument("--hid", type=int, default=50)
argparser.add_argument("--clip_grad", type=float, default=5)

args = argparser.parse_args()

"""
Seeding
"""
torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


class ReversibleField(Field):
    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is list:
            self.use_revtok = False
        else:
            self.use_revtok = True
        if kwargs.get('tokenize') not in ('revtok', 'subword', list):
            kwargs['tokenize'] = 'revtok'
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = ' UNK '
        super(ReversibleField, self).__init__(**kwargs)

    def reverse(self, batch):
        if self.use_revtok:
            try:
                import revtok
            except ImportError:
                print("Please install revtok.")
                raise
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        if self.use_revtok:
            return [revtok.detokenize(ex) for ex in batch]
        return [' '.join(ex) for ex in batch]


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
    print("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
        running_norm / num_non_zero, num_non_zero, total_words))


class RNNClassifier(nn.Module):
    def __init__(self, vocab, emb_dim=50, hidden_size=50, depth=1, nclasses=5,
                 scaled_dot_attn=False, temp_max_pool=False, attn_head=1):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(0.2)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_size,
            depth,
            dropout=0.2,
            bidirectional=False)  # ha...not even bidirectional
        d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses, bias=True)
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if args.emb_update else False

    def forward(self, input, lengths=None):
        embed_input = self.embed(input)

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output)[0]

        return self.out(hidden)


def train_module(model, optimizer,
                 train_iter, valid_iter, max_epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()

    exp_cost = None
    iter = 0
    best_valid = 0.
    epoch = 1

    for n in range(max_epoch):
        for data in train_iter:
            iter += 1

            model.zero_grad()
            (x, x_lengths), y = data.Text, data.Description

            output = model(x)

            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if not exp_cost:
                exp_cost = loss.data[0]
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

            if iter % 200 == 0:
                print("iter {} lr={} train_loss={} exp_cost={} \n".format(iter,
                                                                          optimizer.param_groups[
                                                                              0][
                                                                              'lr'],
                                                                          loss.data[
                                                                              0],
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


def eval_model(model, valid_iter, save_pred=False):
    # when test_final is true, we save predictions
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0

    all_preds = []
    all_y_labels = []
    all_orig_texts = []
    for data in valid_iter:
        (x, x_lengths), y = data.Text, data.Description
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.data[0] * x.size(1)

        preds = output.data.max(1)[1]  # already taking max...I think, max returns a tuple
        correct += preds.eq(y.data).cpu().sum()
        cnt += y.numel()

        orig_text = TEXT.reverse(x.data)
        all_orig_texts.extend(orig_text)

        all_preds.extend(preds.data.cpu().numpy().tolist())
        all_y_labels.extend(y.data.cpu().numpy().tolist())

    print("\n" + metrics.classification_report(all_y_labels, all_preds))
    model.train()

    return correct / cnt


if __name__ == '__main__':
    TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train, val, test = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train, vectors="glove.6B.{}d".format(args.emb))

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
        batch_sizes=(32, 256, 256), device=0,
        sort_within_batch=True, repeat=False)

    vocab = TEXT.vocab
    if args.rand_unk:
        init_emb(vocab, init="randn")

    # IMDB is binary
    model = RNNClassifier(vocab, nclasses=2, emb_dim=args.emb, hidden_size=args.hid)

    if torch.cuda.is_available():
        model.cuda()

    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr=0.001)

    train_module(model, optimizer, train_iter, val_iter,
                 max_epoch=10)

    test_accu = eval_model(model, test_iter)
    print("final test accu: {}".format(test_accu))
