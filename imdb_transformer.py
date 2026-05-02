import logging
import os
import sys
import time
import math
import re
import json

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


test = pd.read_csv("./data/testData.tsv/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.05
device = torch.device('cuda:0')
use_gpu = True
MAX_SEQ_LEN = 512
DEBUG_LOG_PATH = "debug-5b94cc.log"
DEBUG_SESSION_ID = "5b94cc"


def debug_log(run_id, hypothesis_id, location, message, data):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000)
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


# Read data from files
train = pd.read_csv("./data/labeledTrainData.tsv/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./data/testData.tsv/testData.tsv", header=0, delimiter="\t", quoting=3)


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    # if remove_stopwords:
    #     stops = set(stopwords.words("english"))
    #     words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, train, test, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in train:
            for token in sentence:
                token_freqs[token] += 1

        for sentence in test:
            for token in sentence:
                token_freqs[token] += 1

        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def length_to_mask(lengths):
    max_length = torch.max(lengths)
    max_length_int = int(max_length.item())
    # region agent log
    debug_log(
        run_id="post-fix",
        hypothesis_id="H6",
        location="imdb_transformer.py:length_to_mask:before_mask",
        message="length_to_mask device info",
        data={
            "lengths_device": str(lengths.device),
            "max_length": max_length_int
        }
    )
    # endregion
    mask = torch.arange(max_length_int, device=lengths.device).expand(lengths.shape[0], max_length_int) < lengths.unsqueeze(1)
    # region agent log
    debug_log(
        run_id="post-fix",
        hypothesis_id="H6",
        location="imdb_transformer.py:length_to_mask:after_mask",
        message="mask device info",
        data={
            "mask_device": str(mask.device),
            "mask_shape": list(mask.size())
        }
    )
    # endregion
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = self._build_pe(max_len)
        self.register_buffer('pe', pe)

    def _build_pe(self, max_len, device=None):
        pe = torch.zeros(max_len, self.d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        # region agent log
        debug_log(
            run_id="pre-fix",
            hypothesis_id="H1_H2",
            location="imdb_transformer.py:forward:PositionalEncoding",
            message="shape before positional add",
            data={
                "x_shape": list(x.size()),
                "requested_seq_len": int(x.size(0)),
                "pe_shape": list(self.pe.size()),
                "pe_available_seq_len": int(self.pe.size(0))
            }
        )
        # endregion
        if x.size(0) > self.pe.size(0):
            self.pe = self._build_pe(x.size(0), device=x.device)
        x = x + self.pe[:x.size(0), :]
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=128, activation: str = "relu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出层
        self.output = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs, lengths):
        # region agent log
        debug_log(
            run_id="pre-fix",
            hypothesis_id="H2_H4",
            location="imdb_transformer.py:forward:Transformer:entry",
            message="transformer forward input",
            data={
                "inputs_shape_before_transpose": list(inputs.size()),
                "lengths_shape": list(lengths.size()),
                "lengths_max": int(lengths.max().item()),
                "lengths_min": int(lengths.min().item())
            }
        )
        # endregion
        inputs = torch.transpose(inputs, 0, 1)
        # region agent log
        debug_log(
            run_id="pre-fix",
            hypothesis_id="H2_H4",
            location="imdb_transformer.py:forward:Transformer:post_transpose",
            message="shape after transpose",
            data={
                "inputs_shape_after_transpose": list(inputs.size())
            }
        )
        # endregion
        hidden_states = self.embeddings(inputs)
        # region agent log
        debug_log(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="imdb_transformer.py:forward:Transformer:post_embedding",
            message="shape after embedding",
            data={
                "hidden_states_shape": list(hidden_states.size()),
                "embedding_dim": int(self.embedding_dim)
            }
        )
        # endregion
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    raw_lengths = [len(ex[0]) for ex in examples]
    lengths = torch.tensor([min(l, MAX_SEQ_LEN) for l in raw_lengths])
    inputs = [torch.tensor(ex[0][:MAX_SEQ_LEN]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    # region agent log
    debug_log(
        run_id="pre-fix",
        hypothesis_id="H1_H3",
        location="imdb_transformer.py:collate_fn",
        message="batch padded lengths",
        data={
            "batch_size": int(len(examples)),
            "max_length_in_batch": int(lengths.max().item()),
            "min_length_in_batch": int(lengths.min().item()),
            "raw_max_length_in_batch": int(max(raw_lengths)),
            "raw_min_length_in_batch": int(min(raw_lengths)),
            "truncated_to": int(MAX_SEQ_LEN),
            "padded_input_shape": list(inputs.size())
        }
    )
    # endregion
    return inputs, lengths, targets


def collate_fn_test(examples):
    raw_lengths = [len(ex) for ex in examples]
    lengths = torch.tensor([min(l, MAX_SEQ_LEN) for l in raw_lengths])
    inputs = [torch.tensor(ex[:MAX_SEQ_LEN]) for ex in examples]
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    clean_train_reviews, train_labels = [], []
    for i, review in enumerate(train["review"]):
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=False))
        train_labels.append(train["sentiment"][i])

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=False))

    vocab = Vocab.build(clean_train_reviews, clean_test_reviews)

    train_reviews = [(vocab.convert_tokens_to_ids(sentence), train_labels[i])
                     for i, sentence in enumerate(clean_train_reviews)]
    test_reviews = [vocab.convert_tokens_to_ids(sentence)
                     for sentence in clean_test_reviews]

    train_reviews, val_reviews, train_labels, val_labels = train_test_split(train_reviews, train_labels,
                                                                            test_size=0.2, random_state=0)

    net = Transformer(vocab_size=len(vocab), embedding_dim=embed_size, hidden_dim=num_hiddens, num_class=labels)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = TransformerDataset(train_reviews)
    val_set = TransformerDataset(val_reviews)
    test_set = TransformerDataset(test_reviews)

    train_iter = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_test, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, lengths, label in train_iter:
                print(feature, lengths, label)
                n += 1
                net.zero_grad()
                feature = Variable(feature.cuda())
                lengths = Variable(lengths.cuda())
                label = Variable(label.cuda())
                score = net(feature, lengths)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                         dim=1), label.cpu())
                train_loss += loss

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)

            with torch.no_grad():
                for val_feature, val_length, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.cuda()
                    val_length = val_length.cuda()
                    val_label = val_label.cuda()
                    val_score = net(val_feature, val_length)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)
                              })

            # tqdm.write('{epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f}' %
            #       (epoch, train_loss.data / n, train_acc / n, val_losses.data / m, val_acc / m, runtime))

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, test_length in test_iter:
                test_feature = test_feature.cuda()
                test_length = test_length.cuda()
                test_score = net(test_feature, test_length)
                # test_pred.extent
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())

                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/transformer.csv", index=False, quoting=3)
    logging.info('result saved!')

