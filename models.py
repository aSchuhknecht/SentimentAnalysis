# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import nltk

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:

        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:

        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:

        return 1


class NeuralSentimentClassifier(SentimentClassifier):

    def __init__(self, inp, hid, out, embed):
        self.classifier = self.FFNN(inp, hid, out, embed)
        self.word_embed = embed
        # raise Exception("Must be implemented")

    class FFNN(nn.Module):

        def __init__(self, inp, hid, out, embed):

            super().__init__()

            self.embed = embed.get_initialized_embedding_layer(frozen=False)
            self.V1 = nn.Linear(inp, hid)
            self.g1 = nn.ReLU()
            self.V2 = nn.Linear(hid, hid)
            self.g2 = nn.ReLU()
            self.V3 = nn.Linear(hid, hid)
            self.g3 = nn.ReLU()
            self.W = nn.Linear(hid, out)
            self.log_softmax = nn.LogSoftmax(dim=1)
            # Initialize weights according to a formula due to Xavier Glorot.
            nn.init.xavier_uniform_(self.V1.weight)
            nn.init.xavier_uniform_(self.V2.weight)
            nn.init.xavier_uniform_(self.V3.weight)
            nn.init.xavier_uniform_(self.W.weight)

        def forward(self, x, print_data):

            e = self.embed(x)
            avg = torch.mean(e, dim=1)
            if print_data:
                print(x)
                print(avg)
            return self.log_softmax(self.W(self.g3(self.V3(self.g2(self.V2(self.g1(self.V1(avg))))))))

    def predict(self, ex_words: List[str], has_typos: bool) -> int:

        x = []
        max_length = 50
        for i in range(0, len(ex_words)):

            e = self.word_embed.word_indexer.index_of(ex_words[i])
            if e == -1 and 3 < len(ex_words[i]) and has_typos:

                word = list(ex_words[i])
                # word = ['m', 'o', 'v', 'v', 'e']
                # print(word)
                for k in range(0, len(word) - 3):

                    for letter in alphabet:
                        temp = word.copy()
                        temp[len(word) - 1 - k] = letter
                        guess = "".join(temp)
                        res = self.word_embed.word_indexer.index_of(guess)
                        if res != -1:
                            # print("Fix " + guess)
                            e = res
                            break

            x.append(abs(e))

        if len(x) < max_length:
            for z in range(0, max_length - len(x)):
                x.append(0)
        # truncate longer ones
        x = x[0:max_length]

        tens = torch.Tensor(x).int()
        tens = tens.unsqueeze(dim=0)  # for batching

        log_probs = self.classifier.forward(tens, 0)
        res = torch.argmax(log_probs[0]).item()

        # print(log_probs)
        return res


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:

    feat_vec_size = word_embeddings.get_embedding_length()
    embedding_size = 100
    num_classes = 2
    # RUN TRAINING AND TEST
    num_epochs = 10
    batch_size = 32
    max_length = 50  # for padding

    sent = NeuralSentimentClassifier(feat_vec_size, embedding_size, num_classes, word_embeddings)
    initial_learning_rate = 0.001
    optimizer = optim.Adam(sent.classifier.parameters(), lr=initial_learning_rate)

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        num_batches = len(train_exs) // batch_size
        # num_batches = 1

        offset = 0
        for b in range(0, num_batches):

            train_xs = []
            y = []
            for k in range(0, batch_size):
                train_xs.append(train_exs[ex_indices[offset]].words)
                y.append(train_exs[ex_indices[offset]].label)
                offset += 1

            x = []
            for i in range(0, len(train_xs)):
                xi = []
                for j in range(0, len(train_xs[i])):
                    e = word_embeddings.word_indexer.index_of(train_xs[i][j])
                    xi.append(abs(e))  # abs() used to flop -1 to 1
                x.append(xi)

            # perform padding
            for i in range(0, len(x)):
                if len(x[i]) < max_length:
                    for z in range(0, max_length - len(x[i])):
                        x[i].append(0)
                # truncate longer ones
                x[i] = x[i][0:max_length]

            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            sent.classifier.zero_grad()

            log_probs = sent.classifier.forward(torch.Tensor(x).int(), 0)

            loss = nn.NLLLoss()
            output = loss(log_probs, torch.Tensor(y).long())
            total_loss += output
            # Computes the gradient and takes the optimizer step
            output.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return sent
    # raise NotImplementedError

