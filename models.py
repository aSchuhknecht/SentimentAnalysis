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
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
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
            """
            Runs the neural network on the given data and returns log probabilities of the various classes.

            :param x: a [inp]-sized tensor of input data
            :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
            probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
            """

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
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

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

