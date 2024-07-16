# sentiment_data.py

from typing import List
from utils import *
import re
import numpy as np
from collections import Counter
import torch.nn

class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()


def read_sentiment_examples(infile: str, lowercase=True) -> List[SentimentExample]:

    f = open(infile)
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            if len(fields) != 2:
                fields = line.split()
                label = 0 if "0" in fields[0] else 1
                sent = " ".join(fields[1:])
            else:
                # Slightly more robust to reading bad output than int(fields[0])
                label = 0 if "0" in fields[0] else 1
                sent = fields[1]
            if lowercase:
                sent = sent.lower()
            tokenized_cleaned_sent = list(filter(lambda x: x != '', sent.rstrip().split(" ")))
            exs.append(SentimentExample(tokenized_cleaned_sent, label))
    f.close()
    return exs


def read_blind_sst_examples(infile: str) -> List[List[str]]:

    f = open(infile, encoding='utf-8')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            exs.append(line.split(" "))
    return exs


def write_sentiment_examples(exs: List[SentimentExample], outfile: str):

    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([word for word in ex.words]) + "\n")
    o.close()


class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=True):

        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors), freeze=frozen, padding_idx=0)

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):

        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


def read_word_embeddings(embeddings_file: str) -> WordEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    # Make position 0 a PAD token, which can be useful if you
    word_indexer.add_and_get_index("PAD")
    # Make position 1 the UNK token
    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
            # of the file to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


def relativize(file, outfile, word_counter):
    """
    Relativize the word vectors to the given dataset represented by word counts
    :param file: word vectors file
    :param outfile: output file
    :param word_counter: Counter of words occurring in train/dev/test data
    :return:
    """
    f = open(file)
    o = open(outfile, 'w')
    voc = []
    for line in f:
        word = line[:line.find(' ')]
        if word_counter[word] > 0:
            # print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    for word in word_counter:
        if word not in voc:
            count = word_counter[word]
            if count > 1:
                print("Missing " + word + " with count " + repr(count))
    f.close()
    o.close()


def relativize_sentiment_data():

    word_counter = Counter()
    for ex in read_sentiment_examples("data/train.txt"):
        for word in ex.words:
            word_counter[word] += 1
    for ex in read_sentiment_examples("data/dev.txt"):
        for word in ex.words:
            word_counter[word] += 1
    for words in read_blind_sst_examples("data/test-blind.txt"):
        for word in words:
            word_counter[word] += 1
    # Uncomment these to relativize vectors to the dataset
    relativize("data/glove.6B.50d.txt", "data/glove.6B.50d-relativized.txt", word_counter)
    relativize("data/glove.6B.300d.txt", "data/glove.6B.300d-relativized.txt", word_counter)


if __name__=="__main__":
    import sys
    embs = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    query_word_1 = sys.argv[1]
    query_word_2 = sys.argv[2]
    if embs.word_indexer.index_of(query_word_1) == -1:
        print("%s is not in the indexer" % query_word_1)
    elif embs.word_indexer.index_of(query_word_2) == -1:
        print("%s is not in the indexer" % query_word_2)
    else:
        emb1 = embs.get_embedding(query_word_1)
        emb2 = embs.get_embedding(query_word_2)
        print("cosine similarity of %s and %s: %f" % (query_word_1, query_word_2, np.dot(emb1, emb2)/np.sqrt(np.dot(emb1, emb1) * np.dot(emb2, emb2))))
