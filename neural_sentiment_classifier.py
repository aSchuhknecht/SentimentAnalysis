# neural_sentiment_classifier.py

import argparse
import json
import sys
import time
from models import *
from sentiment_data import *
from typing import List


def _parse_args():

    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='DAN', help='model to run (TRIVIAL or DAN)')
    parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--use_typo_setting', dest='use_typo_setting', default=False, action='store_true',
                        help="True to use your typo model and evaluate on typos, false otherwise")
    parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt',
                        help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='test-blind.output.txt', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word embeddings to use')
    # Some common args have been pre-populated for you. Again, you can add more during development, but your code needs
    # to run with the default neural_sentiment_classifier for submission.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size; 1 by default and you do not need to batch unless you want to')
    args = parser.parse_args()
    return args


def evaluate(classifier, exs, has_typos):

    return print_evaluation([ex.label for ex in exs], classifier.predict_all([ex.words for ex in exs], has_typos))


def print_evaluation(golds: List[int], predictions: List[int]):

    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    acc = float(num_correct) / num_total
    output_str = "Accuracy: %i / %i = %f" % (num_correct, num_total, acc)
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    output_str += ";\nPrecision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
    output_str += ";\nRecall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
    output_str += ";\nF1 (harmonic mean of precision and recall): %f;\n" % f1
    print(output_str)
    return acc, f1, output_str


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load train, dev, and test exs and index the words.
    train_exs = read_sentiment_examples(args.train_path)
    dev_exs = read_sentiment_examples(args.dev_path)
    test_exs = read_blind_sst_examples(args.blind_test_path)

    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples")

    word_embeddings = read_word_embeddings(args.word_vecs_path)

    # Train and evaluate
    start_time = time.time()
    if args.model == "DAN":
        model = train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings, args.use_typo_setting)
    else:
        model = TrivialSentimentClassifier()

    print("=====Train Accuracy=====")
    train_acc, train_f1, train_out = evaluate(model, train_exs, has_typos=False)
    print("=====Dev Accuracy=====")
    dev_acc, dev_f1, dev_out = evaluate(model, dev_exs, has_typos=False)
    train_eval_time = time.time() - start_time
    print("Time for training and evaluation: %.2f seconds" % train_eval_time)
    start_time = time.time()

    if args.use_typo_setting:
        print("=====Dev Misspelling Accuracy=====")
        dev_typo_exs = read_sentiment_examples("data/dev-typo.txt")
        dev_typo_acc, dev_typo_f1, dev_typo_out = evaluate(model, dev_typo_exs, has_typos=True)
        typo_eval_time = time.time() - start_time
        print("Time for typo evaluation: %.2f seconds" % typo_eval_time)
    else:
        dev_typo_acc, dev_typo_f1 = (-1, -1)
        dev_typo_out = "Typo setting not evaluated"

    # Write the test set output
    if args.run_on_test:
        test_exs_predicted = [SentimentExample(words, model.predict(words, has_typos=False)) for words in test_exs]
        write_sentiment_examples(test_exs_predicted, args.test_output_path)

    data = {'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_typo_acc': dev_typo_acc, 'dev_typo_f1': dev_typo_f1,
            'execution_time': train_eval_time, 'output': dev_out, 'typo_output': dev_typo_out}
    print("=====Results=====")
    print(json.dumps(data, indent=2))
