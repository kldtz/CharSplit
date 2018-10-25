#!/usr/bin/env python3

import random

from kirke.ebrules.addrclassifier import LogRegModel

DATA_DIR = './dict/addresses/'

# read, shuffle, and separate training and dev data
def parse_train(fname):
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []

    with open(fname) as train:
        train_list = train.readlines()
        cutoff = int(len(train_list) * 0.9)
        random.Random(0).shuffle(train_list)
        train = train_list[:cutoff]
        dev = train_list[cutoff:]

        for line in train:
            data, label = line.split("\t")
            train_data.append(data)
            train_labels.append(int(label.strip()))
        for line in dev:
            data, label = line.split("\t")
            dev_data.append(data)
            dev_labels.append(int(label.strip()))

    return train_data, train_labels, dev_data, dev_labels

# pylint: disable=too-many-locals
def main():
    model = LogRegModel()
    training_file_name = DATA_DIR+"addr_annots.tsv"
    print('reading training file: {}'.format(training_file_name))
    train_data, train_labels, dev_data, dev_labels = parse_train(training_file_name)
    model.fit_model(train_data, train_labels)

    # pylint: disable=invalid-name
    tp, fp, fn = 0, 0, 0
    fps = []
    fns = []
    for i, addr in enumerate(dev_data):
        gold_label = dev_labels[i]
        probs, pred_label = model.predict(addr)
        if gold_label == 1 and pred_label == 1:
            tp += 1
        elif gold_label == 1 and pred_label == 0:
            fns.append([addr, probs[1]])
            fn += 1
        elif gold_label == 0 and pred_label == 1:
            fps.append([addr, probs[1]])
            fp += 1
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    # pylint: disable=invalid-name
    f1 = 2 * ((prec * rec)/(prec+rec))
    print("TP = {}, FP = {}, FN = {}".format(tp, fp, fn))
    print("P = {}, R = {}, F = {}".format(prec, rec, f1))

    model_file_name = DATA_DIR + 'addr_classifier.pkl'
    model.save_model_file(model_file_name)


if __name__ == '__main__':
    main()
