"""
Split German compound words
"""

__author__ = 'don.tuggener@gmail.com'

import sys

import ngram_probs  # trained with char_split_train.py


def split_compound(word: str):
    """
    Return list of possible splits, best first
    :param word: Word to be split
    :return: List of all splits
    """

    word = word.lower()

    # If there is a hyphen in the word, return part of the word behind the last hyphen
    if '-' in word:
        hyphen_index = word.rfind('-')
        return [(1., word[:hyphen_index].title(), word[hyphen_index + 1:].title())]

    scores = []  # Score for each possible split position
    # Iterate through characters, start at forth character, go to 3rd last
    for n in range(3, len(word) - 2):

        pre_slice = cut_off_fugen_s(word[:n])

        # Start, in, and end probabilities
        pre_slice_prob = ngram_probs.suffix.get(pre_slice, -1)
        in_slice_prob = []
        ngram = cut_off_fugen_s(word[n:])
        start_slice_prob = ngram_probs.prefix.get(ngram, -1)

        # Extract all ngrams
        for k in range(len(word), n+2, -1):
            # Probability of ngram in word, if high, split unlikely
            in_ngram = word[n:k]
            in_slice_prob.append(ngram_probs.infix.get(in_ngram, 1))  # Favor ngrams not occurring within words

        in_slice_prob = min(in_slice_prob)  # Lowest, punish splitting of good ingrams
        score = start_slice_prob - in_slice_prob + pre_slice_prob
        scores.append((score, word[:n].title(), word[n:].title()))

    if not scores:
        return [(0, word.title(), word.title())]
    scores.sort(reverse=True)
    return scores


def cut_off_fugen_s(word):
    if word.endswith('ts') or word.endswith('gs') or word.endswith('ks') \
            or word.endswith('hls') or word.endswith('ns'):
        if len(word[:- 1]) > 2:
            return word[:-1]
    return word


def germanet_evaluation(print_errors: bool = False):
    """ Test on GermaNet compounds from http://www.sfs.uni-tuebingen.de/lsd/compounds.shtml """
    cases, correct = 0, 0
    for line in open('split_compounds_from_GermaNet13.0.txt', 'r').readlines()[2:]:
        cases += 1
        sys.stderr.write('\r' + str(cases))
        sys.stderr.flush()
        line = line.strip().split('\t')
        if not len(line) == 3:
            continue  # A few corrupted lines
        split_result = split_compound(line[0])
        if split_result:
            if split_result[0][2] == line[2]:
                correct += 1
            elif print_errors:
                print(line, split_result)
        if cases % 10000 == 0: print(' Accuracy (' + str(correct) + '/' + str(cases) + '): ', 100 * correct / cases)
    print(' Accuracy (' + str(correct) + '/' + str(cases) + '): ', 100 * correct / cases)


if __name__ == '__main__':
    do_eval = False
    if do_eval:
        germanet_evaluation(print_errors=False)
    for x in split_compound(sys.argv[1]):
        print('\t'.join([str(y) for y in x]))