
import nltk

from nltk.data import load

# Copied tom nltk.tokenizer.__init__.py to access span
# Standard sentence tokenizer.
# not tested
def sent_span_tokenize(text, language='english'):
    """
    Return a sentence-tokenized copy of *text*,
    using NLTK's recommended sentence tokenizer
    (currently :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    tokenizer = load('tokenizers/punkt/{0}.pickle'.format(language))
    return tokenizer.span_tokenize(text)

def word_tokenize(text):

    """
        tok_list = word_tokenize(text)
        print("token_list:")
        for tok in tok_list:
            if tok == '...':
                continue
            print("   '{}'".format(tok))
    """
    token_list = nltk.tokenize.word_tokenize(text)
    result = []
    for token in token_list:
        if token == 'd/b/a':
            result.append(token)
        elif '_____' in token:
            result.append('_____')
        elif len(token) > 1 and '/' in token:
            result.extend(token.split('/'))
        else:
            result.append(token)
    return result

    
"""
PUNKT_TOKENIZER = nltk.tokenize.load('tokenizers/punkt/english.pickle')

def sent_span_tokenize(text):
    span_list = PUNKT_TOKENIZER.span_tokenize(text)
    return [(span[0], span[1], text[span[0] : span[1]]) for span in span_list]
"""

"""
def sent_span_tokenize(text):
    span_list = sent_tokenize(text)
    return [(span[0], span[1], text[span[0] : span[1]]) for span in span_list]

this doesn't work
def word_span_tokenize(text):
    span_list = word_tokenize(text)
    return [(span[0], span[1], text[span[0] : span[1]]) for span in span_list]
"""
