import sys
import copy
from collections import namedtuple


# pylint: disable=R0903
class EbToken:
    __slots__ = ['start', 'end', 'word', 'lemma', 'pos', 'index', 'ner']

    # pylint: disable=R0913
    def __init__(self, start, end, word, lemma, pos, index, ner):
        self.start = start
        self.end = end
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.index = index
        self.ner = ner

    def __str__(self):
        return 'EbToken({}, {}, {}, {}, {}, {})'.format(self.word, self.index, self.pos, self.ner, self.start, self.end)

# 13,772 bytes avg per ebsent
# 1.8 Mb in an ebantdoc

# cannot use namedtuple because _fix_incorrect_tokens resets internal
# in ebtext2antdoc
#EbToken = namedtuple('EbToken', ['start', 'end', 'word',
#                                 'lemma', 'pos', 'index', 'ner'])
# 9,927 bytes avg per ebsent
# 1.3 Mb in an ebantdoc



# TODO, In future, without arff, probably don't need this
# Digits might have commas in tokens, "23,000"
# In addition, for "December 17, 2012", there is a comma token also.
# These are removed now.
def eb_tokens_to_st(eb_token_list):
    st_list = []
    for eb_token in eb_token_list:
        word = eb_token.word
        if ',' in word:
            # print("word_with_comma: '{}'".format(word), file=sys.stderr)
            # print("json_tokens has comma: '{}'".format(json_token_list), file=sys.stderr)
            replace_word = word.replace(',', '')
            if replace_word:
                st_list.append(replace_word)
        else:
            st_list.append(word)
    return ' '.join(st_list)


def eb_tokens_to_lemma_st(eb_token_list):
    st_list = []
    for eb_token in eb_token_list:
        word = eb_token.lemma
        if ',' in word:
            # print("word_with_comma: '{}'".format(word), file=sys.stderr)
            # print("json_tokens has comma: '{}'".format(json_token_list), file=sys.stderr)
            replace_word = word.replace(',', '')
            if replace_word:
                st_list.append(replace_word)
        else:
            st_list.append(word)
    return ' '.join(st_list)


# because of the way num_refix_space is used, need to isolate such effect
# here.
def to_eb_tokens(token_list, num_prefix_space):
    result = []
    for token in token_list:
        eb_token = EbToken(start=token['characterOffsetBegin'] + num_prefix_space,
                           end=token['characterOffsetEnd'] + num_prefix_space,
                           word=sys.intern(token['word']),
                           lemma=sys.intern(token['lemma']),
                           pos=sys.intern(token['pos']),
                           index=token['index'],
                           ner=sys.intern(token['ner']))
        result.append(eb_token)
    return result


def merge_ebsents(ebsent_list):
    ebsent0 = ebsent_list[0]

    token_list = []
    text_list = []
    tokens_text_list = []
    entity_list = []
    labels = set([])

    min_start = ebsent0.start
    max_end = ebsent0.end
    for ebsent in ebsent_list:
        token_list.extend(ebsent.tokens)
        text_list.append(ebsent.text)
        tokens_text_list.append(ebsent.tokens_text)
        entity_list.extend(ebsent.entities)
        labels |= ebsent.labels
        if ebsent.start < min_start:
            min_start = ebsent.start
        if ebsent.end > max_end:
            max_end = ebsent.end

    # make a shallow copy to store results
    merged_ebsent = copy.copy(ebsent0)
    # merged_ebsent.file_id is copied by shallow copy
    merged_ebsent.tokens = token_list
    merged_ebsent.start = min_start
    merged_ebsent.end = max_end
    # merged_ebsent.text = '  '.join(text_list)
    # merged_ebsent.tokens_text = '  '.join(tokens_text_list)
    merged_ebsent.entities = entity_list
    merged_ebsent.labels = labels

    return merged_ebsent


# pylint: disable=R0902
class EbSentence:
    __slots__ = ['file_id', 'tokens', 'start', 'end',
                'entities', 'labels']

    def __init__(self, file_id, json_sent, atext, num_prefix_space):
        tokens = json_sent['tokens']
        self.file_id = file_id
        self.tokens = to_eb_tokens(tokens, num_prefix_space)
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end
        # self.text = atext[self.start:self.end]  # migh have page number
        # self.tokens_text = eb_tokens_to_st(self.tokens)          # no page number
        # entities are EbEntity's, not set until populate_ebsent_entities(ebsent),
        # after fix_ner_tags()
        self.entities = []
        # set of strings
        self.labels = set([])

    def extend_tokens(self, tokens, atext):
        self.tokens.extend(tokens)
        self.end = self.tokens[-1].end
        # self.text = atext[self.start:self.end]  # migh have page number
        # self.tokens_text = eb_tokens_to_st(self.tokens)          # no page number
        # TODO, jshaw, question
        # when extending tokens, should modify self.entities also?

    def __str__(self):
        return 'EbSentence({}, {})'.format(self.start, self.end)

    def get_tokens(self):
        return self.tokens

    # sometimes we replace incorrectly tagged entities, such 'Lessee' as location
    def set_tokens(self, tokens):
        self.tokens = tokens

    def get_number_tokens(self):
        return len(self.tokens)

    # this will translate all () -> -lrb- -rrb-, ' -> `` or \'\'
    # no page number
    def get_tokens_text(self):
        return eb_tokens_to_st(self.tokens)

    def get_lemma_text(self):
        return eb_tokens_to_lemma_st(self.tokens)              

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

#    def get_text(self):
#        return self.text

    def get_number_chars(self):
        # return len(self.text)
        return self.end - self.start

    def set_entities(self, entity_list):
        self.entities = entity_list

    def get_entities(self):
        return self.entities

    def get_labels(self):
        return self.labels

    def set_labels(self, labels):
        self.labels = set(labels)