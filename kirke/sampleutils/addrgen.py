import logging
from typing import Dict, List, Optional, Tuple
from kirke.ebrules import addresses, addresses_v2
from kirke.utils import ebantdoc4, ebsentutils, strutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# loads address keywords
ALL_KEYWORDS = addresses.addr_keywords()


# pylint: disable=too-few-public-methods
class AddrContextGenerator:

    def __init__(self,
                 num_prev_words: int,
                 num_post_words: int,
                 candidate_type: str,
                 version: str = None) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.candidate_type = candidate_type
        if version is None:
            self.version = '1.0'
        else:
            self.version = version

    # pylint: disable=too-many-arguments, too-many-locals
    def get_candidates_from_text(self,
                                 nl_text: str,
                                 doc_lang: str,
                                 group_id: int = 0,
                                 # pylint: disable=line-too-long
                                 label_ant_list_param: Optional[List[ebsentutils.ProvisionAnnotation]] = None,
                                 label_list_param: Optional[List[bool]] = None,
                                 label: Optional[str] = None):
         # pylint: disable=line-too-long
        label_ant_list, label_list = [], []  # type: List[ebsentutils.ProvisionAnnotation], List[bool]
        if label_ant_list_param is not None:
            label_ant_list = label_ant_list_param
        if label_list_param is not None:
            label_list = label_list_param

        candidates = [] # type: List[Dict]
        group_id_list = [] # type: List[int]

        if not hasattr(self, 'version') or \
           self.version == '1.0':
            addr_se_cand_list = addresses.find_addresses(nl_text, ALL_KEYWORDS)
        elif self.version == '2.0':
            addr_se_cand_list = addresses_v2.find_addresses(nl_text)
        else:
            # default is use v2
            addr_se_cand_list = addresses_v2.find_addresses(nl_text)

        #finds all addresses in the text and adds window around each as a candidate
        for addr in addr_se_cand_list:
            addr_start, addr_end, addr_st = addr
            is_label = ebsentutils.check_start_end_overlap(addr_start,
                                                           addr_end,
                                                           label_ant_list)
            if doc_lang in set(['zh', 'ja']):
                prev_n_words, prev_spans = \
                    strutils.get_prev_n_chars_as_tokens(nl_text,
                                                        addr_start,
                                                        self.num_prev_words)
                post_n_words, post_spans = \
                    strutils.get_post_n_chars_as_tokens(nl_text,
                                                        addr_end,
                                                        self.num_post_words)
            else:
                prev_n_words, prev_spans = strutils.get_prev_n_clx_tokens(nl_text,
                                                                          addr_start,
                                                                          self.num_prev_words)
                post_n_words, post_spans = strutils.get_post_n_clx_tokens(nl_text,
                                                                          addr_end,
                                                                          self.num_post_words)
            new_bow = '{} {} {}'.format(' '.join(prev_n_words),
                                        addr_st,
                                        ' '.join(post_n_words))
            bow_start = addr_start
            bow_end = addr_end

            prev_15_words = ['PV15_' + wd for wd in prev_n_words[-15:]]
            post_15_words = ['PS15_' + wd for wd in post_n_words[:15]]
            prev_10_words = ['PV10_' + wd for wd in prev_n_words[-10:]]
            post_10_words = ['PS10_' + wd for wd in post_n_words[:10]]
            prev_5_words = ['PV5_' + wd for wd in prev_n_words[-5:]]
            post_5_words = ['PS5_' + wd for wd in post_n_words[:5]]
            prev_2_words = ['PV2_' + wd for wd in prev_n_words[-2:]]
            post_2_words = ['PS2_' + wd for wd in post_n_words[:2]]
            # to deal with n-gram of 2, added 'EOLN' to not mix
            # prev_4_words with others
            prev_n_words_plus = prev_n_words + ['EOLN'] + prev_15_words + ['EOLN'] + prev_10_words + ['EOLN'] + prev_5_words + ['EOLN'] + prev_2_words
            post_n_words_plus = post_n_words + ['EOLN'] + post_15_words + ['EOLN'] + post_10_words + ['EOLN'] + post_5_words + ['EOLN'] + post_2_words

            #update span based on window size
            if prev_spans:
                bow_start = prev_spans[0][0]
            if post_spans:
                bow_end = post_spans[-1][-1]

            a_candidate = {'candidate_type': self.candidate_type,
                           'bow_start': bow_start,
                           'bow_end': bow_end,
                           'text': new_bow,
                           'start': addr_start,
                           'end': addr_end,
                           'chars': addr_st,
                           'prev_n_words': ' '.join(prev_n_words_plus),
                           'post_n_words': ' '.join(post_n_words_plus),
                           'has_addr': True}
            candidates.append(a_candidate)

            #update group ids and label list
            group_id_list.append(group_id)
            if is_label:
                a_candidate['label_human'] = label
                label_list.append(True)
            else:
                label_list.append(False)
        return candidates, group_id_list, label_list

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: Optional[str] = None) \
                                -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc4
            label_list = []  # type: List[bool]
            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            #gets text based on document type
            if antdoc.doc_format in set([ebantdoc4.EbDocFormat.html,
                                         ebantdoc4.EbDocFormat.html_nodocstruct,
                                         ebantdoc4.EbDocFormat.other]):
                nl_text = antdoc.text
            else:
                nl_text = antdoc.get_nl_text()

            if group_id % 10 == 0:
                logger.debug('AddrContextGenerator.documents_to_candidates(), group_id = %d', group_id)

            candidates, group_id_list, label_list = self.get_candidates_from_text(nl_text,
                                                                                  doc_lang=antdoc.doc_lang,
                                                                                  group_id=group_id,
                                                                                  label_ant_list_param=label_ant_list,
                                                                                  label_list_param=label_list,
                                                                                  label=label)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
