import logging
import re
# import time
from typing import List, Tuple

from sklearn.externals import joblib

from kirke.ebrules.addrclassifier import KEYWORDS, LogRegModel
from kirke.utils import strutils


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_DEBUG_ADDRESS = False

DATA_DIR = './dict/addresses/'
NUM_DIGIT_CHUNKS = 10
MIN_ADDRESS_LEN = 5
MAX_ADDRESS_LEN = 100

US_ZIP = r'\d{5}(-\d{4})?'
UK_STD = r'[A-Z]{1,2}[0-9ROI][0-9A-Z]?\s*(?:(?![CIKMOV])[A-Z]?[0-9O][a-zA-Z]{2})'
CAN_STD = r'[A-Z][0-9][A-Z]\s+[0-9][A-Z][0-9]'
ZIP_CODE_YEAR = re.compile(r'\b\d{4,5}\b' + r'|\b{}\b'.format(UK_STD))

LOCS = KEYWORDS['uk'] + KEYWORDS['us'] + KEYWORDS['can'] + \
       KEYWORDS['country_names'] + ['London', 'LONDON']
LOCS = sorted(LOCS, reverse=True)
LOCS = '|'.join(LOCS)

# pylint: disable=too-many-locals
def find_addresses(text: str) -> List[Tuple[int, int, str]]:
    # finditer ( number anything (loc)+ zip )
    # pylint: disable=line-too-long
    matches = re.finditer(r'(?=(\b(\d+|P\.? ?[0O]\.?|One) +(.{{,80}}?\b({})\b)+?(.{{,80}}?(\b{}\b|\b{}\b|\b{}\b))?))'.format(LOCS, US_ZIP, UK_STD, CAN_STD), text, re.DOTALL)
    all_spans = [match.span(1) for match in matches]
    addr_se_st_list = []  # type: List[Tuple[int, int, str]]
    prev_start, prev_end, prev_prob = 0, 0, 0.0

    # start_time = time.time()
    # pylint: disable=too-many-nested-blocks
    for ad_start, ad_end in all_spans:
        addr_st = text[ad_start:ad_end]
        if len(addr_st.split()) < 25:
            address_prob = classify(addr_st)
            if IS_DEBUG_ADDRESS:
                print('  classify {} ({})'.format(address_prob, addr_st))
            if address_prob >= 0.5:
                # if they overlap
                if ad_start > prev_start and ad_start < prev_end:
                    # replace the previous one if the new prob is higher
                    if address_prob >= prev_prob:
                        if addr_se_st_list:
                            addr_se_st_list.pop()
                        addr_se_st_list.append((ad_start, ad_end, addr_st))
                        prev_start, prev_end, prev_prob = ad_start, ad_end, address_prob
                else:
                    addr_se_st_list.append((ad_start, ad_end, addr_st))
                    prev_start, prev_end, prev_prob = ad_start, ad_end, address_prob
    # end_time = time.time()
    # print('addrclassifier.extract_features(%d addr_st) took %.0f msec\n' %
    #       (len(all_spans), (end_time - start_time) * 1000))
    return addr_se_st_list

ADDR_CLASSIFIER = None

def load_address_classifier() -> LogRegModel:
    """Load and prepare the classifier"""
    ADDR_MODEL_FILE_NAME = DATA_DIR + 'addr_classifier.pkl'
    logger.info('loading ADDR_CLASSIFIER(%s)', ADDR_MODEL_FILE_NAME)
    return joblib.load(ADDR_MODEL_FILE_NAME)


def classify(line: str) -> float:
    # pylint: disable=global-statement
    global ADDR_CLASSIFIER

    """Returns probability that line is an addr"""
    len_s = len(line)
    if (len_s <= MIN_ADDRESS_LEN or len_s >= MAX_ADDRESS_LEN + 1) or \
       not strutils.has_alpha(line):
        return 0
    if ADDR_CLASSIFIER is None:
        ADDR_CLASSIFIER = load_address_classifier()
    probs, unused_label = ADDR_CLASSIFIER.predict(line)

    return probs[1]
