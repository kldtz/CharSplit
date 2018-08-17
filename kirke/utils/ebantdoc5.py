# pylint: disable=too-many-lines
import array
from array import ArrayType
import concurrent.futures
from enum import Enum
import json
import logging
import os
import re
import shutil
import sys
import time
# pylint: disable=unused-import
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

# pylint: disable=import-error
from sklearn.externals import joblib

from kirke.abbyyxml import abbyyxmlparser, abbyypbox_syncher, tableutils
from kirke.abbyyxml.pdfoffsets import AbbyyBlock, AbbyyTableBlock, AbbyyXmlDoc
from kirke.eblearn import ebattrvec, sent2ebattrvec
from kirke.docstruct import docutils, fromtomapper, htmltxtparser, linepos, pdftxtparser
from kirke.docstruct.pdfoffsets import PDFTextDoc
from kirke.utils import corenlputils, ebsentutils, memutils, osutils, strutils, txtreader
from kirke.utils.textoffset import TextCpointCunitMapper
from kirke.utils.ebsentutils import ProvisionAnnotation

from kirke.docstruct import paraattrsutils


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

IS_USE_ABBYY_FOR_PARAGRAPH_INFO = False

# if we use pdfbox's text or abbyy's text
CORENLP_JSON_VERSION = '1.6'
EBANTDOC_VERSION = '1.9'

if IS_USE_ABBYY_FOR_PARAGRAPH_INFO:
    CORENLP_JSON_VERSION = '1.7'
    EBANTDOC_VERSION = '1.10'


def get_corenlp_json_fname(txt_basename, work_dir):
    base_fn = txt_basename.replace('.txt',
                                   '.corenlp.v{}.json'.format(CORENLP_JSON_VERSION))
    return '{}/{}'.format(work_dir, base_fn)


def get_nlp_fname(txt_basename, work_dir):
    base_fn = txt_basename.replace('.txt', '.nlp.v{}.txt'.format(CORENLP_JSON_VERSION))
    return '{}/{}'.format(work_dir, base_fn)


def get_ebant_fname(txt_basename, work_dir):
    base_fn = txt_basename.replace('.txt',
                                   '.ebantdoc.v{}.pkl'.format(EBANTDOC_VERSION))
    return '{}/{}'.format(work_dir, base_fn)


class EbDocFormat(Enum):
    html_nodocstruct = 1
    html = 2
    pdf = 3
    other = 4


EB_NUMERIC_FILE_ID_PAT = re.compile(r'\-(\d+)\.txt')

# pylint: disable=too-many-instance-attributes
class EbAnnotatedDoc:

    # pylint: disable=too-many-locals
    def __init__(self,
                 *,   # force specify all parameters by keyword
                 file_name: str,
                 doc_format: EbDocFormat,
                 text: str,
                 cpoint_cunit_mapper: TextCpointCunitMapper,
                 prov_ant_list: List[ProvisionAnnotation],
                 is_test: bool,
                 # para_doc_text,      # adjusted
                 para_prov_ant_list,   # nlp_offset adjusted
                 attrvec_list: List[ebattrvec.EbAttrVec],         # nlp offset adjusted
                 sechead_list: List[Tuple[int, int, str, str, int]],
                 paras_with_attrs,     # nlp_offset adjusted
                 origin_lnpos_list: List[linepos.LnPos],
                 nlp_lnpos_list: List[linepos.LnPos],
                 gap_span_list,  # TODO, jshaw, maybe del in future
                 linebreak_arr: ArrayType,
                 page_offsets_list=None,
                 para_not_linebreak_arr: ArrayType,
                 doc_lang: str = 'en') \
                 -> None:
        self.version = 'v5.0'
        self.file_id = file_name
        self.doc_format = doc_format
        self.text = text
        self.len_text = len(text)   # used to check out of bound

        self.codepoint_to_cunit_mapper = cpoint_cunit_mapper

        self.prov_annotation_list = prov_ant_list
        self.is_test_set = is_test
        self.provision_set = set([prov_ant.label for prov_ant in prov_ant_list])
        self.doc_lang = doc_lang

        self.para_prov_ant_list = para_prov_ant_list
        self.attrvec_list = attrvec_list
        self.sechead_list = sechead_list
        self.paras_with_attrs = paras_with_attrs
        # para_indices and para_attrs should be the same length
        self.para_indices = [x[0] for x in paras_with_attrs]
        self.para_attrs = [x[1] for x in paras_with_attrs]
        # to map to original offsets
        self.nlp_lnpos_list = nlp_lnpos_list
        self.origin_lnpos_list = origin_lnpos_list
        self.gap_span_list = gap_span_list

        self.linebreak_arr = linebreak_arr
        self.para_not_linebreak_arr = para_not_linebreak_arr
        self.page_offsets_list = page_offsets_list

        self.table_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.chart_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.toc_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.pagenum_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.footer_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.signature_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]

        # abbyy's stuff
        self.abbyy_table_list = []  # type: List[AbbyyTableBlock]
        self.abbyy_signature_list = []  # type: List[AbbyyBlock]

    def get_file_id(self):
        return self.file_id

    def get_document_id(self) -> str:
        """Return the identifier of the document, in string"""
        mat = EB_NUMERIC_FILE_ID_PAT.search(self.file_id)
        # must match
        if mat:
            return mat.group(1)
        return 'no-doc-id-found:{}'.format(self.file_id)

    def set_provision_annotations(self, ant_list: List[ProvisionAnnotation]) -> None:
        self.prov_annotation_list = ant_list

    def get_provision_annotations(self,
                                  provision: Optional[str] = None) \
                                  -> List[ProvisionAnnotation]:
        if provision:
            return [prov_ant for prov_ant in self.prov_annotation_list
                    if prov_ant.label == provision]
        return self.prov_annotation_list

    def has_provision_ant(self, provision: str) -> bool:
        for prov_annotation in self.prov_annotation_list:
            if prov_annotation.label == provision:
                return True
        return False

    def get_provision_set(self) -> Set[str]:
        return self.provision_set

    def get_attrvec_list(self) -> List[ebattrvec.EbAttrVec]:
        return self.attrvec_list

    def get_text(self, text_type: str = 'text') -> str:
        if text_type == 'text':
            return self.text
        elif text_type == 'nlp_text':
            return self.get_nlp_text()
        return self.text

    def get_nl_text(self) -> str:
        ch_list = list(self.text)
        for offset in self.linebreak_arr:
            ch_list[offset] = '\n'
        return ''.join(ch_list)

    def get_nlp_text(self) -> str:
        doc_text = self.text
        if not self.paras_with_attrs:  # html or html_no_docstruct
            return doc_text

        para_st_list = []
        skip_count = 0
        for para_with_attrs in self.paras_with_attrs:

            lnpos_pair_list, unused_attrs = para_with_attrs
            skip_st_list = []
            for from_lnpos, unused_to_lnpos in lnpos_pair_list:
                from_start, from_end, unused_from_line_num, is_gap = from_lnpos.to_tuple()
                if not is_gap:
                    para_st_list.append(doc_text[from_start:from_end])
                else:  # this is never called, or reached
                    skip_count += 1
                    skip_st_list.append("skipping #{} [{}]".format(skip_count,
                                                                   doc_text[from_start:from_end]))
        nlp_text = '\n'.join(para_st_list)
        return nlp_text

    def get_paraline_text(self) -> str:
        ch_list = list(self.get_nl_text())
        for offset in self.para_not_linebreak_arr:
            ch_list[offset] = ' '
        return ''.join(ch_list)

    def get_nlp_sx_lnpos_list(self):
        return [(elt.start, elt) for elt in self.nlp_lnpos_list]

    def get_origin_sx_lnpos_list(self):
        return [(elt.start, elt) for elt in self.origin_lnpos_list]

    def has_same_prov_ant_list(self, prov_ant_list2: List[ProvisionAnnotation]) -> bool:
        return self.prov_annotation_list == prov_ant_list2

    def get_doc_format(self) -> EbDocFormat:
        return self.doc_format


def remove_prov_greater_offset(prov_annotation_list, max_offset):
    return [prov_ant for prov_ant in prov_annotation_list
            if prov_ant.start < max_offset]


def load_cached_ebantdoc(eb_antdoc_fn: str) -> Optional[EbAnnotatedDoc]:
    """Load from pickled file if file exist, otherwise None"""

    # if cache version exists, load that and return
    if os.path.exists(eb_antdoc_fn):
        start_time = time.time()
        try:
            eb_antdoc = joblib.load(eb_antdoc_fn)
            end_time = time.time()
            logger.info("loading from cache: %s, took %.0f msec",
                        eb_antdoc_fn, (end_time - start_time) * 1000)

            return eb_antdoc
        # pylint: disable=broad-except
        except Exception:  # if failed to load cache using joblib.load()
            logger.warning("Detected an issue calling load_cached_ebantdoc(%s).  Skip cache.",
                           eb_antdoc_fn)
            return None

    return None


# return attrvec_list, nlp_prov_ant_list, original_sx_lnpos_list, nlp_sx_lnpos_list
# pylint: disable=too-many-arguments, too-many-locals
def nlptxt_to_attrvec_list(para_doc_text: str,
                           txt_file_name: str,
                           txt_base_fname: str,
                           prov_annotation_list,
                           paras_with_attrs: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
                                                        # str,
                                                        List[Any]]],
                           work_dir: str,
                           is_cache_enabled: bool,
                           doc_lang: str = 'en',
                           is_use_corenlp: bool = True) \
                           -> Tuple[List[ebattrvec.EbAttrVec],
                                    List[ebsentutils.ProvisionAnnotation],
                                    List[linepos.LnPos],
                                    List[linepos.LnPos]]:

    # At this point, put all document structure information into
    # ebsent_list
    # We also adjust all annotations from CoreNlp into the offsets from original
    # document.  Offsets is no NLP-based.
    # from_list_xx, to_list_xx = fromtomapper.paras_to_fromto_lists(paras_with_attrs)

    if paras_with_attrs:
        fromto_mapper = fromtomapper.paras_to_fromto_mapper_sorted_by_from(paras_with_attrs)

        nlp_prov_ant_list = []
        for prov_annotation in prov_annotation_list:
            orig_start, orig_end = prov_annotation.start, prov_annotation.end
            orig_label = prov_annotation.label

            xstart, xend = fromto_mapper.get_se_offsets(orig_start, orig_end)

            nlp_prov_ant_list.append(ebsentutils.ProvisionAnnotation(xstart, xend, orig_label))
    else:
        # fromto_mapper = None
        nlp_prov_ant_list = prov_annotation_list


    # let's adjust the offsets in prov_annotation to keep things simple and
    # maximize reuse of existing code.
    if not is_use_corenlp:  # this is for some analyzer in candidate-gen-pipeline
        attrvec_list = []  # type: List[ebattrvec.EbAttrVec]
    else:  # all sentence pipeline use this
        # para_doc_text is what is sent, not txt_base_fname
        corenlp_json = text_to_corenlp_json(para_doc_text,
                                            txt_base_fname,
                                            work_dir=work_dir,
                                            is_cache_enabled=is_cache_enabled,
                                            doc_lang=doc_lang)

        ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name,
                                                               corenlp_json,
                                                               para_doc_text,
                                                               is_doc_structure=True)

        if paras_with_attrs:
            # still haven't add the sechead info back into
            ebsentutils.update_ebsents_with_sechead(ebsent_list, paras_with_attrs)

        # fix any domain specific entity extraction, such as 'Lessee' as a location
        # this is a in-place replacement
        # We only handle up to "exhibit_appendix,exhibit_appendix_complete"
        for ebsent in ebsent_list:
            ebsentutils.fix_ner_tags(ebsent)
            ebsentutils.populate_ebsent_entities(ebsent,
                                                 para_doc_text[ebsent.start:ebsent.end],
                                                 lang=doc_lang)

            overlap_provisions = (ebsentutils.get_labels_if_start_end_overlap(ebsent.start,
                                                                              ebsent.end,
                                                                              nlp_prov_ant_list)
                                  if nlp_prov_ant_list else [])
            ebsent.set_labels(overlap_provisions)

        attrvec_list = []
        num_sent = len(ebsent_list)
        # we need prev and next sentences because such information are used in the
        # feature extraction
        prev_ebsent, next_ebsent = None, None
        for sent_idx, ebsent in enumerate(ebsent_list):
            if sent_idx != num_sent-1:
                next_ebsent = ebsent_list[sent_idx + 1]
            else:
                next_ebsent = None
            fvec = sent2ebattrvec.sent2ebattrvec(ebsent,
                                                 sent_idx + 1,
                                                 prev_ebsent,
                                                 next_ebsent,
                                                 para_doc_text)
            attrvec_list.append(fvec)
            prev_ebsent = ebsent

    # these lists has to be sorted by nlp_sx_lnpos_list
    original_lnpos_list, nlp_lnpos_list = fromtomapper.paras_to_fromto_lnpos_lists(paras_with_attrs)

    return attrvec_list, nlp_prov_ant_list, original_lnpos_list, nlp_lnpos_list


# stop at 'exhibit_appendix' or 'exhibit_appendix_complete'
# pylint: disable=too-many-locals
def html_no_docstruct_to_ebantdoc(txt_file_name,
                                  work_dir,
                                  is_cache_enabled=True,
                                  doc_lang="en",
                                  is_use_corenlp: bool = True) -> EbAnnotatedDoc:
    logging.debug('html_to_ebantdoc(%s)', txt_file_name)
    debug_mode = False
    start_time = time.time()
    txt_base_fname = os.path.basename(txt_file_name)

    txt_file_name, doc_text, prov_annotation_list, is_test, cpoint_cunit_mapper = \
        chop_at_exhibit_complete(txt_file_name, txt_base_fname, work_dir, debug_mode)
    paras_with_attrs = []  # type: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]], List[Any]]]
    attrvec_list, nlp_prov_ant_list, _, _ = nlptxt_to_attrvec_list(doc_text,
                                                                   txt_file_name,
                                                                   txt_base_fname,
                                                                   prov_annotation_list,
                                                                   paras_with_attrs,
                                                                   work_dir,
                                                                   is_cache_enabled,
                                                                   doc_lang=doc_lang,
                                                                   is_use_corenlp=is_use_corenlp)

    nlp_prov_ant_list = prov_annotation_list
    origin_lnpos_list = []  # type: List[linepos.LnPos]
    nlp_lnpos_list = []  # type: List[linepos.LnPos]
    gap_span_list = []  # type: List[Tuple[int, int]]
    eb_antdoc = EbAnnotatedDoc(file_name=txt_file_name,
                               doc_format=EbDocFormat.html_nodocstruct,
                               text=doc_text,
                               cpoint_cunit_mapper=cpoint_cunit_mapper,
                               prov_ant_list=prov_annotation_list,
                               is_test=is_test,
                               para_prov_ant_list=nlp_prov_ant_list,
                               attrvec_list=attrvec_list,
                               # TODO, jshaw
                               # Maybe still add sechead info for .txt files.
                               # no sechead information for txt file
                               sechead_list=[],
                               paras_with_attrs=paras_with_attrs,
                               origin_lnpos_list=origin_lnpos_list,
                               nlp_lnpos_list=nlp_lnpos_list,
                               gap_span_list=gap_span_list,
                               linebreak_arr=array.array('i'),
                               para_not_linebreak_arr=array.array('i'),
                               doc_lang=doc_lang)

    # eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)

    # We don't want to cache a document that's not complete.
    # It must be 'is_cache_enabled', 'is_doc_structure', 'is_use_corenlp'.
    # html_no_docstruct_to_ebantdoc5 has 'is_doc_structure=False, so no cache.
    #
    # if txt_file_name and is_cache_enabled and is_use_corenlp:
    #     start_time = time.time()
    #     joblib.dump(eb_antdoc, eb_antdoc_fn)
    #     end_time = time.time()
    #     logger.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
    #                 eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time = time.time()
    logger.info("html_no_docstruct_to_ebantdoc: %s, took %.0f msec; %d attrvecs",
                'not_saving_html_no_docstruct_to_ebandoc',
                (end_time - start_time) * 1000, len(attrvec_list))
    return eb_antdoc


# return txt_file_name, doc_text, prov_annotation_list, is_test, cpoint_cunit_mapper
def chop_at_exhibit_complete(txt_file_name: str,
                             txt_base_fname: str,
                             work_dir: str,
                             debug_mode: bool = False) \
                             -> Tuple[str, str, List[ProvisionAnnotation], bool,
                                      TextCpointCunitMapper]:
    doc_text = txtreader.loads(txt_file_name)
    # sub single newlines for spaces to preserve paragraphs in text documents
    doc_text = re.sub('(?<![\r\n])(\n)(?! *[\r\n])', ' ', doc_text)
    cpoint_cunit_mapper = TextCpointCunitMapper(doc_text)
    prov_annotation_list, is_test = ebsentutils.load_prov_annotation_list(txt_file_name,
                                                                          cpoint_cunit_mapper)
    max_txt_size = len(doc_text)
    is_chopped = False
    for prov_ant in prov_annotation_list:
        if prov_ant.label in set(['exhibit_appendix', 'exhibit_appendix_complete']):
            if prov_ant.start < max_txt_size:
                max_txt_size = prov_ant.start
                is_chopped = True
    if is_chopped:
        doc_text = doc_text[:max_txt_size]
        prov_annotation_list = remove_prov_greater_offset(prov_annotation_list, max_txt_size)


    # perform document structuring all all text file
    # First perform document structuring
    #   The new text file will have section header in separate lines, plus
    #   no page number.
    # Then, go through CoreNlp with the new text file
    # Adjusted the offsets in the annotation from corenlp

    # write the shortened files
    # if file is not shortened, write to dir-work
    txt_file_name = '{}/{}'.format(work_dir, txt_base_fname)
    txtreader.dumps(doc_text, txt_file_name)
    if debug_mode:
        print('wrote {}'.format(txt_file_name, file=sys.stderr))

    return txt_file_name, doc_text, prov_annotation_list, is_test, cpoint_cunit_mapper


# stop at 'exhibit_appendix' or 'exhibit_appendix_complete'
def html_to_ebantdoc(txt_file_name: str,
                     work_dir: str,
                     is_cache_enabled: bool = True,
                     doc_lang: str = 'en',
                     is_use_corenlp: bool = True) -> EbAnnotatedDoc:
    logging.debug('html_to_ebantdoc(%s)', txt_file_name)
    debug_mode = False
    start_time1 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)

    txt_file_name, doc_text, prov_annotation_list, is_test, cpoint_cunit_mapper = \
        chop_at_exhibit_complete(txt_file_name, txt_base_fname, work_dir, debug_mode)
    paras_with_attrs, para_doc_text, gap_span_list, unused_orig_doc_txt, sechead_list = \
        htmltxtparser.parse_document(txt_file_name,
                                     work_dir=work_dir,
                                     is_combine_line=True)

    txt4nlp_fname = get_nlp_fname(txt_base_fname, work_dir)
    txtreader.dumps(para_doc_text, txt4nlp_fname)
    if debug_mode:
        print("wrote {}".format(txt4nlp_fname), file=sys.stderr)

    attrvec_list, nlp_prov_ant_list, origin_lnpos_list, nlp_lnpos_list = \
        nlptxt_to_attrvec_list(para_doc_text,
                               txt_file_name,
                               txt_base_fname,
                               prov_annotation_list,
                               paras_with_attrs,
                               work_dir,
                               is_cache_enabled,
                               doc_lang=doc_lang,
                               is_use_corenlp=is_use_corenlp)

    eb_antdoc = EbAnnotatedDoc(file_name=txt_file_name,
                               doc_format=EbDocFormat.html,
                               text=doc_text,
                               cpoint_cunit_mapper=cpoint_cunit_mapper,
                               prov_ant_list=prov_annotation_list,
                               is_test=is_test,
                               para_prov_ant_list=nlp_prov_ant_list,
                               attrvec_list=attrvec_list,
                               sechead_list=sechead_list,
                               paras_with_attrs=paras_with_attrs,
                               origin_lnpos_list=origin_lnpos_list,
                               nlp_lnpos_list=nlp_lnpos_list,
                               gap_span_list=gap_span_list,
                               # there is no page_offsets_list
                               linebreak_arr=array.array('i'),
                               para_not_linebreak_arr=array.array('i'),
                               doc_lang=doc_lang)

    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled and is_use_corenlp:
        t2_start_time = time.time()
        # tmpFileName = tempfile.NamedTemporaryFile(dir=KIRKE_TMP_DIR, delete=False)
        # joblib.dump(eb_antdoc, tmpFileName.name)
        # shutil.move(tmpFileName.name, eb_antdoc_fn)
        osutils.joblib_atomic_dump(eb_antdoc, eb_antdoc_fn)
        t2_end_time = time.time()
        if (t2_end_time - t2_start_time) * 1000 > 30000:
            logger.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
                        eb_antdoc_fn, len(attrvec_list), (t2_end_time - t2_start_time) * 1000)

    end_time1 = time.time()
    logger.info("html_to_ebantdoc: %s, took %.0f msec; %d attrvecs",
                eb_antdoc_fn, (end_time1 - start_time1) * 1000, len(attrvec_list))
    return eb_antdoc


def update_special_block_info(eb_antdoc, pdf_txt_doc):
    eb_antdoc.table_list = pdf_txt_doc.special_blocks_map.get('table', [])
    eb_antdoc.chart_list = pdf_txt_doc.special_blocks_map.get('chart', [])
    eb_antdoc.signature_list = pdf_txt_doc.special_blocks_map.get('signature', [])
    eb_antdoc.toc_list = pdf_txt_doc.special_blocks_map.get('toc', [])
    eb_antdoc.pagenum_list = pdf_txt_doc.special_blocks_map.get('pagenum', [])
    eb_antdoc.footer_list = pdf_txt_doc.special_blocks_map.get('footer', [])


# this parses both originally text and html documents
# It's main goal is to detect sechead
# optionally pagenum, footer, toc, signature
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def pdf_to_ebantdoc(txt_file_name: str,
                    offsets_file_name: str,
                    pdfxml_file_name: str,
                    work_dir: str,
                    is_cache_enabled: bool = True,
                    doc_lang: str = 'en',
                    is_use_corenlp: bool = True) -> EbAnnotatedDoc:
    logging.debug('pdf_to_ebantdoc(%s)', txt_file_name)
    debug_mode = False
    start_time0 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)
    offsets_base_fname = os.path.basename(offsets_file_name)
    pdfxml_base_fname = ''
    if os.path.exists(pdfxml_file_name):
        pdfxml_base_fname = os.path.basename(pdfxml_file_name)

    # PDF files are mostly used by our users, not for training and test.
    # Chopping text at exhibit_complete messes up all the offsets info from offsets.json.
    # To avoid this situation, we currently do NOT chop PDF text.  For HTML and others,
    # such chopping is not an issue since there is associated no offsets.json file.
    # A correct solution is to not chop, but put a market
    # in the ebantdoc and all the code acts accordingly, such as not to run corenlp on such
    # text because such chopped text should not participate in training and evaluation of
    # ML classification, or any classification.
    ## txt_file_name, _, prov_annotation_list, is_test = \
    ##    chop_at_exhibit_complete(txt_file_name, txt_base_fname, work_dir, debug_mode)

    # copy txt file to work/txt_base_name, to be consistent with html_to_ebantdoc()
    if txt_file_name != '{}/{}'.format(work_dir, txt_base_fname):
        shutil.copy2(txt_file_name, '{}/{}'.format(work_dir, txt_base_fname))
        shutil.copy2(offsets_file_name, '{}/{}'.format(work_dir, offsets_base_fname))
        if pdfxml_base_fname:
            shutil.copy2(pdfxml_file_name, '{}/{}'.format(work_dir, pdfxml_base_fname))

    doc_text, unused_nl_text, linebreak_arr, \
        unused_paraline_text, para_not_linebreak_arr, cpoint_cunit_mapper = \
            pdftxtparser.to_nl_paraline_texts(txt_file_name, offsets_file_name, work_dir=work_dir)

    prov_annotation_list, is_test = ebsentutils.load_prov_annotation_list(txt_file_name,
                                                                          cpoint_cunit_mapper)

    pdf_txt_doc = pdftxtparser.parse_document(txt_file_name, work_dir=work_dir)  # type: PDFTextDoc

    xml_fname = txt_file_name.replace('.txt', '.pdf.xml')
    # For test documents, there is no new .pdf.xml file available.
    # In this case, only use the information available from PDFBox.
    abbyy_xml_doc = None  # type: Optional[AbbyyXmlDoc]
    if os.path.exists(xml_fname):
        abbyy_xml_doc = abbyyxmlparser.parse_document(xml_fname, work_dir=work_dir)

        abbyypbox_syncher.sync_doc_offsets(abbyy_xml_doc, pdf_txt_doc)

        txt_unsync_fname = '{}/{}'.format(work_dir, txt_base_fname.replace('.txt', '.txt.unsync'))
        with open(txt_unsync_fname, 'wt') as unsync_fout:
            abbyypbox_syncher.print_abbyy_pbox_unsync(abbyy_xml_doc,
                                                      file=unsync_fout)
            print('wrote {}'.format(txt_unsync_fname))

        # Current still use PDFBox's paragraph.
        # Will switch to Abbyy's after more testing.
        # As it is, not working.
        # Tried with "Carousel Wind PPA 12-27-13.pdf"
        if IS_USE_ABBYY_FOR_PARAGRAPH_INFO:
            paras2_with_attrs, para2_doc_text = \
                abbyyxmlparser.to_paras_with_attrs(abbyy_xml_doc,
                                                   txt_file_name,
                                                   work_dir=work_dir,
                                                   debug_mode=False)

            tmp_para5attrs_fname = txt_base_fname.replace('.txt', '.abbyy.para5attrs')
            paraattrsutils.print_paras_with_attrs(paras2_with_attrs,
                                                  doc_text,
                                                  para2_doc_text,
                                                  '{}/{}'.format(work_dir,
                                                                 tmp_para5attrs_fname))

    if not IS_USE_ABBYY_FOR_PARAGRAPH_INFO:
        # paras2 here is based on information from pdfbox.
        # Current pdfbox outputs lines with only spaces, so it sometime put the text
        # of a whole page as one block, with lines with only spaces as textual lines.
        # To preserve the original annotation performance, we still use this not-so-great
        # txt file as input to corenlp.
        # A better input file could be *.paraline.txt, which is used for lineannotator.
        # In *.paraline.txt, each line is a paragraph, based on some semi-English heuristics.
        # Section header for *.praline.txt is much better than trying to identify section for
        # pages with only 1 block.  Cannot really switch to *.paraline.txt now because
        # double-lined text might cause more trouble.
        paras2_with_attrs, para2_doc_text, unused_gap2_span_list = \
            pdftxtparser.to_paras_with_attrs(pdf_txt_doc,
                                             txt_file_name,
                                             work_dir=work_dir,
                                             debug_mode=False)

        tmp_para5attrs_fname = txt_base_fname.replace('.txt', '.pbox.para5attrs')
        paraattrsutils.print_paras_with_attrs(paras2_with_attrs,
                                              doc_text,
                                              para2_doc_text,
                                              '{}/{}'.format(work_dir,
                                                             tmp_para5attrs_fname))

    text4nlp_fn = get_nlp_fname(txt_base_fname, work_dir)
    txtreader.dumps(para2_doc_text, text4nlp_fn)
    if debug_mode:
        print('wrote {}'.format(text4nlp_fn), file=sys.stderr)

    attrvec_list, nlp_prov_ant_list, origin_lnpos_list, nlp_lnpos_list = \
        nlptxt_to_attrvec_list(para2_doc_text,
                               txt_file_name,
                               txt_base_fname,
                               prov_annotation_list,
                               paras2_with_attrs,
                               work_dir,
                               is_cache_enabled,
                               doc_lang=doc_lang,
                               is_use_corenlp=is_use_corenlp)

    eb_antdoc = EbAnnotatedDoc(file_name=txt_file_name,
                               doc_format=EbDocFormat.pdf,
                               text=doc_text,
                               cpoint_cunit_mapper=cpoint_cunit_mapper,
                               prov_ant_list=prov_annotation_list,
                               is_test=is_test,
                               para_prov_ant_list=nlp_prov_ant_list,
                               attrvec_list=attrvec_list,
                               sechead_list=pdf_txt_doc.sechead_list,
                               paras_with_attrs=paras2_with_attrs,
                               origin_lnpos_list=origin_lnpos_list,
                               nlp_lnpos_list=nlp_lnpos_list,
                               gap_span_list=[],
                               linebreak_arr=linebreak_arr,
                               page_offsets_list=pdf_txt_doc.get_page_offsets(),
                               para_not_linebreak_arr=para_not_linebreak_arr,
                               doc_lang=doc_lang)

    update_special_block_info(eb_antdoc, pdf_txt_doc)

    if abbyy_xml_doc:
        eb_antdoc.abbyy_table_list = tableutils.get_abbyy_table_list(abbyy_xml_doc)
        eb_antdoc.abbyy_signature_list = tableutils.get_abbyy_signature_list(abbyy_xml_doc)

    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled and is_use_corenlp:
        t2_start_time = time.time()
        # tmpFileName = tempfile.NamedTemporaryFile(dir=KIRKE_TMP_DIR, delete=False)
        # joblib.dump(eb_antdoc, tmpFileName.name)
        # shutil.move(tmpFileName.name, eb_antdoc_fn)
        osutils.joblib_atomic_dump(eb_antdoc, eb_antdoc_fn)
        t2_end_time = time.time()
        if (t2_end_time - t2_start_time) * 1000 > 30000:
            logger.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
                        eb_antdoc_fn, len(attrvec_list), (t2_end_time - t2_start_time) * 1000)

    end_time1 = time.time()
    logger.info("pdf_to_ebantdoc: %s, took %.0f msec; %d attrvecs",
                eb_antdoc_fn, (end_time1 - start_time0) * 1000, len(attrvec_list))
    return eb_antdoc


def text_to_corenlp_json(doc_text: str,  # this is what is really processed by corenlp
                         txt_base_fname: str,  # this is only for reference file name
                         work_dir: str,
                         is_cache_enabled: bool = False,
                         doc_lang: str = 'en'):

    # if cache version exists, load that and return
    start_time = time.time()

    # we don't bother to check for is_use_corenlp, assume that's True
    if is_cache_enabled:
        json_fn = get_corenlp_json_fname(txt_base_fname, work_dir)
        if os.path.exists(json_fn):
            corenlp_json = json.loads(strutils.loads(json_fn))
            end_time = time.time()
            logger.info("loading from cache: %s, took %.0f msec",
                        json_fn, (end_time - start_time) * 1000)

            if isinstance(corenlp_json, str):
                # Error in corenlp json file.  Probably caused invalid
                # characters, such as ctrl-a.  Might be related to
                # urlencodeing also.
                # Delete the cache file and try just once more.
                os.remove(json_fn)
                # rest is the same as the 'else' part of no such file exists
                logger.info('calling corenlp on [%s/%s], lang=%s, len=%d',
                            work_dir, txt_base_fname, doc_lang, len(doc_text))
                corenlp_json = corenlputils.annotate_for_enhanced_ner(doc_text, doc_lang=doc_lang)
                # strutils.dumps(json.dumps(corenlp_json), json_fn)
                osutils.atomic_dumps(json.dumps(corenlp_json), json_fn)
                end_time = time.time()
                logger.info("wrote cache file: %s, took %.0f msec",
                            json_fn, (end_time - start_time) * 1000)
        else:
            logger.info('calling corenlp on [%s/%s], lang=%s, len=%d',
                        work_dir, txt_base_fname, doc_lang, len(doc_text))
            corenlp_json = corenlputils.annotate_for_enhanced_ner(doc_text, doc_lang=doc_lang)
            # strutils.dumps(json.dumps(corenlp_json), json_fn)
            osutils.atomic_dumps(json.dumps(corenlp_json), json_fn)
            end_time = time.time()
            logger.info("wrote cache file: %s, took %.0f msec",
                        json_fn, (end_time - start_time) * 1000)
    else:
        logger.info('calling corenlp on [%s/%s], lang=%s, len=%d',
                    work_dir, txt_base_fname, doc_lang, len(doc_text))
        corenlp_json = corenlputils.annotate_for_enhanced_ner(doc_text, doc_lang=doc_lang)
        end_time = time.time()
        logger.info("calling corenlp, took %.0f msec", (end_time - start_time) * 1000)

    return corenlp_json


def text_to_ebantdoc(txt_fname: str,
                     work_dir: Optional[str] = None,
                     is_cache_enabled: bool = True,
                     is_bespoke_mode: bool = False,
                     is_doc_structure: bool = True,
                     doc_lang: str = 'en',
                     is_use_corenlp: bool = True) \
                     -> EbAnnotatedDoc:
    try:
        txt_base_fname = os.path.basename(txt_fname)
        if work_dir is None:
            work_dir = '/tmp'
        eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)

        if is_cache_enabled and is_use_corenlp:
            try:
                # check if file exist, if it is, load it and return
                # regardless of the existing PDF or HtML or is_doc_structure
                eb_antdoc = load_cached_ebantdoc(eb_antdoc_fn)
                if is_bespoke_mode and eb_antdoc:
                    tmp_prov_ant_list, unused_is_test = \
                        ebsentutils.load_prov_annotation_list(txt_fname,
                                                              eb_antdoc.codepoint_to_cunit_mapper)
                    if eb_antdoc.has_same_prov_ant_list(tmp_prov_ant_list):
                        eb_antdoc.file_id = txt_fname
                        return eb_antdoc
                    eb_antdoc = None   # if the annotation has changed, create the whole eb_antdoc
                if eb_antdoc:
                    eb_antdoc.file_id = txt_fname
                    return eb_antdoc
            # pylint: disable=broad-except
            except Exception:
                logger.warning('failed to load ebantdoc2 cache file: [%s].  Load without cache.',
                               eb_antdoc_fn)
                # simply fall through to load the document without loading cache file


        pdf_offsets_filename = txt_fname.replace('.txt', '.offsets.json')
        pdf_xml_filename = txt_fname.replace('.txt', '.pdf.xml')

        # if no doc_structure, simply do the simplest
        if not is_doc_structure:
            eb_antdoc = html_no_docstruct_to_ebantdoc(txt_fname,
                                                      work_dir=work_dir,
                                                      doc_lang=doc_lang,
                                                      is_use_corenlp=is_use_corenlp)
        elif os.path.exists(pdf_offsets_filename):
            eb_antdoc = pdf_to_ebantdoc(txt_fname,
                                        offsets_file_name=pdf_offsets_filename,
                                        pdfxml_file_name=pdf_xml_filename,
                                        work_dir=work_dir,
                                        is_cache_enabled=is_cache_enabled,
                                        doc_lang=doc_lang,
                                        is_use_corenlp=is_use_corenlp)
        else:
            eb_antdoc = html_to_ebantdoc(txt_fname,
                                         work_dir=work_dir,
                                         is_cache_enabled=is_cache_enabled,
                                         doc_lang=doc_lang,
                                         is_use_corenlp=is_use_corenlp)

        ## in doclist, we only want "export-train" dir, not "work_dir"
        # We need to keep .txt with .ebdata together
        eb_antdoc.file_id = txt_fname
        return eb_antdoc
    except IndexError:
        # currently, don't want to indicate it's index error to user.  Too detailed.
        # For developers, we switched the double quote and quote in the message.
        unused_error_type, error_instance, traceback = sys.exc_info()
        error_instance.filename = txt_fname  # type: ignore
        # pylint: disable=line-too-long
        error_instance.user_message = "Problem with parsing document '%s', lang=%s." % (txt_base_fname, doc_lang)  # type: ignore
        error_instance.__traceback__ = traceback  # type: ignore
        raise error_instance  # type: ignore

    # pylint: disable=broad-except
    except Exception:
        unused_error_type, error_instance, traceback = sys.exc_info()
        error_instance.filename = txt_fname  # type: ignore
        # pylint: disable=line-too-long
        error_instance.user_message = 'Problem with parsing document "%s", lang=%s.' % (txt_base_fname, doc_lang)  # type: ignore
        error_instance.__traceback__ = traceback  # type: ignore
        raise error_instance  # type: ignore


def doclist_to_ebantdoc_list_linear(doclist_file: str,
                                    work_dir: str,
                                    is_bespoke_mode: bool = False,
                                    is_doc_structure: bool = False,
                                    doc_lang: str = 'en',
                                    is_use_corenlp: bool = True,
                                    is_sort_by_file_id: bool = False) -> List[EbAnnotatedDoc]:
    logger.debug('ebantdoc.doclist_to_ebantdoc_list_linear(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logger.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    eb_antdoc_list = []
    with open(doclist_file, 'rt') as fin:
        for unused_i, txt_file_name in enumerate(fin, 1):
            txt_file_name = txt_file_name.strip()
            eb_antdoc = text_to_ebantdoc(txt_file_name,
                                         work_dir,
                                         is_bespoke_mode=is_bespoke_mode,
                                         is_doc_structure=is_doc_structure,
                                         doc_lang=doc_lang,
                                         is_use_corenlp=is_use_corenlp)
            eb_antdoc_list.append(eb_antdoc)

    if is_sort_by_file_id:
        eb_antdoc_list = sorted(eb_antdoc_list, key=lambda x: x.file_id)
    logger.debug('Finished ebantdoc.doclist_to_ebantdoc_list_linear()')
    return eb_antdoc_list


def doclist_to_ebantdoc_list(doclist_file: str,
                             work_dir: str,
                             is_cache_enabled: bool = True,
                             is_bespoke_mode: bool = False,
                             is_doc_structure: bool = False,
                             doc_lang: str = 'en',
                             is_use_corenlp: bool = True,
                             is_sort_by_file_id: bool = False) \
                             -> List[EbAnnotatedDoc]:
    logger.debug('ebantdoc.doclist_to_ebantdoc_list(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logger.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)
    txt_fn_list = []
    with open(doclist_file, 'rt') as fin:
        for txt_file_name in fin:
            txt_fn_list.append(txt_file_name.strip())

    fn_eb_antdoc_map = {}
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        future_to_antdoc = {executor.submit(text_to_ebantdoc,
                                            txt_fn,
                                            work_dir,
                                            is_cache_enabled=is_cache_enabled,
                                            is_bespoke_mode=is_bespoke_mode,
                                            is_doc_structure=is_doc_structure,
                                            doc_lang=doc_lang,
                                            is_use_corenlp=is_use_corenlp):
                            txt_fn for txt_fn in txt_fn_list}
        for future in concurrent.futures.as_completed(future_to_antdoc):
            txt_fn = future_to_antdoc[future]
            data = future.result()
            fn_eb_antdoc_map[txt_fn] = data

    eb_antdoc_list = []
    for txt_fn in txt_fn_list:
        eb_antdoc_list.append(fn_eb_antdoc_map[txt_fn])

    if is_sort_by_file_id:
        eb_antdoc_list = sorted(eb_antdoc_list, key=lambda x: x.file_id)
    logger.debug('Finished doclist_to_ebantdoc_list(%s, %s), len= %d',
                 doclist_file, work_dir, len(txt_fn_list))
    return eb_antdoc_list


# Just an alias, in case anyone prefer this.
def doclist_to_ebantdoc_list_no_corenlp(doclist_file: str,
                                        work_dir: str,
                                        is_bespoke_mode: bool = False,
                                        is_doc_structure: bool = False,
                                        doc_lang: str = 'en',
                                        is_sort_by_file_id: bool = False):
    logger.debug('ebantdoc.doclist_to_ebantdoc_list_no_corenlp(%s, %s)', doclist_file, work_dir)
    eb_antdoc_list = doclist_to_ebantdoc_list(doclist_file,
                                              work_dir,
                                              is_bespoke_mode=is_bespoke_mode,
                                              is_doc_structure=is_doc_structure,
                                              doc_lang=doc_lang,
                                              is_use_corenlp=False,
                                              is_sort_by_file_id=is_sort_by_file_id)
    return eb_antdoc_list


def fnlist_to_fn_ebantdoc_map(fn_list: List[str],
                              work_dir: str,
                              is_doc_structure: bool = False):
    logger.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logger.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}

    for i, txt_file_name in enumerate(fn_list, 1):
        eb_antdoc = text_to_ebantdoc(txt_file_name,
                                     work_dir,
                                     is_doc_structure=is_doc_structure)
        fn_ebantdoc_map[txt_file_name] = eb_antdoc
        if i % 10 == 0:
            print("loaded #{} ebantdoc".format(i))
    logger.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map


class EbAntdocProvSet:

    def __init__(self, ebantdoc: EbAnnotatedDoc) -> None:
        self.file_id = ebantdoc.get_file_id()
        self.provset = ebantdoc.get_provision_set()
        self.is_test_set = ebantdoc.is_test_set
        self.prov_annotation_list = ebantdoc.prov_annotation_list

    def get_file_id(self) -> str:
        return self.file_id

    def get_provision_set(self):
        return self.provset


# pylint: disable=invalid-name
def fnlist_to_fn_ebantdoc_provset_map(fn_list: List[str],
                                      work_dir: str,
                                      is_doc_structure: bool = False) -> Dict[str, EbAntdocProvSet]:
    logger.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logger.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}
    for i, txt_file_name in enumerate(fn_list, 1):
        # if i % 10 == 0:
        logger.info("loaded #%d ebantdoc: %s", i, txt_file_name)

        eb_antdoc = text_to_ebantdoc(txt_file_name,
                                     work_dir,
                                     is_doc_structure=is_doc_structure)

        fn_ebantdoc_map[txt_file_name] = EbAntdocProvSet(eb_antdoc)
    logger.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map


# pylint: disable=line-too-long
# para_list has the following format
# ([((2206, 2344, 11), (2183, 2321, 11))], '(a)           The definition of "Applicable Committed Loan Margin" in Article 1 is hereby amended and restated to read in full as follows:', [('sechead', '2.', 'Amendments  to  Credit  Agreement.     ', 52)]),
# the type is
# List, a tuple of
#    span_se_list: List[Tuple[linepos.LnPos, linepos.LnPos]]
#    line: str
#    attr_list: List[Tuple]
# this is sorted_by_from

# List[Tuple[List[Tuple[linepos.LnPos,
#           linepos.LnPos]],
#  str,
# List[Tuple[Any]]]]

# this is not tested
def print_para_list(eb_antdoc):
    doc_text = eb_antdoc.text
    for i, para_with_attr in enumerate(eb_antdoc.paras_with_attrs, 1):
        span_frto_list, unused_para_text, attr_list = para_with_attr
        (orig_start, orig_end), (unused_to_start, unused_to_end) = docutils.span_frto_list_to_fromto(span_frto_list)
        para_text2 = doc_text[orig_start:orig_end].replace(r'[\n\t]', ' ')[:30]

        cols = [str(i), '{}\t{}'.format(span_frto_list, str(attr_list)),
                para_text2]
        print('\t'.join(cols))


def print_attrvec_list(eb_antdoc: EbAnnotatedDoc):
    doc_text = eb_antdoc.get_nlp_text()
    for i, attrvec in enumerate(eb_antdoc.attrvec_list, 1):
        tmp_start = attrvec.start
        tmp_end = attrvec.end
        sent_text = doc_text[tmp_start:tmp_end].replace(r'[\n\t]', ' ')
        labels_st = ""
        if attrvec.labels:
            labels_st = ','.join(sorted(attrvec.labels))
        cols = [str(i), '[{}]'.format(labels_st), sent_text]
        print('\t'.join(cols))


def print_line_list(eb_antdoc):
    for i, para_with_attr in enumerate(eb_antdoc.paras_with_attrs, 1):
        print('{}\t{}'.format(i, para_with_attr))

# pylint: disable=invalid-name
def dump_ebantdoc_attrvec_with_secheads(eb_antdoc: EbAnnotatedDoc):
    print("dump_ebantdoc_attrvec_with_secheads: len(attrevec_list) = {}".format(len(eb_antdoc.attrvec_list)))
    for attrvec in eb_antdoc.attrvec_list:
        print("attrvec = {}".format(attrvec))


# this is in-place operations
def prov_ants_cpoint_to_cunit(prov_ants_map, cpoint_to_cunit_mapper):
    for unused_prov, ant_list in prov_ants_map.items():
        for ant_json in ant_list:
            ant_json['cpoint_start'], ant_json['cpoint_end'] = ant_json['start'], ant_json['end']
            ant_json['start'], ant_json['end'] = \
                cpoint_to_cunit_mapper.to_cunit_offsets(ant_json['start'],
                                                        ant_json['end'])

            try:
                ant_json['span_list']
            except KeyError:
                ant_json['span_list'] = [{'start': ant_json['start'], 'end': ant_json['end']}]

            for span_json in ant_json['span_list']:
                span_json['cpoint_start'], span_json['cpoint_end'] = span_json['start'], span_json['end']
                span_json['start'], span_json['end'] = \
                    cpoint_to_cunit_mapper.to_cunit_offsets(span_json['start'],
                                                            span_json['end'])
