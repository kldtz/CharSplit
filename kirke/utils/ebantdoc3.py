import concurrent.futures
from enum import Enum
import json
import logging
import os
import shutil
import sys
import time
from typing import Any, DefaultDict, Dict, List, Set, Tuple

from sklearn.externals import joblib

from kirke.docstruct import docutils, fromtomapper, htmltxtparser, pdfoffsets, pdftxtparser
from kirke.utils import corenlputils, strutils, osutils, textoffset, txtreader, ebsentutils


CORENLP_JSON_VERSION = '1.5'
EBANTDOC_VERSION = '1.5'

def get_nlp_fname(txt_basename: str, work_dir: str) -> str:
    base_fn = txt_basename.replace('.txt', '.nlp.v{}.txt'.format(CORENLP_JSON_VERSION))
    return '{}/{}'.format(work_dir, base_fn)


def get_ebant_fname(txt_basename: str, work_dir: str) -> str:
    base_fn =  txt_basename.replace('.txt',
                                    '.ebantdoc.v{}.pkl'.format(EBANTDOC_VERSION))
    return '{}/{}'.format(work_dir, base_fn)


class EbDocFormat(Enum):
    html_nodocstruct = 1
    html = 2
    pdf = 3
    other = 4

class EbAnnotatedDoc3:

    # pylint: disable=R0913
    def __init__(self,
                 *,   # force specify all parameters by keyword
                 file_name,
                 doc_format: EbDocFormat,
                 text,
                 cpoint_cunit_mapper,
                 prov_ant_list,
                 is_test,
                 para_doc_text,        # adjusted
                 para_prov_ant_list,   # adjusted
                 paras_with_attrs,     # adjusted
                 origin_sx_lnpos_list,
                 nlp_sx_lnpos_list,
                 gap_span_list,  # TODO, jshaw, maybe del in future
                 nl_text='',
                 page_offsets_list=None,
                 paraline_text='',
                 doc_lang='en') -> None:
        self.file_id = file_name
        self.doc_format = doc_format
        self.text = text
        self.len_text = len(text)   # used to check out of bound

        self.codepoint_to_cunit_mapper = cpoint_cunit_mapper

        self.prov_annotation_list = prov_ant_list
        self.is_test_set = is_test
        self.provision_set = set([prov_ant.label for prov_ant in prov_ant_list])
        self.doc_lang = doc_lang

        self.nlp_text = para_doc_text
        self.para_prov_ant_list = para_prov_ant_list
        self.paras_with_attrs = paras_with_attrs
        # to map to original offsets
        self.nlp_sx_lnpos_list = nlp_sx_lnpos_list     # now it's list of (start, linepos.LnPos)
        self.origin_sx_lnpos_list = origin_sx_lnpos_list
        self.gap_span_list = gap_span_list

        self.nl_text = nl_text
        self.paraline_text = paraline_text
        self.page_offsets_list = page_offsets_list

        self.table_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.chart_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.toc_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.pagenum_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.footer_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]
        self.signature_list = []  # type: List[Tuple[int, int, Dict[str, Any]]]


    def get_file_id(self) -> str:
        return self.file_id

    def set_provision_annotations(self, ant_list: List[Dict[str, Any]]) -> None:
        self.prov_annotation_list = ant_list

    def get_provision_annotations(self) -> List[Dict[str, Any]]:
        return self.prov_annotation_list

    def get_provision_set(self) -> Set[str]:
        return self.provision_set

    def get_text(self) -> str:
        return self.text

    def has_same_prov_ant_list(self, prov_ant_list2) -> bool:
        return self.prov_annotation_list == prov_ant_list2

    def get_doc_format(self) -> EbDocFormat:
        return self.doc_format

    def get_nlp_text(self) -> str:
        return self.nlp_text


def remove_prov_greater_offset(prov_annotation_list, max_offset: int):
    return [prov_ant for prov_ant in prov_annotation_list
            if prov_ant.start < max_offset]


def load_cached_ebantdoc3(eb_antdoc_fn: str):
    """Load from pickled file if file exist, otherwise None"""

    # if cache version exists, load that and return
    if os.path.exists(eb_antdoc_fn):
        start_time = time.time()
        # print("before loading\t{}".format(eb_antdoc_fn))
        eb_antdoc = joblib.load(eb_antdoc_fn)
        # print("done loading\t{}".format(eb_antdoc_fn))
        end_time = time.time()
        logging.info("loading from cache: %s, took %.0f msec",
                     eb_antdoc_fn, (end_time - start_time) * 1000)

        return eb_antdoc

    return None


def nlptxt_to_attrvec_list(para_doc_text,
                           txt_file_name,
                           txt_base_fname,
                           prov_annotation_list,
                           paras_with_attrs,
                           work_dir,
                           is_cache_enabled,
                           doc_lang="en"):
    # para_doc_text is what is sent, not txt_base_fname

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
        fromto_mapper = None
        nlp_prov_ant_list = prov_annotation_list

    # print("prov_annotation: {}".format(prov_annotation))

    # let's adjust the offsets in prov_annotation to keep things simple and
    # maximize reuse of existing code.

    # these lists has to be sorted by nlp_sx_lnpos_list
    original_sx_lnpos_list, nlp_sx_lnpos_list = fromtomapper.paras_to_fromto_lists(paras_with_attrs)

    return nlp_prov_ant_list, original_sx_lnpos_list, nlp_sx_lnpos_list


# stop at 'exhibit_appendix' or 'exhibit_appendix_complete'
def html_no_docstruct_to_ebantdoc3(txt_file_name,
                                   work_dir,
                                   is_cache_enabled=True,
                                   doc_lang="en"):
    debug_mode = False
    start_time0 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)

    txt_file_name, doc_text, prov_annotation_list, is_test, cpoint_cunit_mapper = \
        chop_at_exhibit_complete(txt_file_name, txt_base_fname, work_dir, debug_mode)

    paras_with_attrs = []
    nlp_prov_ant_list, _, _ = nlptxt_to_attrvec_list(doc_text,
                                                     txt_file_name,
                                                     txt_base_fname,
                                                     prov_annotation_list,
                                                     paras_with_attrs,
                                                     work_dir,
                                                     is_cache_enabled,
                                                     doc_lang=doc_lang)

    # there is no nlp.txt
    para_doc_text = doc_text
    nlp_prov_ant_list = prov_annotation_list
    origin_sx_lnpos_list = []
    nlp_sx_lnpos_list = []
    gap_span_list = []
    eb_antdoc = EbAnnotatedDoc3(file_name=txt_file_name,
                                doc_format=EbDocFormat.html_nodocstruct,
                                text=doc_text,
                                cpoint_cunit_mapper=cpoint_cunit_mapper,
                                prov_ant_list=prov_annotation_list,
                                is_test=is_test,
                                para_doc_text=para_doc_text,
                                para_prov_ant_list=nlp_prov_ant_list,
                                paras_with_attrs=paras_with_attrs,
                                origin_sx_lnpos_list=origin_sx_lnpos_list,
                                nlp_sx_lnpos_list=nlp_sx_lnpos_list,
                                gap_span_list=gap_span_list,
                                # there is no nl_text
                                # page_offsets_list
                                # paraline_text
                                doc_lang=doc_lang)
                                
    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled:
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        logging.info("wrote cache file: %s, took %.0f msec",
                     eb_antdoc_fn, (end_time - start_time) * 1000)

    end_time = time.time()
    return eb_antdoc


def chop_at_exhibit_complete(txt_file_name: str,
                             txt_base_fname: str,
                             work_dir: str,
                             debug_mode: bool=False) -> Tuple[str, str, List[ebsentutils.ProvisionAnnotation],
                                                              bool, textoffset.TextCpointCunitMapper]:
    doc_text = txtreader.loads(txt_file_name)
    cpoint_cunit_mapper = textoffset.TextCpointCunitMapper(doc_text)
    prov_annotation_list, is_test = ebsentutils.load_prov_annotation_list(txt_file_name, cpoint_cunit_mapper)
    max_txt_size = len(doc_text)
    is_chopped = False
    for prov_ant in prov_annotation_list:
        if prov_ant.label in set(['exhibit_appendix', 'exhibit_appendix_complete']):
            # print('{}\t{}\t{}\t{}'.format(txt_file_name, prov_ant.start, prov_ant.end, prov_ant.label))
            if prov_ant.start < max_txt_size:
                max_txt_size = prov_ant.start
                is_chopped = True
    if is_chopped:
        # print("txt_chopped: [{}]".format(doc_text[max_txt_size:max_txt_size+50] + "..."))
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
def html_to_ebantdoc3(txt_file_name,
                      work_dir,
                      is_cache_enabled=True,
                      doc_lang='en'):
    debug_mode = False
    start_time1 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)
    # print("html_to_ebantdoc3({}, {}, is_cache_eanbled={}".format(txt_file_name, work_dir, is_cache_enabled))

    txt_file_name, doc_text, prov_annotation_list, is_test, cpoint_cunit_mapper = \
        chop_at_exhibit_complete(txt_file_name, txt_base_fname, work_dir, debug_mode)

    paras_with_attrs, para_doc_text, gap_span_list, _ = \
            htmltxtparser.parse_document(txt_file_name,
                                         work_dir=work_dir,
                                         is_combine_line=True)

    txt4nlp_fname = get_nlp_fname(txt_base_fname, work_dir)
    txtreader.dumps(para_doc_text, txt4nlp_fname)
    if debug_mode:
        print("wrote {}".format(txt4nlp_fname), file=sys.stderr)

    nlp_prov_ant_list, origin_sx_lnpos_list, nlp_sx_lnpos_list = \
        nlptxt_to_attrvec_list(para_doc_text,
                               txt_file_name,
                               txt_base_fname,
                               prov_annotation_list,
                               paras_with_attrs,
                               work_dir,
                               is_cache_enabled,
                               doc_lang=doc_lang)

    eb_antdoc = EbAnnotatedDoc3(file_name=txt_file_name,
                                doc_format=EbDocFormat.html,
                                text=doc_text,
                                cpoint_cunit_mapper=cpoint_cunit_mapper,
                                prov_ant_list=prov_annotation_list,
                                is_test=is_test,
                                para_doc_text=para_doc_text,
                                para_prov_ant_list=nlp_prov_ant_list,
                                paras_with_attrs=paras_with_attrs,
                                origin_sx_lnpos_list=origin_sx_lnpos_list,
                                nlp_sx_lnpos_list=nlp_sx_lnpos_list,
                                gap_span_list=gap_span_list,
                                # there is no nl_text
                                # page_offsets_list
                                # paraline_text
                                doc_lang=doc_lang)
                                
    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled:
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        #logging.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
        #             eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time1 = time.time()
    logging.info("html_to_ebantdoc3: %s, took %.0f msec",
                 eb_antdoc_fn, (end_time1 - start_time1) * 1000)
    return eb_antdoc

def update_special_block_info(eb_antdoc: EbAnnotatedDoc3,
                              pdf_txt_doc: pdfoffsets.PDFTextDoc) -> None:
    eb_antdoc.table_list = pdf_txt_doc.special_blocks_map.get('table', [])
    eb_antdoc.chart_list = pdf_txt_doc.special_blocks_map.get('chart', [])
    eb_antdoc.signature_list = pdf_txt_doc.special_blocks_map.get('signature', [])
    eb_antdoc.toc_list = pdf_txt_doc.special_blocks_map.get('toc', [])
    eb_antdoc.pagenum_list = pdf_txt_doc.special_blocks_map.get('pagenum', [])
    eb_antdoc.footer_list = pdf_txt_doc.special_blocks_map.get('footer', [])


# this parses both originally text and html documents
# It's main goal is to detect sechead
# optionally pagenum, footer, toc, signature
def pdf_to_ebantdoc3(txt_file_name,
                     offsets_file_name,
                     work_dir,
                     is_cache_enabled=True,
                     doc_lang='en'):
    debug_mode = False
    start_time0 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)
    offsets_base_fname = os.path.basename(offsets_file_name)

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

    # copy txt file to work/txt_base_name, to be consistent with html_to_ebantdoc3()
    if txt_file_name != '{}/{}'.format(work_dir, txt_base_fname):
        shutil.copy2(txt_file_name, '{}/{}'.format(work_dir, txt_base_fname))
        shutil.copy2(offsets_file_name, '{}/{}'.format(work_dir, offsets_base_fname))

    doc_text, nl_text, paraline_text, nl_fname, paraline_fname, cpoint_cunit_mapper = \
        pdftxtparser.to_nl_paraline_texts(txt_file_name, offsets_file_name, work_dir=work_dir)

    prov_annotation_list, is_test = ebsentutils.load_prov_annotation_list(txt_file_name, cpoint_cunit_mapper)

    pdf_text_doc = pdftxtparser.parse_document(txt_file_name, work_dir=work_dir)

    # paras2 here is based on information from pdfbox.
    # Current pdfbox outputs lines with only spaces, so it sometime put the text
    # of a whole page as one block, with lines with only spaces as textual lines.
    # To preserve the original annotation performance, we still use this not-so-great
    # txt file as input to corenlp.
    # A better input file could be *.paraline.txt, which is used for lineannotator.
    # In *.paraline.txt, each line is a paragraph, based on some semi-English heuristics.
    # Section header for *.praline.txt is much better than trying to identify section for
    # pages with only 1 block.  Cannot really switch to *.paraline.txt now because double-lined text
    # might cause more trouble.
    paras2_with_attrs, para2_doc_text, gap2_span_list = \
        pdftxtparser.to_paras_with_attrs(pdf_text_doc, txt_file_name, work_dir=work_dir, debug_mode=False)

    text4nlp_fn = get_nlp_fname(txt_base_fname, work_dir)
    txtreader.dumps(para2_doc_text, text4nlp_fn)
    if debug_mode:
        print('wrote {}'.format(text4nlp_fn), file=sys.stderr)

    nlp_prov_ant_list, origin_sx_lnpos_list, nlp_sx_lnpos_list = \
        nlptxt_to_attrvec_list(para2_doc_text,
                               txt_file_name,
                               txt_base_fname,
                               prov_annotation_list,
                               paras2_with_attrs,
                               work_dir,
                               is_cache_enabled,
                               doc_lang=doc_lang)

    eb_antdoc = EbAnnotatedDoc3(file_name=txt_file_name,
                                doc_format=EbDocFormat.pdf,
                                text=doc_text,
                                cpoint_cunit_mapper=cpoint_cunit_mapper,
                                prov_ant_list=prov_annotation_list,
                                is_test=is_test,
                                para_doc_text=para2_doc_text,
                                para_prov_ant_list=nlp_prov_ant_list,
                                paras_with_attrs=paras2_with_attrs,
                                origin_sx_lnpos_list=origin_sx_lnpos_list,
                                nlp_sx_lnpos_list=nlp_sx_lnpos_list,
                                gap_span_list=gap2_span_list,
                                nl_text=nl_text,
                                page_offsets_list=pdf_text_doc.get_page_offsets(),
                                paraline_text=paraline_text,
                                doc_lang=doc_lang)
                                # paraline_text=paraline2_text)

    update_special_block_info(eb_antdoc, pdf_text_doc)

    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled:
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        # logging.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
        #             eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time1 = time.time()
    logging.info("pdf_to_ebantdoc3: %s, took %.0f msec",
                 eb_antdoc_fn, (end_time1 - start_time0) * 1000)
    return eb_antdoc


def text_to_ebantdoc3(txt_fname,
                      work_dir=None,
                      is_cache_enabled=True,
                      is_bespoke_mode=False,
                      is_doc_structure=True,
                      doc_lang="en"):
    txt_base_fname = os.path.basename(txt_fname)
    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    # never want to save in bespoke_mode because annotation can change
        #if os.path.exists(eb_antdoc_fn):
        #    os.remove(eb_antdoc_fn)
        # corenlp should be cache so that we don't run it again for same
        # files.
        # is_cache_enabled = False
        
    if is_cache_enabled:
        # check if file exist, if it is, load it and return
        # regarless of the existing PDF or HtML or is_doc_structure
        eb_antdoc = load_cached_ebantdoc3(eb_antdoc_fn)
        if is_bespoke_mode and eb_antdoc:
            tmp_prov_ant_list, is_test = ebsentutils.load_prov_annotation_list(txt_fname,
                                                                               eb_antdoc.codepoint_to_cunit_mapper)
            if eb_antdoc.has_same_prov_ant_list(tmp_prov_ant_list):
                return eb_antdoc
            eb_antdoc = None   # if the annotation has changed, create the whole eb_antdoc
        if eb_antdoc:
            return eb_antdoc

    pdf_offsets_filename = txt_fname.replace('.txt', '.offsets.json')
    # if no doc_structure, simply do the simplest
    if not is_doc_structure:
        eb_antdoc = html_no_docstruct_to_ebantdoc3(txt_fname, work_dir=work_dir, doc_lang=doc_lang)
    elif os.path.exists(pdf_offsets_filename):
        eb_antdoc = pdf_to_ebantdoc3(txt_fname, pdf_offsets_filename, work_dir=work_dir,
                                     is_cache_enabled=is_cache_enabled, doc_lang=doc_lang)
    else:
        eb_antdoc = html_to_ebantdoc3(txt_fname, work_dir=work_dir,
                                      is_cache_enabled=is_cache_enabled, doc_lang=doc_lang)

    return eb_antdoc


def doclist_to_ebantdoc_list_linear(doclist_file,
                                    work_dir,
                                    is_bespoke_mode=False,
                                    is_doc_structure=True):
    logging.debug('ebantdoc3.doclist_to_ebantdoc_list_linear(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)
        
    eb_antdoc_list = []
    with open(doclist_file, 'rt') as fin:
        for i, txt_file_name in enumerate(fin, 1):
            txt_file_name = txt_file_name.strip()
            eb_antdoc = text_to_ebantdoc3(txt_file_name,
                                          work_dir,
                                          is_bespoke_mode=is_bespoke_mode,
                                          is_doc_structure=is_doc_structure)
            eb_antdoc_list.append(eb_antdoc)
    logging.debug('Finished ebantdoc3.doclist_to_ebantdoc_list_linear()')
    return eb_antdoc_list


# is_doc_structure is false make loeading much faster
# since there is no heading detection.
def doclist_to_ebantdoc_list(doclist_file,
                             work_dir,
                             is_bespoke_mode=False,
                             is_doc_structure=True):
    logging.debug('ebantdoc3.doclist_to_ebantdoc_list(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    txt_fn_list = []
    with open(doclist_file, 'rt') as fin:
        for txt_file_name in fin:
            txt_fn_list.append(txt_file_name.strip())

    fn_eb_antdoc_map = {}
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        future_to_antdoc = {executor.submit(text_to_ebantdoc3,
                                            txt_fn,
                                            work_dir,
                                            is_bespoke_mode=is_bespoke_mode,
                                            is_doc_structure=is_doc_structure):
                            txt_fn for txt_fn in txt_fn_list}
        for future in concurrent.futures.as_completed(future_to_antdoc):
            txt_fn = future_to_antdoc[future]
            data = future.result()
            fn_eb_antdoc_map[txt_fn] = data

    eb_antdoc_list = []
    for txt_fn in txt_fn_list:
        eb_antdoc_list.append(fn_eb_antdoc_map[txt_fn])

    logging.debug('Finished doclist_to_ebantdoc_list({}, {}), len= {}'.format(doclist_file,
                                                                              work_dir,
                                                                              len(txt_fn_list)))

    return eb_antdoc_list


def fnlist_to_fn_ebantdoc_map(fn_list, work_dir, is_doc_structure=False):
    logging.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}

    for i, txt_file_name in enumerate(fn_list, 1):
        eb_antdoc = text_to_ebantdoc3(txt_file_name,
                                      work_dir,
                                      is_doc_structure=is_doc_structure)
        fn_ebantdoc_map[txt_file_name] = eb_antdoc
        if i % 10 == 0:
            print("loaded #{} ebantdoc".format(i))
    logging.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map


class EbAntdocProvSet:

    def __init__(self, ebantdoc) -> None:
        self.file_id = ebantdoc.get_file_id()
        self.provset = ebantdoc.get_provision_set()
        self.is_test_set = ebantdoc.is_test_set

    def get_file_id(self):
        return self.file_id
    
    def get_provision_set(self):
        return self.provset


def fnlist_to_fn_ebantdoc_provset_map(fn_list, work_dir, is_doc_structure=False):
    logging.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}
    for i, txt_file_name in enumerate(fn_list, 1):
        # if i % 10 == 0:
        logging.info("loaded #{} ebantdoc: {}".format(i, txt_file_name))

        eb_antdoc = text_to_ebantdoc3(txt_file_name,
                                      work_dir,
                                      is_doc_structure=is_doc_structure)

        fn_ebantdoc_map[txt_file_name] = EbAntdocProvSet(eb_antdoc)
    logging.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map


# this is not tested
def print_para_list(eb_antdoc):
    doc_text = eb_antdoc.text
    for i, para_with_attr in enumerate(eb_antdoc.paras_with_attrs, 1):
        # print('{}\t{}'.format(i, para_with_attr))
        span_frto_list, para_text, attr_list = para_with_attr
        (orig_start, orig_end), (to_start, to_end) = docutils.span_frto_list_to_fromto(span_frto_list)
        #  orig_start, orig_end = orig_offsets
        para_text2 = doc_text[orig_start:orig_end].replace(r'[\n\t]', ' ')[:30]

        """
        tmp_list = []
        for attr in attr_list:
            attr_name, attr_val, attr_text, attr_offset = attr
            tmp_list.append((attr_name, attr_val, attr_text[:20], attr_offset))
        attr_list = tmp_list
        """

        cols = [str(i), '{}\t{}'.format(span_frto_list, str(attr_list)),
                para_text2]
        print('\t'.join(cols))


def print_line_list(eb_antdoc):
    doc_text = eb_antdoc.text
    for i, para_with_attr in enumerate(eb_antdoc.paras_with_attrs, 1):
        print('{}\t{}'.format(i, para_with_attr))
