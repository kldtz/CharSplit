import json
import logging
import os
import re
import sys
import time
from pathlib import Path
import concurrent.futures

from sklearn.externals import joblib

from kirke.eblearn import sent2ebattrvec, ebsentutils
from kirke.docstruct import doc_pdf_reader

from kirke.eblearn import ebattrvec

from kirke.utils import corenlputils, ebantdoc, mathutils, strutils, osutils, entityutils, txtreader
from kirke.docstruct import htmltxtparser, docreader

from kirke.utils import ebantdoc, txtreader

CORENLP_JSON_VERSION = '1.2'
EBANTDOC_VERSION = '1.2'

def get_corenlp_json_fname(txt_basename, work_dir):
    base_fn = txt_basename.replace('.txt',
                                   '.corenlp.v{}.json'.format(CORENLP_JSON_VERSION))
    return '{}/{}'.format(work_dir, base_fn)

def get_ebant_fname(txt_basename, work_dir):
    base_fn =  txt_basename.replace('.txt',
                                    '.ebantdoc.v{}.pkl'.format(EBANTDOC_VERSION))
    return '{}/{}'.format(work_dir, base_fn)

def get_nlp_fname(txt_basename, work_dir):
    base_fn = txt_basename.replace('.txt', '.nlp.txt')
    return '{}/{}'.format(work_dir, base_fn)
        

# use the following code to map the offsets back
"""
        # translate the offsets
        all_prov_ant_list = []
        for provision, ant_list in prov_labels_map.items():
            for antx in ant_list:
                # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
                xstart = antx['start']
                xend = antx['end']
                antx['corenlp_start'] = xstart
                antx['corenlp_end'] = xend
                antx['start'] = docreader.find_offset_to(xstart, from_list, to_list)
                antx['end'] = docreader.find_offset_to(xend, from_list, to_list)

                all_prov_ant_list.append(antx)

        # this update the 'start_end_span_list' in each antx in-place
        docreader.update_ant_spans(all_prov_ant_list, gap_span_list, orig_doc_text)
"""

class EbAnnotatedDoc2:

    # pylint: disable=R0913
    def __init__(self,
                 file_name,
                 text, 
                 prov_ant_list,
                 is_test,
                 para_doc_text,        # adjusted
                 para_prov_ant_list,   # adjusted
                 attrvec_list,         # adjusted
                 paras_with_attrs,     # adjusted
                 to_list,
                 from_list,
                 gap_span_list,
                 nl_text='',
                 paraline_text=''):
        self.file_id = file_name
        self.text = text
        self.prov_annotation_list = prov_ant_list
        self.is_test_set = is_test
        self.provision_set = [prov_ant.label for prov_ant in prov_ant_list]

        self.nlp_text = para_doc_text
        self.para_prov_ant_list = para_prov_ant_list
        self.attrvec_list = attrvec_list
        self.paras_with_attrs = paras_with_attrs
        # to map to original offsets
        self.to_list = to_list
        self.from_list = from_list
        self.gap_span_list = gap_span_list

    def get_file_id(self):
        return self.file_id

    def set_provision_annotations(self, ant_list):
        self.prov_annotation_list = ant_list

    def get_provision_annotations(self):
        return self.prov_annotation_list

    def get_provision_set(self):
        return self.provision_set

    def get_attrvec_list(self):
        return self.attrvec_list

    def get_text(self):
        return self.text


def remove_prov_greater_offset(prov_annotation_list, max_offset):
    return [prov_ant for prov_ant in prov_annotation_list if prov_ant.start < max_offset]


def load_cached_ebantdoc2(eb_antdoc_fn: str):
    """Load from pickled file if file exist, otherwise None"""

    # if cache version exists, load that and return
    if os.path.exists(eb_antdoc_fn):
        start_time = time.time()
        eb_antdoc = joblib.load(eb_antdoc_fn)
        end_time = time.time()
        logging.info("loading from cache: %s, took %.0f msec", eb_antdoc_fn, (end_time - start_time) * 1000)

        # TODO, jshaw, remove after debugging
        # save_ebantdoc_sents(eb_antdoc, txt_file_name)
        return eb_antdoc

    return None


def dump_ebantdoc_attrvec(eb_antdoc):
    from_list = eb_antdoc.from_list
    to_list = eb_antdoc.to_list
    for attrvec in eb_antdoc.attrvec_list:
        print('{}\t{}\t{}'.format(attrvec.start,
                                  attrvec.end, eb_antdoc.nlp_text[attrvec.start:attrvec.end]))

        if from_list:
            xstart = attrvec.start
            xend = attrvec.end
            orig_start = docreader.find_offset_to(xstart, from_list, to_list)
            orig_end = docreader.find_offset_to(xend, from_list, to_list)
            print('{}\t{}\t{}'.format(orig_start,  orig_end, eb_antdoc.text[orig_start:orig_end]))

def dump_ebantdoc_attrvec_with_secheads(eb_antdoc):
    from_list = eb_antdoc.from_list
    to_list = eb_antdoc.to_list
    for attrvec in eb_antdoc.attrvec_list:
        print('{}\t{}\t{}\t{}'.format(attrvec.start,
                                  attrvec.end,
                                  attrvec.sechead,
                                  eb_antdoc.nlp_text[attrvec.start:attrvec.end]))

        if from_list:
            xstart = attrvec.start
            xend = attrvec.end
            orig_start = docreader.find_offset_to(xstart, from_list, to_list)
            orig_end = docreader.find_offset_to(xend, from_list, to_list)
            print('{}\t{}\t{}\t{}'.format(orig_start, orig_end, attrvec.sechead,
                                          eb_antdoc.text[orig_start:orig_end]))

        
def nlptxt_to_attrvec_list(para_doc_text,
                           txt_file_name,
                           txt_base_fname,
                           prov_annotation_list,
                           paras_with_attrs,
                           work_dir,
                           is_cache_enabled):
    # para_doc_text is what is sent, not txt_base_fname
    corenlp_json = text_to_corenlp_json(para_doc_text,
                                        txt_base_fname,
                                        work_dir=work_dir,
                                        is_cache_enabled=is_cache_enabled)

    if paras_with_attrs:
        from_list_xx, to_list_xx = htmltxtparser.paras_to_fromto_lists(paras_with_attrs)
        # At this point, put all document structure information into
        # ebsent_list
        # We also adjust all annotations from CoreNlp into the offsets from original
        # document.  Offsets is no NLP-based.
        nlp_prov_ant_list = []
        for prov_annotation in prov_annotation_list:
            orig_start, orig_end = prov_annotation.start, prov_annotation.end
            orig_label = prov_annotation.label

            # print("prov_annotation: {}".format(prov_annotation))
            # print("\torig\t[{}]".format(orig_doc_text[orig_start:orig_end]))
            # nlp_start = docreader.find_offset_to(orig_start, from_list_xx, to_list_xx)
            # nlp_end = docreader.find_offset_to(orig_end, from_list_xx, to_list_xx)
            # print("\tnlp\t[{}]".format(para_doc_text[nlp_start:nlp_end]))

            xstart = docreader.find_offset_to(orig_start, from_list_xx, to_list_xx)
            xend = docreader.find_offset_to(orig_end, from_list_xx, to_list_xx)
            nlp_prov_ant_list.append(ebantdoc.ProvisionAnnotation(xstart, xend, orig_label))
        # print("prov_annotation: {}".format(prov_annotation))
    else:
        nlp_prov_ant_list = prov_annotation_list


    # let's adjust the offsets in prov_annotation to keep things simple and
    # maximize reuse of existing code.

    ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name,
                                                           corenlp_json,
                                                           para_doc_text,
                                                           is_doc_structure=True)
    # ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name, corenlp_json, para_doc_text)
    # print('number of sentences: {}'.format(len(ebsent_list)))

    if paras_with_attrs:
        # still haven't add the sechead info back into
        ebsentutils.update_ebsents_with_sechead(ebsent_list, paras_with_attrs)

    # fix any domain specific entity extraction, such as 'Lessee' as a location
    # this is a in-place replacement
    # We only handle up to "exhibit_appendix,exhibit_appendix_complete"
    for ebsent in ebsent_list:
        ebsentutils.fix_ner_tags(ebsent)
        ebsentutils.populate_ebsent_entities(ebsent, para_doc_text[ebsent.start:ebsent.end])

        overlap_provisions = (ebsentutils.get_labels_if_start_end_overlap(ebsent.start,
                                                                          ebsent.end,
                                                                          nlp_prov_ant_list)
                              if nlp_prov_ant_list else [])
        # logging.info("overlap_provisions: {}".format(overlap_provisions))

        ebsent.set_labels(overlap_provisions)

    attrvec_list = []
    num_sent = len(ebsent_list)
    # we need prev and next sentences because such information are used in the
    # feature extraction
    prev_ebsent, next_ebsent = None, None
    for sent_idx, ebsent in enumerate(ebsent_list):
        # sent_st = ebsent.get_text()
        if sent_idx != num_sent-1:
            next_ebsent = ebsent_list[sent_idx + 1]
        else:
            next_ebsent = None
        fvec = sent2ebattrvec.sent2ebattrvec(txt_file_name, ebsent, sent_idx + 1,
                                             prev_ebsent, next_ebsent, para_doc_text)
        attrvec_list.append(fvec)
        prev_ebsent = ebsent

    return attrvec_list, nlp_prov_ant_list


# stop at 'exhibit_appendix' or 'exhibit_appendix_complete'
def html_no_docstruct_to_ebantdoc2(txt_file_name,
                                   work_dir,
                                   is_cache_enabled=True):
    debug_mode = True
    start_time0 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)

    doc_text = txtreader.loads(txt_file_name)
    prov_annotation_list, is_test = ebantdoc.load_prov_annotation_list(txt_file_name)
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


    # write the shortened files
    # if file is not shortened, write to dir-work
    txt_file_name = '{}/{}'.format(work_dir, txt_base_fname)
    txtreader.dumps(doc_text, txt_file_name)
    if debug_mode:
        print('wrote {}'.format(txt_file_name, file=sys.stderr))

    paras_with_attrs = []
    attrvec_list, nlp_prov_ant_list = nlptxt_to_attrvec_list(doc_text,
                                                             txt_file_name,
                                                             txt_base_fname,
                                                             prov_annotation_list,
                                                             paras_with_attrs,
                                                             work_dir,
                                                             is_cache_enabled)

    # there is no nlp.txt
    para_doc_text = doc_text
    nlp_prov_ant_list = prov_annotation_list
    to_list = []
    from_list = []
    gap_span_list = []
    eb_antdoc = EbAnnotatedDoc2(txt_file_name,
                                doc_text,
                                prov_annotation_list,
                                is_test,
                                para_doc_text,
                                nlp_prov_ant_list,
                                attrvec_list,
                                paras_with_attrs,
                                to_list,
                                from_list,
                                gap_span_list)
                                
    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled:
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        logging.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
                     eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time = time.time()
    logging.info("html_no_docstruct_to_ebantdoc2: %s, took %.0f msec; %d attrvecs",
                 eb_antdoc_fn, (end_time - start_time) * 1000, len(attrvec_list))    
    return eb_antdoc


# stop at 'exhibit_appendix' or 'exhibit_appendix_complete'
def html_to_ebantdoc2(txt_file_name,
                      work_dir,
                      is_cache_enabled=True):
    debug_mode = True
    start_time0 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)

    doc_text = txtreader.loads(txt_file_name)
    prov_annotation_list, is_test = ebantdoc.load_prov_annotation_list(txt_file_name)
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

    paras_with_attrs, para_doc_text, gap_span_list, _ = \
            htmltxtparser.parse_document(txt_file_name,
                                         work_dir=work_dir,
                                         is_combine_line=True)
    # I am a little messed up on from_to lists
    # not sure exactly what "from" means, original text or nlp text
    to_list, from_list = htmltxtparser.paras_to_fromto_lists(paras_with_attrs)

    txt4nlp_fname = get_nlp_fname(txt_base_fname, work_dir)
    txtreader.dumps(para_doc_text, txt4nlp_fname)
    if debug_mode:
        print("wrote {}".format(txt4nlp_fname), file=sys.stderr)

    attrvec_list, nlp_prov_ant_list = nlptxt_to_attrvec_list(para_doc_text,
                                                             txt_file_name,
                                                             txt_base_fname,
                                                             prov_annotation_list,
                                                             paras_with_attrs,
                                                             work_dir,
                                                             is_cache_enabled)

    eb_antdoc = EbAnnotatedDoc2(txt_file_name,
                                doc_text,
                                prov_annotation_list,
                                is_test,
                                para_doc_text,
                                nlp_prov_ant_list,
                                attrvec_list,
                                paras_with_attrs,
                                to_list,
                                from_list,
                                gap_span_list)
                                
    start_time1 = time.time()

    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled:
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        logging.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
                     eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time = time.time()
    logging.info("html_to_ebantdoc2: %s, took %.0f msec; %d attrvecs",
                 eb_antdoc_fn, (end_time - start_time0) * 1000, len(attrvec_list))
    return eb_antdoc


# this parses both originally text and html documents
# It's main goal is to detect sechead
# optionally pagenum, footer, toc, signature
def pdf_to_ebantdoc2(txt_file_name,
                     offsets_file_name,
                     work_dir,
                     is_cache_enabled=True):
    debug_mode = True
    start_time0 = time.time()
    txt_base_fname = os.path.basename(txt_file_name)    

    doc_text, nl_text, paraline_text, nl_fname, paraline_fname = \
       doc_pdf_reader.to_nl_paraline_texts(txt_file_name, offsets_file_name, work_dir=work_dir)
    prov_annotation_list, is_test = ebantdoc.load_prov_annotation_list(txt_file_name)

    # gap_span_list is for sentv2.txt or xxx.txt?
    # the offsets in para_list is for doc_text
    #doc_text, gap_span_list, text4nlp_fn, text4nlp_offsets_fn, para_list = \
    #     docreader.parse_html_document(txt_file_name, linfo_file_name, work_dir=work_dir)

    paras_with_attrs, para_doc_text, gap_span_list, _ = \
        htmltxtparser.parse_document(txt_file_name,
                                     work_dir=work_dir,
                                     is_combine_line=False)  # this line diff from annotate_htmled_document()
    # I am a little messed up on from_to lists
    # not sure exactly what "from" means, original text or nlp text
    to_list, from_list = htmltxtparser.paras_to_fromto_lists(paras_with_attrs)

    text4nlp_fn = '{}/{}'.format(work_dir, txt_base_fname.replace('.txt', '.nlp.txt'))
    txtreader.dumps(para_doc_text, text4nlp_fn)
    if debug_mode:
        print('wrote 235 {}'.format(text4nlp_fn), file=sys.stderr)


    txt4nlp_fname = get_nlp_fname(txt_base_fname, work_dir)
    txtreader.dumps(para_doc_text, txt4nlp_fname)
    if debug_mode:
        print("wrote {}".format(txt4nlp_fname), file=sys.stderr)

    attrvec_list, nlp_prov_ant_list = nlptxt_to_attrvec_list(para_doc_text,
                                                             txt_file_name,
                                                             txt_base_fname,
                                                             prov_annotation_list,
                                                             paras_with_attrs,
                                                             work_dir,
                                                             is_cache_enabled)

    eb_antdoc = EbAnnotatedDoc2(txt_file_name,
                                doc_text,
                                prov_annotation_list,
                                is_test,
                                para_doc_text,
                                nlp_prov_ant_list,
                                attrvec_list,
                                paras_with_attrs,
                                to_list,
                                from_list,
                                gap_span_list,
                                nl_text = nl_text,
                                paraline_text = paraline_text)

    start_time1 = time.time()

    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    if txt_file_name and is_cache_enabled:
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        logging.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
                     eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time = time.time()
    logging.info("pdf_to_ebantdoc2: %s, took %.0f msec; %d attrvecs",
                 eb_antdoc_fn, (end_time - start_time) * 1000, len(attrvec_list))
    return eb_antdoc


def text_to_corenlp_json(doc_text,  # this is what is really processed by corenlp
                         txt_base_fname,  # this is only for reference file name
                         work_dir,
                         is_cache_enabled=False):

    # if cache version exists, load that and return
    start_time = time.time()
    
    if is_cache_enabled:
        json_fn = get_corenlp_json_fname(txt_base_fname, work_dir)
        if os.path.exists(json_fn):
            corenlp_json = json.loads(strutils.loads(json_fn))
            end_time = time.time()
            logging.info("loading from cache: %s, took %.0f msec", json_fn, (end_time - start_time) * 1000)

            if isinstance(corenlp_json, str):
                # Error in corenlp json file.  Probably caused invalid
                # characters, such as ctrl-a.  Might be related to
                # urlencodeing also.
                # Delete the cache file and try just once more.
                os.remove(json_fn)
                # rest is the same as the 'else' part of no such file exists
                corenlp_json = corenlputils.annotate_for_enhanced_ner(doc_text)
                strutils.dumps(json.dumps(corenlp_json), json_fn)
                end_time = time.time()
                logging.info("wrote cache file: %s, took %.0f msec", json_fn, (end_time - start_time) * 1000)
        else:
            corenlp_json = corenlputils.annotate_for_enhanced_ner(doc_text)
            strutils.dumps(json.dumps(corenlp_json), json_fn)
            end_time = time.time()
            logging.info("wrote cache file: %s, took %.0f msec", json_fn, (end_time - start_time) * 1000)
    else:
        corenlp_json = corenlputils.annotate_for_enhanced_ner(doc_text)
        end_time = time.time()
        logging.info("calling corenlp, took %.0f msec", (end_time - start_time) * 1000)

    return corenlp_json


def text_to_ebantdoc2(txt_fname,
                      work_dir=None,
                      is_cache_enabled=True,
                      is_bespoke_mode=False,
                      is_doc_structure=True):
    txt_base_fname = os.path.basename(txt_fname)
    eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
    # never want to save in bespoke_mode because annotation can change
    if is_bespoke_mode:
        if os.path.exists(eb_antdoc_fn):
            os.remove(eb_antdoc_fn)
        is_cache_enabled = False
        
    if is_cache_enabled:
        # check if file exist, if it is, load it and return
        # regarless of the existing PDF or HtML or is_doc_structure
        eb_antdoc = load_cached_ebantdoc2(eb_antdoc_fn)
        if eb_antdoc:
            return eb_antdoc

    pdf_offsets_filename = txt_fname.replace('.txt', '.offsets.json')
    # if no doc_structure, simply do the simplest
    if not is_doc_structure:
        eb_antdoc = html_no_docstruct_to_ebantdoc2(txt_fname, work_dir=work_dir)
    elif os.path.exists(pdf_offsets_filename):
        eb_antdoc = pdf_to_ebantdoc2(txt_fname, pdf_offsets_filename, work_dir=work_dir,
                                     is_cache_enabled=is_cache_enabled)
    else:
        eb_antdoc = html_to_ebantdoc2(txt_fname, work_dir=work_dir,
                                      is_cache_enabled=is_cache_enabled)

    return eb_antdoc



def doclist_to_ebantdoc_list_linear(doclist_file,
                             work_dir,
                             is_bespoke_mode=False,
                             is_doc_structure=False):
    logging.debug('doclist_to_ebantdoc_list(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    eb_antdoc_list = []
    with open(doclist_file, 'rt') as fin:
        for i, txt_file_name in enumerate(fin, 1):
            txt_file_name = txt_file_name.strip()
            eb_antdoc = text_to_ebantdoc2(txt_file_name,
                                          work_dir,
                                          is_bespoke_mode=is_bespoke_mode,
                                          is_doc_structure=is_doc_structure)
            eb_antdoc_list.append(eb_antdoc)
    logging.debug('Finished ebantdoc2.doc_list_to_ebantdoc_list_linear()')

    return eb_antdoc_list


def doclist_to_ebantdoc_list_linear(doclist_file,
                                    work_dir,
                                    is_bespoke_mode=False,
                                    is_doc_structure=False):
    logging.debug('ebantdoc2.doclist_to_ebantdoc_list_linear(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)
        
    eb_antdoc_list = []
    with open(doclist_file, 'rt') as fin:
        for i, txt_file_name in enumerate(fin, 1):
            txt_file_name = txt_file_name.strip()
            eb_antdoc = text_to_ebantdoc2(txt_file_name,
                                          work_dir,
                                          is_bespoke_mode=is_bespoke_mode,
                                          is_doc_structure=is_doc_structure)
            eb_antdoc_list.append(eb_antdoc)
    logging.debug('Finished ebantdoc2.doc_list_to_ebantdoc_list_linear()')


def doclist_to_ebantdoc_list(doclist_file,
                             work_dir,
                             is_bespoke_mode=False,
                             is_doc_structure=False):
    logging.debug('ebantdoc2.doclist_to_ebantdoc_list(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    txt_fn_list = []
    with open(doclist_file, 'rt') as fin:
        for txt_file_name in fin:
            txt_fn_list.append(txt_file_name.strip())

    fn_eb_antdoc_map = {}
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        future_to_antdoc = {executor.submit(text_to_ebantdoc2,
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

    logging.debug('Finished doclist_to_ebantdoc_list({}/{}), len= {}'.format(work_dir,
                                                                             doclist_file,
                                                                             len(txt_fn_list)))

    return eb_antdoc_list





#    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(list(txt_file_set), work_dir=work_dir)
def fnlist_to_fn_ebantdoc_map(fn_list, work_dir, is_doc_structure=False):
    logging.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}

    for i, txt_file_name in enumerate(fn_list, 1):
        eb_antdoc = text_to_ebantdoc2(txt_file_name,
                                      work_dir,
                                      is_doc_structure=is_doc_structure)
        fn_ebantdoc_map[txt_file_name] = eb_antdoc
        if i % 10 == 0:
            print("loaded #{} ebantdoc".format(i))
    logging.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map


class EbAntdocProvSet:

    def __init__(self, ebantdoc):
        self.file_id = ebantdoc.get_file_id()
        self.provset = ebantdoc.get_provision_set()
        self.is_test_set = ebantdoc.is_test_set

    def get_file_id(self):
        return self.file_id
    
    def get_provision_set(self):
        return self.provset


#    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(list(txt_file_set), work_dir=work_dir)
def fnlist_to_fn_ebantdoc_provset_map(fn_list, work_dir, is_doc_structure=False):
    logging.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}
    for i, txt_file_name in enumerate(fn_list, 1):
        # if i % 10 == 0:
        logging.info("loaded #{} ebantdoc: {}".format(i, txt_file_name))

        eb_antdoc = text_to_ebantdoc2(txt_file_name,
                                      work_dir,
                                      is_doc_structure=is_doc_structure)

        fn_ebantdoc_map[txt_file_name] = EbAntdocProvSet(eb_antdoc)
    logging.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map
