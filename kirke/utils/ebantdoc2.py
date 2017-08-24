import json
import logging
import os
import re
import sys
import time
from pathlib import Path
import concurrent.futures

from sklearn.externals import joblib

from kirke.eblearn import sent2ebattrvec
from kirke.eblearn import ebsentutils

# TODO, remove.  this is mainly for printing out sentence text for debug
# at the end of parsexxx
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
                 to_list,
                 from_list):
        self.file_id = file_name
        self.text = text
        self.prov_annotation_list = prov_ant_list
        self.is_test_set = is_test
        self.provision_set = [prov_ant.label for prov_ant in prov_ant_list]

        self.para_doc_text = para_doc_text
        self.para_prov_ant_list = para_prov_ant_list
        self.attrvec_list = attrvec_list
        # to map to original offsets
        self.to_list = to_list
        self.from_list = from_list

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

"""
def html_to_ebantdoc2(txt_fname, work_dir=None):

    # perform document structuring all all text file
    # First perform document structuring
    #   The new text file will have section header in separate lines, plus
    #   no page number.
    # Then, go through CoreNlp with the new text file
    # Adjusted the offsets in the annotation from corenlp
    paras_with_attrs, para_doc_text, gap_span_list, orig_doc_text = \
            htmltxtparser.parse_document(txt_file_name,
                                         work_dir=work_dir,
                                         is_combine_line=is_combine_line)

    # TODO, jshaw, remove later
    base_fname = os.path.basename(txt_file_name)
    txt4nlp_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.txt'))
    txtreader.dumps(para_doc_text, txt4nlp_fname)
    # print("wrote {}".format(txt4nlp_fname), file=sys.stderr)

    corenlp_json = text_file_name_to_corenlp_json(txt_file_name,
                                                  para_doc_text=para_doc_text,
                                                  is_bespoke_mode=is_bespoke_mode,
                                                  work_dir=work_dir,
                                                  is_cache_enabled=is_cache_enabled)
    
    prov_ant_fn = txt_file_name.replace('.txt', '.ant')
    prov_ant_file = Path(prov_ant_fn)
    prov_ebdata_fn = txt_file_name.replace('.txt', '.ebdata')
    prov_ebdata_file = Path(prov_ebdata_fn)

    prov_annotation_list = []
    is_test = False
    if os.path.exists(prov_ant_fn):
        # in is_bespoke_mode, only the annotation for a particular provision
        # is returned.
        prov_annotation_list = (ebantdoc.load_provision_annotations(prov_ant_fn, provision)
                                if prov_ant_file.is_file() else [])
    elif os.path.exists(prov_ebdata_fn):
        prov_annotation_list, is_test = (ebantdoc.load_prov_ebdata(prov_ebdata_fn)
                                         if prov_ebdata_file.is_file() else ([], False))

    from_list, to_list = htmltxtparser.paras_to_fromto_lists(paras_with_attrs)
    # At this point, put all document structure information into
    # ebsent_list
    # We also adjust all annotations from CoreNlp into the offsets from original
    # document.  Offsets is no NLP-based.
    for prov_annotation in prov_annotation_list:
        orig_start, orig_end = prov_annotation.start, prov_annotation.end

        # print("prov_annotation: {}".format(prov_annotation))
        # print("\torig\t[{}]".format(orig_doc_text[orig_start:orig_end]))
        # nlp_start = docreader.find_offset_to(orig_start, from_list, to_list)
        # nlp_end = docreader.find_offset_to(orig_end, from_list, to_list)
        # print("\tnlp\t[{}]".format(para_doc_text[nlp_start:nlp_end]))

        prov_annotation.start = docreader.find_offset_to(orig_start, from_list, to_list)
        prov_annotation.end = docreader.find_offset_to(orig_end, from_list, to_list)
        # print("prov_annotation: {}".format(prov_annotation))

    # let's adjust the offsets in prov_annotation to keep things simple and
    # maximize reuse of existing code.

    ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name,
                                                           corenlp_json,
                                                           para_doc_text,
                                                           is_doc_structure=True)
    # ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name, corenlp_json, para_doc_text)
    # print('number of sentences: {}'.format(len(ebsent_list)))

    # still haven't add the sechead info back into
    update_ebsents_with_sechead(ebsent_list, paras_with_attrs)

    # fix any domain specific entity extraction, such as 'Lessee' as a location
    # this is a in-place replacement
    # We only handle up to "exhibit_appendix,exhibit_appendix_complete"
    ebsents_without_exhibit = []
    exhibit_appendix_start = -1
    for ebsent in ebsent_list:
        fix_ner_tags(ebsent)
        populate_ebsent_entities(ebsent, para_doc_text[ebsent.start:ebsent.end])

        overlap_provisions = (get_labels_if_start_end_overlap(ebsent.start,
                                                              ebsent.end,
                                                              prov_annotation_list)
                              if prov_annotation_list else [])
        # logging.info("overlap_provisions: {}".format(overlap_provisions))

        ebsent.set_labels(overlap_provisions)
        if ('exhibit_appendix' in overlap_provisions or
            'exhibit_appendix_complete' in overlap_provisions):
            exhibit_appendix_start = ebsent.start
            # logging.info('exhibit_appendix_start: {}'.format(exhibit_appendix_start))
            break
        ebsents_without_exhibit.append(ebsent)

    # we need to chop provisions after exhibit_appendix_start also
    if exhibit_appendix_start != -1:
        tmp_prov_annotation_list = []
        for prov_annotation in prov_annotation_list:
            if (exhibit_appendix_start <= prov_annotation.start or
                mathutils.start_end_overlap((exhibit_appendix_start, exhibit_appendix_start+1),
                                            (prov_annotation.start, prov_annotation.end))):
                #logging.info("skipping prov '{}' {}, after appendix offset {}".format(prov_annotation.label,
                #                                                                      prov_annotation.start,
                #                                                                      exhibit_appendix_start))
                pass
            else:
                tmp_prov_annotation_list.append(prov_annotation)
        prov_annotation_list = tmp_prov_annotation_list

    # we reset ebsent_list to ebsents_without_exhibit
    ebsent_list = ebsents_without_exhibit

    start_time0 = time.time()
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

    eb_antdoc = ebantdoc.EbAnnotatedDoc(txt_file_name,
                                        prov_annotation_list,
                                        attrvec_list,
                                        para_doc_text,
                                        is_test)
    start_time1 = time.time()
    logging.info("sent2ebattrvec: %d attrvecs, took %.0f msec", len(attrvec_list), (start_time1 - start_time0) * 1000)

    # never want to save in bespoke_mode because annotation can change
    if txt_file_name and is_cache_enabled and not is_bespoke_mode:
        txt_basename = os.path.basename(txt_file_name)
        # if cache version exists, load that and return
        # eb_antdoc_fn = work_dir + "/" + txt_basename.replace('.txt', '.ebantdoc.pkl')
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        logging.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
                     eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time = time.time()
    # logging.debug("parse_to_ebantdoc: %s, took %.0f msec", eb_antdoc_fn, (end_time - start_time) * 1000)

    # TODO, jshaw, remove, this saves the sentence text version
    # if txt_file_name:
    #    save_ebantdoc_sents(eb_antdoc, txt_file_name)

    return eb_antdoc

def pdf_to_ebantdoc2(txt_fname, offsets_fname, work_dir=None):
    orig_text, nl_text, paraline_text, nl_fname, paraline_fname = \
        doc_pdf_reader.to_nl_paraline_texts(txt_fname, offsets_fname, work_dir=work_dir)

        base_fname = os.path.basename(file_name)    

    paras_with_attrs, para_doc_text, gap_span_list, orig_doc_text = \
        htmltxtparser.parse_document(file_name,
                                     work_dir=work_dir)
    text4nlp_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.txt'))
    txtreader.dumps(para_doc_text, text4nlp_fn)
    if debug_mode:
        print('wrote 234 {}'.format(text4nlp_fn), file=sys.stderr)

    # I am a little messed up on from_to lists
    # not sure exactly what "from" means, original text or nlp text
    to_list, from_list = htmltxtparser.paras_to_fromto_lists(paras_with_attrs)
"""

def remove_prov_greater_offset(prov_annotation_list, max_offset):
    return [prov_ant for prov_ant in prov_annotation_list if prov_ant.start < max_offset]


# stop at 'exhibit_appendix' or 'exhibit_appendix_complete'
def html_to_ebantdoc2(txt_file_name, work_dir, is_cache_enabled=True, is_bespoke_mode=False):
    debug_mode = True
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

    txt_base_fname = os.path.basename(txt_file_name)
    # write the shortened files
    txt_file_name = '{}/{}'.format(work_dir, txt_base_fname)
    txtreader.dumps(doc_text, txt_file_name)
    if debug_mode:
        print('wrote {}'.format(txt_file_name, file=sys.stderr))
    
    paras_with_attrs, para_doc_text, gap_span_list, _ = \
            htmltxtparser.parse_document(txt_file_name,
                                         work_dir=work_dir,
                                         is_combine_line=True)

    txt4nlp_fname = get_nlp_fname(txt_base_fname, work_dir)
    txtreader.dumps(para_doc_text, txt4nlp_fname)
    if debug_mode:
        print("wrote {}".format(txt4nlp_fname), file=sys.stderr)

    # para_doc_text is what is sent, not txt_base_fname
    corenlp_json = text_to_corenlp_json(para_doc_text,
                                        txt_base_fname,
                                        work_dir=work_dir,
                                        is_cache_enabled=is_cache_enabled)
    
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

    # let's adjust the offsets in prov_annotation to keep things simple and
    # maximize reuse of existing code.

    ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name,
                                                           corenlp_json,
                                                           para_doc_text,
                                                           is_doc_structure=True)
    # ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name, corenlp_json, para_doc_text)
    # print('number of sentences: {}'.format(len(ebsent_list)))

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

    start_time0 = time.time()
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

    # I am a little messed up on from_to lists
    # not sure exactly what "from" means, original text or nlp text
    to_list, from_list = htmltxtparser.paras_to_fromto_lists(paras_with_attrs)

    eb_antdoc = EbAnnotatedDoc2(txt_file_name,
                                doc_text,
                                prov_annotation_list,
                                is_test,
                                para_doc_text,
                                nlp_prov_ant_list,
                                attrvec_list,
                                to_list,
                                from_list)
                                
    start_time1 = time.time()
    logging.info("sent2ebattrvec: %d attrvecs, took %.0f msec", len(attrvec_list), (start_time1 - start_time0) * 1000)

    # never want to save in bespoke_mode because annotation can change
    if txt_file_name and is_cache_enabled and not is_bespoke_mode:
        eb_antdoc_fn = get_ebant_fname(txt_base_fname, work_dir)
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        logging.info("wrote cache file: %s, num_sent = %d, took %.0f msec",
                     eb_antdoc_fn, len(attrvec_list), (end_time - start_time) * 1000)

    end_time = time.time()
    # logging.debug("parse_to_ebantdoc: %s, took %.0f msec", eb_antdoc_fn, (end_time - start_time) * 1000)

    # TODO, jshaw, remove, this saves the sentence text version
    # if txt_file_name:
    #    save_ebantdoc_sents(eb_antdoc, txt_file_name)

    return eb_antdoc
        
def dump_ebantdoc_attrvec(eb_antdoc):
    from_list = eb_antdoc.from_list
    to_list = eb_antdoc.to_list
    for attrvec in eb_antdoc.attrvec_list:
        print('{}\t{}\t{}'.format(attrvec.start,  attrvec.end, eb_antdoc.para_doc_text[attrvec.start:attrvec.end]))

        xstart = attrvec.start
        xend = attrvec.end
        orig_start = docreader.find_offset_to(xstart, from_list, to_list)
        orig_end = docreader.find_offset_to(xend, from_list, to_list)
        print('{}\t{}\t{}'.format(orig_start,  orig_end, eb_antdoc.text[orig_start:orig_end]))        

        
    #if ('exhibit_appendix' in overlap_provisions or
    #        'exhibit_appendix_complete' in overlap_provisions):
                
def text_to_ebantdoc2(txt_fname, work_dir=None):
    debug_mode = True
    time1 = time.time()

    pdf_offsets_filename = txt_fname.replace('.txt', '.offsets.json')

    if os.path.exists(pdf_offsets_filename):
        eb_antdoc2 = pdf_to_ebantdoc2(txt_fname, pdf_offsets_filename, work_dir=work_dir)
    else:
        eb_antdoc2 = html_to_ebantdoc2(txt_fname, work_dir=work_dir)

    return eb_antdoc2



def text_to_corenlp_json(doc_text,  # this is what is really processed by corenlp
                         txt_basename,  # this is only for reference file name
                         work_dir,
                         is_cache_enabled=False):

    # if cache version exists, load that and return
    start_time = time.time()
    
    if is_cache_enabled:
        json_fn = get_corenlp_json_fname(txt_basename, work_dir)
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
