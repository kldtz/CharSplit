import json
import logging
import os
import sys
import time
import concurrent.futures

from sklearn.externals import joblib

from kirke.eblearn import sent2ebattrvec, ebsentutils


# TODO, remove.  this is mainly for printing out sentence text for debug
# at the end of parsexxx
from kirke.eblearn import ebattrvec

from kirke.utils import corenlputils, ebantdoc, mathutils, strutils, osutils, entityutils, txtreader
from kirke.docstruct import htmltxtparser, docreader


# TODO, reverse this in production
DEFAULT_IS_CACHE_ENABLED = True
# DEFAULT_IS_CACHE_ENABLED = False
CORENLP_JSON_VERSION='1.1'


# def filter_feature_start_end(feat_json_list, feature_name_set):
#    result_list = []
#    for feat_json in feat_json_list:
#        if feat_json['type'] in feature_name_set:
#            result_list.append((feat_json['type'], feat_json['start'], feat_json['end']))
#    return result_list


def save_ebantdoc_sents(eb_antdoc, txt_file_name):
    txt_basename = os.path.basename(txt_file_name)
    doc_sents_dir = 'dir-doc-sents'
    doc_sents_fn = doc_sents_dir + "/" + txt_basename.replace('.txt', '.sent')
    doc_text = eb_antdoc.text
    ts_col = 'TRAIN'
    if eb_antdoc.is_test_set:
        ts_col = 'TEST'
    # print("doc_sents_fn = {}".format(doc_sents_fn))
    with open(doc_sents_fn, 'wt') as fout3:
        for i, attrvec in enumerate(eb_antdoc.attrvec_list, 1):
            # print("attrvec = {}".format(attrvec))
            tmp_start = attrvec.start
            tmp_end = attrvec.end
            sent_text = doc_text[tmp_start:tmp_end].replace(r'[\n\t]', ' ')
            labels_st = ""
            if attrvec.labels:
                labels_st = ','.join(sorted(attrvec.labels))
            cols = [str(i), ts_col, labels_st, sent_text]
            print('\t'.join(cols), file=fout3)


def load_cached_ebantdoc(txt_file_name,
                         is_bespoke_mode,
                         work_dir,
                         is_cache_enabled=False):
    """Load from pickled file if file exist, otherwise None"""
    txt_basename = os.path.basename(txt_file_name)
    eb_antdoc_fn = '{}/{}'.format(work_dir, txt_basename.replace('.txt', '.ebantdoc.pkl'))

    # make sure we do not have cached ebantdoc if in bespoke_mode
    if is_bespoke_mode and os.path.isfile(eb_antdoc_fn):
        os.remove(eb_antdoc_fn)

    # if cache version exists, load that and return
    if is_cache_enabled:
        if not is_bespoke_mode and os.path.exists(eb_antdoc_fn):
            start_time = time.time()
            eb_antdoc = joblib.load(eb_antdoc_fn)
            end_time = time.time()
            logging.info("loading from cache: %s, took %.0f msec", eb_antdoc_fn, (end_time - start_time) * 1000)

            # TODO, jshaw, remove after debugging
            # save_ebantdoc_sents(eb_antdoc, txt_file_name)
            return eb_antdoc, eb_antdoc_fn

    return None, eb_antdoc_fn


def text_file_name_to_corenlp_json(txt_file_name,
                                   para_doc_text,
                                   is_bespoke_mode,
                                   work_dir,
                                   is_cache_enabled=False):
    # print("txt_file_name= [{}]".format(txt_file_name))
    if txt_file_name:
        txt_basename = os.path.basename(txt_file_name)

        # if cache version exists, load that and return
        if is_cache_enabled:
            json_fn = '{}/{}'.format(work_dir, txt_basename.replace('.txt', '.corenlp.v{}.json'.format(CORENLP_JSON_VERSION)))
            if os.path.exists(json_fn):
                start_time = time.time()
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
                    start_time = time.time()
                    corenlp_json = corenlputils.annotate_for_enhanced_ner(para_doc_text)
                    end_time = time.time()
                    strutils.dumps(json.dumps(corenlp_json), json_fn)
                    logging.info("wrote cache file: %s, took %.0f msec", json_fn, (end_time - start_time) * 1000)
            else:
                start_time = time.time()
                corenlp_json = corenlputils.annotate_for_enhanced_ner(para_doc_text)
                end_time = time.time()
                strutils.dumps(json.dumps(corenlp_json), json_fn)
                logging.info("wrote cache file: %s, took %.0f msec", json_fn, (end_time - start_time) * 1000)
        else:
            start_time = time.time()
            corenlp_json = corenlputils.annotate_for_enhanced_ner(para_doc_text)
            end_time = time.time()
            logging.info("calling corenlp, took %.0f msec", (end_time - start_time) * 1000)
    else:
        start_time = time.time()
        corenlp_json = corenlputils.annotate_for_enhanced_ner(para_doc_text)
        end_time = time.time()
        logging.info("calling corenlp, took %.0f msec", (end_time - start_time) * 1000)

    return corenlp_json


# output_json is not None for debugging purpose
# pylint: disable=R0914
# If is_bespoke mode, the annotation can change across different bespoke runs.
# As a result, never cache .ebantdoc.pkl, but can reuse corenlp.json
def parse_to_eb_antdoc(atext,
                       txt_file_name,
                       work_dir=None,
                       is_bespoke_mode=False,
                       provision=None):
    # load/save the corenlp file if output_dir is specified
    is_cache_enabled = False if work_dir is None else DEFAULT_IS_CACHE_ENABLED

    start_time = time.time()
    eb_antdoc, eb_antdoc_fn = load_cached_ebantdoc(txt_file_name,
                                                   is_bespoke_mode=is_bespoke_mode,
                                                   work_dir=work_dir,
                                                   is_cache_enabled=is_cache_enabled)
    if eb_antdoc:
        end_time = time.time()
        logging.debug("parse_to_eb_antdoc(): %s, took %.0f msec",
                      eb_antdoc_fn,
                      (end_time - start_time) * 1000)
        return eb_antdoc

    corenlp_json = text_file_name_to_corenlp_json(txt_file_name,
                                                  para_doc_text=atext,
                                                  is_bespoke_mode=is_bespoke_mode,
                                                  work_dir=work_dir,
                                                  is_cache_enabled=is_cache_enabled)

    prov_annotation_list, is_test = ebantdoc.load_prov_annotation_list(txt_file_name, provision)

    ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name, corenlp_json, atext)
    # print('number of sentences: {}'.format(len(ebsent_list)))

    # fix any domain specific entity extraction, such as 'Lessee' as a location
    # this is a in-place replacement
    # We only handle up to "exhibit_appendix,exhibit_appendix_complete"
    ebsents_without_exhibit = []
    exhibit_appendix_start = -1
    for ebsent in ebsent_list:
        ebsentutils.fix_ner_tags(ebsent)
        ebsentutils.populate_ebsent_entities(ebsent, atext[ebsent.start:ebsent.end])

        overlap_provisions = (ebsentutils.get_labels_if_start_end_overlap(ebsent.start,
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

    # we reset ebsent_list to ebsents_withotu_exhibit
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
                                             prev_ebsent, next_ebsent, atext)
        attrvec_list.append(fvec)
        prev_ebsent = ebsent

    eb_antdoc = ebantdoc.EbAnnotatedDoc(txt_file_name, prov_annotation_list, attrvec_list, atext, is_test)
    start_time1 = time.time()
    logging.info("sent2ebattrvec: %d attrvecs, took %.0f msec", len(attrvec_list), (start_time1 - start_time0) * 1000)

    # never want to save in bespoke_mode because annotation can change
    if txt_file_name and is_cache_enabled and not is_bespoke_mode:
        txt_basename = os.path.basename(txt_file_name)
        # if cache version exists, load that and return
        eb_antdoc_fn = work_dir + "/" + txt_basename.replace('.txt', '.ebantdoc.pkl')
        start_time = time.time()
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        end_time = time.time()
        logging.info("wrote cache file: %s, took %.0f msec", eb_antdoc_fn, (end_time - start_time) * 1000)

    end_time = time.time()
    logging.debug("parse_to_ebantdoc: %s, took %.0f msec", eb_antdoc_fn, (end_time - start_time) * 1000)

    # TODO, jshaw, remove, this saves the sentence text version
    # if txt_file_name:
    #    save_ebantdoc_sents(eb_antdoc, txt_file_name)

    return eb_antdoc


# output_json is not None for debugging purpose
# pylint: disable=R0914
# If is_bespoke mode, the annotation can change across different bespoke runs.
# As a result, never cache .ebantdoc.pkl, but can reuse corenlp.json
def parse_to_eb_antdoc_with_doc_structure(atext_ignore,
                                          txt_file_name,
                                          work_dir=None,
                                          is_bespoke_mode=False,
                                          provision=None,
                                          is_combine_line=True):
    # load/save the corenlp file if output_dir is specified
    is_cache_enabled = False if work_dir is None else DEFAULT_IS_CACHE_ENABLED

    start_time = time.time()
    eb_antdoc, eb_antdoc_fn = load_cached_ebantdoc(txt_file_name,
                                                   is_bespoke_mode=is_bespoke_mode,
                                                   work_dir=work_dir,
                                                   is_cache_enabled=is_cache_enabled)
    if eb_antdoc:
        end_time = time.time()
        logging.debug("parse_to_eb_antdoc_with_doc_structure: %s, took %.0f msec",
                      eb_antdoc_fn, (end_time - start_time) * 1000)
        return eb_antdoc

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
    
    prov_annotation_list, is_test = ebantdoc.load_prov_annotation_list(txt_file_name, provision)

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
    ebsentutils.update_ebsents_with_sechead(ebsent_list, paras_with_attrs)

    # fix any domain specific entity extraction, such as 'Lessee' as a location
    # this is a in-place replacement
    # We only handle up to "exhibit_appendix,exhibit_appendix_complete"
    ebsents_without_exhibit = []
    exhibit_appendix_start = -1
    for ebsent in ebsent_list:
        ebsentutils.fix_ner_tags(ebsent)
        ebsentutils.populate_ebsent_entities(ebsent, para_doc_text[ebsent.start:ebsent.end])

        overlap_provisions = (ebsentutils.get_labels_if_start_end_overlap(ebsent.start,
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


def doc_to_ebantdoc(txt_file_name,
                    work_dir,
                    is_bespoke_mode=False,
                    is_doc_structure=False,
                    provision=None,
                    is_combine_line=True):
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    start_time = time.time()
    doc_text = txtreader.loads(txt_file_name)
    if is_doc_structure:
        eb_antdoc = parse_to_eb_antdoc_with_doc_structure(doc_text,
                                                          txt_file_name,
                                                          work_dir=work_dir,
                                                          is_bespoke_mode=is_bespoke_mode,
                                                          provision=provision,
                                                          is_combine_line=is_combine_line)
    else:
        eb_antdoc = parse_to_eb_antdoc(doc_text,
                                       txt_file_name,
                                       work_dir=work_dir,
                                       is_bespoke_mode=is_bespoke_mode,
                                       provision=provision)
    now_time = time.time()
    logging.debug('doc_to_ebantdoc(): %s, took %.2f sec',
                  txt_file_name, now_time - start_time)
    # in the case or processing many files at once, make sure gevent has
    # the opportunity to run other waiting threads
    # gevent.sleep(.00001)
    return eb_antdoc


def doc_to_ebantdoc_with_paras(txt_file_name,
                               work_dir,
                               is_bespoke_mode=False,
                               is_doc_structure=False,
                               provision=None,
                               is_combine_line=True):
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    start_time = time.time()
    paras_with_attrs, para_doc_text, gap_span_list, orig_doc_text = \
        htmltxtparser.parse_document(txt_file_name,
                                     work_dir=work_dir)

    # TODO, jshaw, remove later
    base_fname = os.path.basename(txt_file_name)
    text4nlp_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.txt'))
    txtreader.dumps(para_doc_text, text4nlp_fn)
    print('wrote {}'.format(text4nlp_fn), file=sys.stderr)

    doc_text = txtreader.loads(txt_file_name)
    if is_doc_structure:
        eb_antdoc = parse_to_eb_antdoc_with_doc_structure(doc_text,
                                                          txt_file_name,
                                                          work_dir=work_dir,
                                                          is_bespoke_mode=is_bespoke_mode,
                                                          provision=provision,
                                                          is_combine_line=is_combine_line)
    else:
        eb_antdoc = parse_to_eb_antdoc(doc_text,
                                       txt_file_name,
                                       work_dir=work_dir,
                                       is_bespoke_mode=is_bespoke_mode,
                                       provision=provision)


    now_time = time.time()
    logging.debug('doc_to_ebantdoc_with_paras(): %s, took %.2f sec',
                  txt_file_name, now_time - start_time)

    return eb_antdoc, paras_with_attrs, para_doc_text


# paralle version
def doclist_to_ebantdoc_list(doclist_file,
                             work_dir,
                             is_bespoke_mode=False,
                             is_doc_structure=False,
                             provision=None,
                             is_combine_line=True):
    logging.debug('doclist_to_ebantdoc_list(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    txt_fn_list = []
    with open(doclist_file, 'rt') as fin:
        for txt_file_name in fin:
            txt_fn_list.append(txt_file_name.strip())

    fn_eb_antdoc_map = {}
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        future_to_antdoc = {executor.submit(doc_to_ebantdoc,
                                            txt_fn,
                                            work_dir,
                                            is_bespoke_mode,
                                            is_doc_structure,
                                            provision,
                                            is_combine_line):
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


def doclist_to_ebantdoc_list_linear(doclist_file,
                                    work_dir,
                                    is_bespoke_mode=False,
                                    is_doc_structure=False):
    logging.debug('doclist_to_ebantdoc_list_linear(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)
    eb_antdoc_list = []
    with open(doclist_file, 'rt') as fin:
        for i, txt_file_name in enumerate(fin, 1):
            txt_file_name = txt_file_name.strip()
            eb_antdoc = doc_to_ebantdoc(txt_file_name, work_dir, is_bespoke_mode,
                                        is_doc_structure=is_doc_structure)
            eb_antdoc_list.append(eb_antdoc)
    logging.debug('Finished doc_list_to_ebantdoc_list_linear()')

    return eb_antdoc_list


def doclist_to_ebantdoc_list_with_paras(doclist_file,
                                        work_dir,
                                        is_bespoke_mode=False,
                                        is_doc_structure=False):
    logging.debug('doclist_to_ebantdoc_list_with_paras(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)
    eb_antdoc_list = []
    paras_with_attrs_list = []
    paras_text_list = []
    with open(doclist_file, 'rt') as fin:
        for i, txt_file_name in enumerate(fin, 1):
            txt_file_name = txt_file_name.strip()
            eb_antdoc, paras_with_attrs, paras_text = \
                doc_to_ebantdoc_with_paras(txt_file_name,
                                           work_dir,
                                           is_bespoke_mode,
                                           is_doc_structure=is_doc_structure)
            eb_antdoc_list.append(eb_antdoc)
            paras_with_attrs_list.append(paras_with_attrs)
            paras_text_list.append(paras_text)
    logging.debug('Finished doc_list_to_ebantdoc_list_with_paras()')

    return eb_antdoc_list, paras_with_attrs_list, paras_text_list


#    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(list(txt_file_set), work_dir=work_dir)
def fnlist_to_fn_ebantdoc_map(fn_list, work_dir, is_doc_structure=False):
    logging.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}
    
    for i, txt_file_name in enumerate(fn_list, 1):
        eb_antdoc = doc_to_ebantdoc(txt_file_name, work_dir, is_doc_structure=is_doc_structure)
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

        eb_antdoc = doc_to_ebantdoc(txt_file_name, work_dir, is_doc_structure=is_doc_structure)
        
        fn_ebantdoc_map[txt_file_name] = EbAntdocProvSet(eb_antdoc)
    logging.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map


def print_ebantdoc(eb_antdoc):
    doc_text = eb_antdoc.text
    ts_col = 'TRAIN'
    if eb_antdoc.is_test_set:
        ts_col = 'TEST'
    # print("doc_sents_fn = {}".format(doc_sents_fn))
    for i, attrvec in enumerate(eb_antdoc.attrvec_list, 1):
        # print("attrvec = {}".format(attrvec))
        tmp_start = attrvec.start
        tmp_end = attrvec.end
        sent_text = doc_text[tmp_start:tmp_end].replace(r'[\n\t]', ' ')
        # sent_text = attrvec.bag_of_words
        labels_st = ""
        if attrvec.labels:
            labels_st = ','.join(sorted(attrvec.labels))
        cols = [str(i), '({}, {})'.format(tmp_start, tmp_end),
                ts_col, labels_st, sent_text, str(attrvec)]
        print('\t'.join(cols))

