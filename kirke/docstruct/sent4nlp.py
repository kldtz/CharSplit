
# the goal here is to keep track of mapping from the text to be parsed by NLP module
# to the offsets given in the SentV2 (from Java).
# Here is the different levels of SentV1, SentV2, and Sent4Nlp.
#
# SentV1: offsets from PDFTextStripper.  Offsets that understand by both UI and backend.
#         For example, the TOC portion is messed up, so are the tables.  The line breaks
#         are generally unreliable.
# SentV2: structured document with better line breaks.  TOC is parsed correctly, so are
#         the page numbers and footers.  Regex is used to identify headers also.
#         Tables are identified using rules.
#         It contains offsets from SentV1.
# Sent4Nlp: Texts that will be sent to NLP modules.  Page numbers, footer, and non-text, which
#         includes tables without NL text, are stripped and will not be sent.  Contains new
#         offsets mapping.  It contains offsets from SentV2.

import sys

from kirke.docstruct.sentv2 import SentV2
from kirke.utils import strutils

DEBUG_MODE = False

class FromToSpan:

    # 'from' is sentNlp
    # 'to' is raw_text
    def __init__(self, from_start, from_end, to_start, to_end):    
        self.from_start = from_start
        self.from_end = from_end
        self.to_start = to_start
        self.to_end = to_end

    def to_dict(self):
        return {'from_start': self.from_start,
                'from_end': self.from_end,
                'to_start': self.to_start,
                'to_end': self.to_end}

    
class Sent4Nlp:

    def __init__(self, start, end, text, fromtospan, pagenum, category, is_english):
        self.start = start
        self.end = end
        self.text = text
        self.fromtospan = fromtospan
        self.pagenum = pagenum
        self.category = category
        self.is_english = is_english

    def __repr__(self):
        return "SentV4(({}, {}) page={} cat={}, eng={}, {} {})".format(self.start,
                                                                       self.end,
                                                                       self.pagenum,
                                                                       self.category,
                                                                       self.is_english,
                                                                       self.fromtospan,
                                                                       self.text)

# print to file only if file is not None
def debug_print(line, xfile=None):
    if xfile:
        print(line, file=xfile)


def page_sentV2s_list_to_sent4nlp_list(doc_text, page_sentV2s_list, file_name=None):
    result = []
    cur_offset = 0
    sentV4 = None

    para_counter = 1
    prev_para_count = 1
    is_last_sent_english = True
    is_last_sent_incomplete = False

    pv2_out = None
    if file_name:
        pv2_out = open(file_name, 'wt')

    prev_category = '---'
    for pagenum, page_sentV2s in enumerate(page_sentV2s_list, 1):

        debug_print('\n\n==== page #%d: ==========' % (pagenum,), xfile=pv2_out)
        is_first_para_in_page = True
        for sentV2 in page_sentV2s:
            category = sentV2.category

            if category == '---':
                if is_first_para_in_page and is_last_sent_incomplete:
                    para_counter -= 1
                is_first_para_in_page = False

                debug_print("para #{} {}".format(para_counter, sentV2.category), xfile=pv2_out)
                if prev_para_count != para_counter:
                    debug_print('', xfile=pv2_out)
                    result.append(Sent4Nlp(cur_offset, cur_offset, "", None,
                                           pagenum=pagenum, category=category,
                                           is_english=False))
                    cur_offset += 1

                for j, linfo in enumerate(sentV2.lineinfo_list, 1):
                    debug_print("\t\t| {}.{} |\t{}".format(para_counter, j, linfo.text), xfile=pv2_out)
                    start4 = cur_offset
                    end4 = start4 + len(linfo.text)
                    offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                    sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                      pagenum=pagenum, category=category,
                                      is_english=linfo.is_english)
                    result.append(sentV4)
                    cur_offset += (end4 - start4) + 1   # for '\n'

                    is_last_sent_english = linfo.is_english
                    if not linfo.text.strip():
                        # print("wow, empty string......", file=pv2_out)
                        continue
                    if linfo.text.strip() and linfo.text.strip()[-1] in set(['.', ':', ';', '!', '?']):
                        is_last_sent_incomplete = False
                    else:
                        is_last_sent_incomplete = True

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
            elif category in set(['signature', 'footer', 'pagenum', 'toc', 'exhibit', 'sechead']):
                # WARNING: at the end of a page, this will miss some newline breaks
                # when TOC + PAGENUM
                pass
            elif category in set(['table', 'graph']):
                for linfo in sentV2.lineinfo_list:
                    if linfo.is_english:
                        debug_print('', xfile=pv2_out)
                        result.append(Sent4Nlp(cur_offset, cur_offset, "", None,
                                               pagenum=pagenum, category=category,
                                               is_english=False))
                        cur_offset += 1

                        debug_print("\t\t{}\t{}".format(sentV2.category, linfo.text), xfile=pv2_out)
                        start4 = cur_offset
                        end4 = start4 + len(linfo.text)
                        offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                        # create a temporary sentV2 to satisfy SentV4's requirement of SentV2
                        sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                          pagenum=pagenum, category=category,
                                          is_english=linfo.is_english)
                        result.append(sentV4)
                        cur_offset += (end4 - start4) + 1   # for '\n'

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
                is_last_sent_incomplete = False
            elif category is None:
                debug_print(sentV2.text, xfile=pv2_out)
                print("None???\t\t{}".format(sentV2.text), file=sys.stderr)
            else:
                debug_print(sentV2.text, xfile=pv2_out)
                print("unknown2 {}\t\t{}".format(sentV2.category, sentV2.text), file=sys.stderr)
            prev_category = category

    if pv2_out:
        print("wrote {}".format(file_name))
        pv2_out.close()

    sentV4_st_list = []
    for i, sentV4 in enumerate(result):
        # print("{}\t{}".format(i, sentV4.text))
        sentV4_st_list.append(sentV4.text)

    textV4 = '\n'.join(sentV4_st_list)
    return textV4, result


def page_sentV2s_list_to_sent4lined_list(doc_text, page_sentV2s_list, file_name=None):
    result = []
    cur_offset = 0
    sentV4 = None

    para_counter = 1
    prev_para_count = 1
    is_last_sent_english = True
    is_last_sent_incomplete = False

    pv2_out = None
    if file_name:
        pv2_out = open(file_name, 'wt')

    for pagenum, page_sentV2s in enumerate(page_sentV2s_list, 1):

        debug_print('\n\n==== page #%d: ==========' % (pagenum,), xfile=pv2_out)
        is_first_para_in_page = True
        for sentV2 in page_sentV2s:
            category = sentV2.category

            if category == '---':
                if is_first_para_in_page and is_last_sent_incomplete:
                    para_counter -= 1
                is_first_para_in_page = False

                if prev_para_count != para_counter:
                    debug_print('', xfile=pv2_out)
                    result.append(Sent4Nlp(cur_offset, cur_offset, "", None,
                                           pagenum=pagenum, category=category,
                                           is_english=False))
                    cur_offset += 1

                for j, linfo in enumerate(sentV2.lineinfo_list, 1):
                    debug_print("\t\t| {}.{} |\t{}".format(para_counter, j, linfo.text), xfile=pv2_out)
                    start4 = cur_offset
                    end4 = start4 + len(linfo.text)
                    offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                    sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                      pagenum=pagenum, category=category,
                                      is_english=linfo.is_english)
                    result.append(sentV4)
                    cur_offset += (end4 - start4) + 1   # for '\n'

                    is_last_sent_english = linfo.is_english
                    if not linfo.text.strip():
                        # print("wow, empty string......", file=pv2_out)
                        continue
                    if linfo.text.strip() and linfo.text.strip()[-1] in set(['.', ':', ';', '!', '?']):
                        is_last_sent_incomplete = False
                    else:
                        is_last_sent_incomplete = True

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
            # elif category in set(['signature', 'toc', 'footer', 'pagenum', 'exhibit', 'sechead']):
            # pass
            # elif category in set(['table', 'graph']):
            elif category in set(['signature', 'toc', 'footer', 'pagenum',
                                  'table', 'graph',
                                  'exhibit', 'sechead']):
                for linfo in sentV2.lineinfo_list:
                    debug_print('', xfile=pv2_out)
                    result.append(Sent4Nlp(cur_offset, cur_offset, "", None,
                                           pagenum=pagenum, category=category,
                                           is_english=False))
                    cur_offset += 1

                    debug_print("\t\t{}\t{}".format(sentV2.category, linfo.text), xfile=pv2_out)
                    start4 = cur_offset
                    end4 = start4 + len(linfo.text)
                    offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                    # create a temporary sentV2 to satisfy SentV4's requirement of SentV2
                    sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                      pagenum=pagenum, category=category,
                                      is_english=linfo.is_english)
                    result.append(sentV4)
                    cur_offset += (end4 - start4) + 1   # for '\n'

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
                is_last_sent_incomplete = False
            elif category is None:
                debug_print(sentV2.text, xfile=pv2_out)
                print("None???\t\t{}".format(sentV2.text), file=sys.stderr)
            else:
                debug_print(sentV2.text, xfile=pv2_out)
                print("unknown3 {}\t\t{}".format(sentV2.category, sentV2.text), file=sys.stderr)

    if pv2_out:
        print("wrote {}".format(file_name))
        pv2_out.close()

    sentV4_st_list = []
    for i, sentV4 in enumerate(result):
        # print("jjj34\t{}\t{}".format(i, sentV4.text))
        sentV4_st_list.append(sentV4.text)

    textV4 = '\n'.join(sentV4_st_list)
    return textV4, result

class ParaNlp:

    def __init__(self, sentv4_list, sechead_list):
        self.sentv4_list = sentv4_list
        self.sechead_list = sechead_list


def para_list_to_paranlp_list(para_list, file_name=None):
    sechead_list = []
    prev_category = '---'
    result = []
    for para in para_list:
        category = para[0].category
        if category in set(['sechead', 'exhibit']):
            if prev_category != category:
                sechead_list = []
            sechead_list.extend(para)
        elif category == 'toc':
            if sechead_list:
                sechead_list = []
            continue
        elif category == '---':
            result.append(ParaNlp(para, sechead_list))
        elif category in set(['table', 'graph']):
            for sentV4 in para:
                if sentV4.is_english:
                    result.append(ParaNlp([sentV4], sechead_list))
        prev_category = category

    if file_name:
        # print("len(para_list) = {}".format(len(para_list)))
        with open(file_name, 'wt') as pv2_out:
            for i, paranlp in enumerate(result, 1):
                para = paranlp.sentv4_list
                sechead_list = paranlp.sechead_list
                sechead_words = []
                for sentv4 in sechead_list:
                    words = sentv4.text.split()
                    sechead_words.extend(words)
                print(file=pv2_out)
                for sentV4 in para:
                    print("para #{}, pg={}\t{}\t{}\t{}".format(i, sentV4.pagenum, sentV4.category, sentV4.text, sechead_words), file=pv2_out)
        print("wrote {}".format(file_name))

    nlp_lines = []
    out_sentV4_list = []
    for i, paranlp in enumerate(result, 1):
        para = paranlp.sentv4_list
        for sentV4 in para:
            nlp_lines.append(sentV4.text)
            out_sentV4_list.append(sentV4)
        nlp_lines.append('')
    nlp_text = '\n'.join(nlp_lines)

    return nlp_text, result

# a table is now a list of sentv4
def page_sentV2s_list_to_para_list(doc_text, page_sentV2s_list, file_name=None):
    result = []
    cur_offset = 0
    sentV4 = None

    para_counter = 1
    prev_para_count = 1
    is_last_sent_english = True
    is_last_sent_incomplete = False

    cur_para = []
    para_list = [cur_para]

    pv2_out = None
    if file_name:
        pv2_out = open(file_name, 'wt')

    for pagenum, page_sentV2s in enumerate(page_sentV2s_list, 1):
        is_first_para_in_page = True
        prev_category = '---'

        for sentV2 in page_sentV2s:
            category = sentV2.category

            if category == '---':
                if is_first_para_in_page and is_last_sent_incomplete:
                    para_counter -= 1
                is_first_para_in_page = False

                if prev_para_count != para_counter:
                    cur_para = []
                    para_list.append(cur_para)
                    result.append(Sent4Nlp(cur_offset, cur_offset, "", None,
                                           pagenum=pagenum, category=category,
                                           is_english=False))
                    cur_offset += 1

                for j, linfo in enumerate(sentV2.lineinfo_list, 1):
                    # start4 = cur_offset
                    # end4 = start4 + len(linfo.text)
                    start4 = linfo.start
                    end4 = linfo.end
                    offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                    sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                      pagenum=pagenum, category=category,
                                      is_english=linfo.is_english)
                    cur_para.append(sentV4)
                    result.append(sentV4)
                    cur_offset += (end4 - start4) + 1   # for '\n'

                    is_last_sent_english = linfo.is_english
                    if not linfo.text.strip():
                        # print("wow, empty string......", file=pv2_out)
                        continue
                    if linfo.text.strip() and linfo.text.strip()[-1] in set(['.', ':', ';', '!', '?']):
                        is_last_sent_incomplete = False
                    else:
                        is_last_sent_incomplete = True

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
                prev_category = category
            # elif category in set(['signature', 'toc', 'footer', 'pagenum', 'exhibit', 'sechead']):
            # pass
            # elif category in set(['table', 'graph']):
            elif category in set(['footer', 'pagenum']):
                # prev_category = category
                continue
            elif category in set(['signature', 'toc',
                                  'table', 'graph',
                                  'exhibit', 'sechead']):
                for linfo in sentV2.lineinfo_list:
                    if category == 'sechead' or prev_category != category:
                        result.append(Sent4Nlp(cur_offset, cur_offset, "", None,
                                               pagenum=pagenum, category=category,
                                               is_english=False))
                        cur_para = []
                        para_list.append(cur_para)
                        cur_offset += 1
                        prev_category = category  # already separated

                    # start4 = cur_offset
                    # end4 = start4 + len(linfo.text)
                    start4 = linfo.start
                    end4 = linfo.end
                    offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                    # create a temporary sentV2 to satisfy SentV4's requirement of SentV2
                    sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                      pagenum=pagenum, category=category,
                                      is_english=linfo.is_english)
                    cur_para.append(sentV4)
                    result.append(sentV4)
                    cur_offset += (end4 - start4) + 1   # for '\n'

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
                is_last_sent_incomplete = False
                prev_category = category
            elif category is None:
                print("None???\t\t{}".format(sentV2.text), file=sys.stderr)
            else:
                print("unknown4 {}\t\t{}".format(sentV2.category, sentV2.text), file=sys.stderr)

    if file_name:
        # print("len(para_list) = {}".format(len(para_list)))
        with open(file_name, 'wt') as pv2_out:
            for i, para in enumerate(para_list, 1):
                print(file=pv2_out)
                for sentV4 in para:
                    print("para #{}\tpg={}\t{}\t{}".format(i, sentV4.pagenum, sentV4.category, sentV4.text), file=pv2_out)
        print("wrote {}".format(file_name))


        with open(file_name + ".yyyy", 'wt') as pv2_out:
            for i, sentV4 in enumerate(result, 1):
                print("sentV4 #{}\tpg={}\t{}\t{}".format(i, sentV4.pagenum, sentV4.category, sentV4.text), file=pv2_out)
        print("wrote {}".format(file_name + ".yyyy"))

    # return para_list, result
    return para_list


def page_sentV2s_list_to_group_list(page_sentV2s_list, file_name=None):
    result = []

    para_counter = 1
    prev_para_count = 1
    is_last_sent_english = True
    is_last_sent_incomplete = False

    cur_para = []
    para_list = [cur_para]

    pv2_out = None
    if file_name:
        pv2_out = open(file_name, 'wt')

    for pagenum, page_sentV2s in enumerate(page_sentV2s_list, 1):
        is_first_para_in_page = True
        prev_category = '---'

        for sentV2 in page_sentV2s:
            category = sentV2.category

            if category == '---':
                if is_first_para_in_page and is_last_sent_incomplete:
                    para_counter -= 1
                is_first_para_in_page = False

                if prev_para_count != para_counter:
                    cur_para = []
                    para_list.append(cur_para)

                if sentV2.lineinfo_list:
                    last_linfo = sentV2.lineinfo_list[-1]
                    is_last_sent_english = last_linfo.is_english
                    if not last_linfo.text.strip():
                        # print("wow, empty string......", file=pv2_out)
                        continue
                    if last_linfo.text.strip()[-1] in set(['.', ':', ';', '!', '?']):
                        is_last_sent_incomplete = False
                    else:
                        is_last_sent_incomplete = True

                cur_para.append(sentV2)

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
                prev_category = category
            # elif category in set(['signature', 'toc', 'footer', 'pagenum', 'exhibit', 'sechead']):
            # pass
            # elif category in set(['table', 'graph']):
            elif category in set(['footer', 'pagenum']):
                # prev_category = category
                continue
            elif category in set(['signature', 'toc',
                                  'table', 'graph',
                                  'exhibit', 'sechead']):

                if category == 'sechead' or prev_category != category:
                    cur_para = []
                    para_list.append(cur_para)
                    prev_category = category  # already separated

                cur_para.append(sentV2)

                prev_para_count = para_counter
                #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                para_counter += 1
                is_last_sent_incomplete = False
                prev_category = category
            elif category is None:
                print("None???\t\t{}".format(sentV2.text), file=sys.stderr)
            else:
                print("unknown4 {}\t\t{}".format(sentV2.category, sentV2.text), file=sys.stderr)

    if file_name:
        # print("len(para_list) = {}".format(len(para_list)))
        with open(file_name, 'wt') as pv2_out:
            for i, para in enumerate(para_list, 1):
                print(file=pv2_out)
                for sentV2 in para:
                    print("para #{}\tpg={}\t{}\t{}".format(i, sentV2.pagenum, sentV2.category, sentV2.text), file=pv2_out)
        print("wrote {}".format(file_name))

    return para_list


def para_sentv2_to_text4nlp(para_sentv2_list, file_name=None):
    sentV4_list = []
    cur_offset, pagenum = 0, 0
    category = '---'
    is_prev_line_empty = True
    for para_sentv2 in para_sentv2_list:
        for sentV2 in para_sentv2:
            pagenum = sentV2.pagenum
            category = sentV2.category
            # for those categories, only print english ones
            # elif category in set(['signature', 'footer', 'pagenum', 'toc', 'exhibit', 'sechead']):            
            if category in set(['table', 'graph']):
                for linfo in sentV2.lineinfo_list:
                    if linfo.is_english:
                        if not is_prev_line_empty:
                            sentV4_list.append(Sent4Nlp(cur_offset, cur_offset, '', None,
                                                        pagenum=pagenum, category=category,
                                                        is_english=False))
                            cur_offset += 1
                            # is_prev_line_empty
                            
                        start4 = cur_offset
                        end4 = start4 + len(linfo.text)
                        offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                        # create a temporary sentV2 to satisfy SentV4's requirement of SentV2
                        sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                          pagenum=pagenum, category=category,
                                          is_english=linfo.is_english)
                        sentV4_list.append(sentV4)
                        cur_offset += (end4 - start4) + 1   # for '\n'
                        is_prev_line_empty = False
            elif category in set(['signature', 'toc', 'exhibit', 'sechead']):
                # pass
                continue
            else:
                for j, linfo in enumerate(sentV2.lineinfo_list, 1):
                    start4 = cur_offset
                    end4 = start4 + len(linfo.text)
                    offset_pairs = FromToSpan(start4, end4, linfo.start, linfo.end)
                    sentV4 = Sent4Nlp(start4, end4, linfo.text, offset_pairs,
                                      pagenum=pagenum, category=category,
                                      is_english=linfo.is_english)
                    sentV4_list.append(sentV4)
                    cur_offset += (end4 - start4) + 1   # for '\n'
                    is_prev_line_empty = False

        if not is_prev_line_empty:
            sentV4_list.append(Sent4Nlp(cur_offset, cur_offset, '', None,
                                        pagenum=pagenum, category=category,
                                        is_english=False))
            cur_offset += 1
            is_prev_line_empty = True
        
    text_st_list = []
    for sentV4 in sentV4_list:
        text_st_list.append(sentV4.text)

    return '\n'.join(text_st_list), sentV4_list
