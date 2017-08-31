from typing import List

from collections import namedtuple, defaultdict
from kirke.utils import strutils

from kirke.docstruct import lxlineinfo
from kirke.docstruct.lxlineinfo import LineInfo

from kirke.utils import mathutils

SentV1 = namedtuple('SentV1', ['start', 'end', 'text'])

# sentV1 has the original line offsets in the text file
def text2SentV1_list(text: str) -> List[SentV1]:
    offset_v1 = 0
    sentV1_list = []
    for i, sent_text in enumerate(text.split('\n'), 1):
        offset_v1end = offset_v1 + len(sent_text)
        sentV1_list.append(SentV1(offset_v1, offset_v1end, sent_text))
        # print("sent #{}: {}".format(i, sentV1))
        offset_v1 = offset_v1end + 1  # for '\n'

    # now we have the correct offsets, remove empty sentV1
    sentV1_list = [sentV1 for sentV1 in sentV1_list if sentV1.text]
    return sentV1_list

            



class SentV2:

    def __init__(self, start, end, category, text, lineinfo_list, pagenum=None):
        self.start = start
        self.end = end
        if pagenum:
            self.pagenum = pagenum
        else:
            self.pagenum = -1   # not set yet
        self.category = category
        self.text = text
        self.lineinfo_list = lineinfo_list
        self.align_list = []

    def __repr__(self):
        return "SentV2({}, {}, {}, '{}', '{}', {}, {})".format(self.start,
                                                               self.end,
                                                               self.pagenum,
                                                               self.category,
                                                               self.text,
                                                               self.lineinfo_list,
                                                               self.align_list)

    def to_tuple(self):
        return (self.start, self.end, self.pagenum, self.category, self.text, self.lineinfo_list)

    def get_ystart(self):
        if self.lineinfo_list:
            return self.lineinfo_list[0].yStart
        return 0

    def update_align_list(self, is_relax_check=False):
        lineinfo_list = self.lineinfo_list
        if len(lineinfo_list) == 1:
            lineinfo = lineinfo_list[0]
            if lineinfo.align_label:
                self.align_list.append(lineinfo.align_label)
        else:
            aligned_set = set({})
            total_xstart = 0
            for lineinfo in lineinfo_list:
                aligned_set.add(lineinfo.align_label)
                total_xstart += lineinfo.xStart
            avg_xstart = total_xstart / len(lineinfo_list)
            # print("aligned_set = {}".format(aligned_set))
            if len(aligned_set) == 1 and next(iter(aligned_set)):  # not empty
                    self.align_list.append(next(iter(aligned_set)))
            elif len(lineinfo_list) > 1:
                if lineinfo.align_label:
                    self.align_list.append(lineinfo.align_label)

        

# because of inserted space as linebreak, sentV1 might have extra space
# at the end
def is_ends_off_by_1(sentV1, lineinfo):
    if sentV1.end == lineinfo.end + 1:
        # print("last sent1 char is '{}'".format(sentV1.text[-1]))
        return sentV1.text[-1] == ' '
    return False

# splited into multiple list with differnt categories
def split_group_lineinfo_list(lineinfo_list):
    cur_lineinfo_group = [lineinfo_list[0]]
    lineinfo_group_list = [cur_lineinfo_group]
    cur_category = lineinfo_list[0].category
    for lineinfo in lineinfo_list[1:]:
        if lineinfo.category == cur_category:
            cur_lineinfo_group.append(lineinfo)
        else:
            cur_lineinfo_group = [lineinfo]
            lineinfo_group_list.append(cur_lineinfo_group)
        cur_category = lineinfo.category

    return lineinfo_group_list



def enrich_style(sentV2_list, lineinfo_list):
    """Assign lineinfo_list to each sentV2 in sentV2_lists"""
    debug_mode = False

    max_lineinfo_index = len(lineinfo_list)
    lineinfo_index = 0
    if lineinfo_index < max_lineinfo_index:
        lineinfo = lineinfo_list[lineinfo_index]
    else:
        lineinfo = None

    max_sentV2_index = len(sentV2_list)
    sentV2_index = 0
    sentV2 = sentV2_list[sentV2_index]
    while sentV2_index < max_sentV2_index and lineinfo_index < max_lineinfo_index:
        while sentV2_index < max_sentV2_index and sentV2.category != '---':
            if debug_mode:
                print("\nenrich X")
                print("\tskipping sentV2: {}".format(sentV2))
            sentV2_index += 1
            if sentV2_index < max_sentV2_index:
                sentV2 = sentV2_list[sentV2_index]

        #if sentV2.start >= 11500:
        #    print("watchout 2")
        #    debug_mode = True

        if debug_mode:
            print("\nenrich X")
            print("\tsentV2: {}".format(sentV2))
            print("\tlineinfo: {}".format(lineinfo))

        if sentV2_index < max_sentV2_index:
            if (sentV2.start == lineinfo.start and
                (sentV2.end == lineinfo.end or is_ends_off_by_1(sentV2, lineinfo))):
                sentV2.lineinfo_list = [lineinfo]
                
                sentV2_index += 1
                if sentV2_index < max_sentV2_index:
                    sentV2 = sentV2_list[sentV2_index]
                lineinfo_index += 1
                if lineinfo_index < max_lineinfo_index:
                    lineinfo = lineinfo_list[lineinfo_index]
                if debug_mode:
                    print("good.  one lineinfo matched")
            # overlap, sentV2 is bigger
            elif sentV2.start <= lineinfo.start and sentV2.end >= lineinfo.end:
                tmp_lineinfo_list = []
                while lineinfo_index < max_lineinfo_index and lineinfo.end <= sentV2.end:
                    tmp_lineinfo_list.append(lineinfo)
                    if debug_mode:
                        print("adding jjj {}".format(lineinfo))
                    lineinfo_index += 1
                    # check if last one
                    if lineinfo_index < max_lineinfo_index:
                        # there is no way lineinfo_indx will exceed max_lineinfo_index
                        lineinfo = lineinfo_list[lineinfo_index]
                        # print("checking enrich: {}".format(lineinfo))

                sentV2.lineinfo_list = tmp_lineinfo_list
                # now much found everyone lineinfo for this sentV2
                sentV2_index += 1
                if sentV2_index < max_sentV2_index:
                    sentV2 = sentV2_list[sentV2_index]
            elif lineinfo.end < sentV2.start and strutils.is_all_spaces(lineinfo.text):
                if debug_mode:
                    print("skipping jjj {}".format(lineinfo))
                lineinfo_index += 1
                # check if last one
                if lineinfo_index < max_lineinfo_index:
                    # there is no way lineinfo_indx will exceed max_lineinfo_index
                    lineinfo = lineinfo_list[lineinfo_index]
                    # print("checking enrich: {}".format(lineinfo))                
            else:
                # it's possible that the lineinfo and sentV2 will be out of order
                # because of the way they are listed?.  For example
                # - 52 -
                # EXHIBIT D-1: Special Condition for XXX
                # Generally:
                # In above, EXHIBIT D-1 might be out of order
                print("**************** weird")
                debug_mode = True

    # update all align info for all sentV2
    for sentV2 in sentV2_list:
        if sentV2.start < 500:
            is_relax_check = True
        else:
            is_relax_check = False
        sentV2.update_align_list(is_relax_check)

from collections import OrderedDict

# jshaw, bug, xxxx
# 13555 lineinfo start
# the issue is with found_lineinfo_list, when there is a gap
# the merge will simply skip over it when covering before and after
# this cause mismatch in later stage.
# 'confidential information' was skipped for test2.txt
#
# we need acces to lineinfo_list because when there is a gap in overlap of sentV1,
# we need the right pointers to setup chopped sentV1.
# With only found_lineinfo_list, we cannot do it correctly.
def create_partial_sentV2s(text, found_lineinfo_list, lineinfo_list):
    #for lineinfo in found_lineinfo_list:
    #    print("found lineinfo: {}".format(lineinfo))

    #for i, sentV1 in enumerate(sentV1_list):
    #    print("rrr sentV1 #{}: {}".format(i, sentV1))

    # it's possible to have found_lineinfo_list overlap, due to
    # pagenum and segment_span, clean it up.
    # It's also possible to have tables overlap with something else.
    # Must make the found_lineinfo_list unique, otherwise, infinite loop
    found_lineinfo_list = list(OrderedDict.fromkeys(found_lineinfo_list))
    """
    uniq_found_lineinfo_list = []
    uniq_found_set = set([])
    for flinfo in found_lineinfo_list:
        if flinfo not in uniq_found_set:
            uniq_found_lineinfo_list.append(flinfo)
            uniq_found_set.add(flinfo)
        #else:
        #    print("found non-uniq-found_lineinfo_list: {}".format(flinfo))
    found_lineinfo_list = uniq_found_lineinfo_list
    """
    for nx in found_lineinfo_list:
       print("found_lineinfo: {}".format(nx))

    sentV1_list = text2SentV1_list(text)

    debug_mode = False

    sentV2_list = []
    max_lineinfo_index = len(found_lineinfo_list)
    lineinfo_index = 0
    # it's possible that the document has 1 page, thus no pagenum or footer
    if lineinfo_index < max_lineinfo_index:
        lineinfo = found_lineinfo_list[lineinfo_index]
    else:
        lineinfo = None

    max_sentV1_index = len(sentV1_list)
    sentV1_index = 0
    sentV1 = sentV1_list[sentV1_index]
    while sentV1_index < max_sentV1_index and lineinfo_index < max_lineinfo_index:
    # while sentV1_index < max_sentV1_index:

        ## TODO, jshaw, debug
        #if sentV1.start >= 12708 or lineinfo.start >= 13555:
        #    print("watchout")
        #    debug_mode = True
        # if sentV1.start >= 261462:
        #    print("watchout")
        #    debug_mode = True

        if debug_mode:
            print("\n\njjj sentV1: {}".format(sentV1))
            print("\njjj lineInfo: {}".format(lineinfo))

        # find matching lineinfo
        if sentV1.end <= lineinfo.start:
            if debug_mode:
                print("branch #1, less")
            sentV2_list.append(SentV2(sentV1.start, sentV1.end, "---", sentV1.text, []))
            sentV1_index += 1
            sentV1 = sentV1_list[sentV1_index]
        elif sentV1.start == lineinfo.start and sentV1.end == lineinfo.end:
            if debug_mode:
                print("branch #2, same")
            sentV2_list.append(SentV2(sentV1.start, sentV1.end, lineinfo.category,
                                      sentV1.text, [lineinfo]))
            lineinfo_index += 1
            if lineinfo_index < max_lineinfo_index:
                lineinfo = found_lineinfo_list[lineinfo_index]
            sentV1_index += 1
            if sentV1_index < max_sentV1_index:
                sentV1 = sentV1_list[sentV1_index]
        else:  # lineinfo greater or overlap
            # if mathutils.start_end_overlap((sentV1.start, sentV1.end),
            #                               (lineinfo.start, lineinfo.end)):
            # it must be sentV1 covers lineinfo

            if debug_mode:
                print("branch #3, overlap")
                
            if sentV1.start <= lineinfo.start and sentV1.end >= lineinfo.end:
                latest_lineinfo = None
                while (lineinfo_index < max_lineinfo_index and lineinfo.end <= sentV1.end and
                           (not latest_lineinfo or lineinfo.sid == latest_lineinfo.sid + 1)):

                    # if lineinfo overlap doesn't start from beginning of sentV1.start
                    # create a SentV2 first.  The rest of the procedure is the same
                    if not latest_lineinfo and sentV1.start < lineinfo.start:
                        prev_notfound_lineinfo = lineinfo_list[lineinfo.sid - 1]
                        reverse_prev_lineinfo_list = []
                        while prev_notfound_lineinfo.start >= sentV1.start:
                            reverse_prev_lineinfo_list.append(prev_notfound_lineinfo)
                            if prev_notfound_lineinfo.sid == 0:
                                break
                            prev_notfound_lineinfo = lineinfo_list[prev_notfound_lineinfo.sid - 1]
                        # there is this case sentV1 = '    1541 xxxx'
                        #                  lineinfo = '1541'
                        # going to simply skip this
                        if reverse_prev_lineinfo_list:
                            notfound_lineinfo_list = list(reversed(reverse_prev_lineinfo_list))
                            prev_notfound_lineinfo = notfound_lineinfo_list[-1]
                            sentV2_list.append(SentV2(sentV1.start, prev_notfound_lineinfo.end, "---",
                                                      sentV1.text[:prev_notfound_lineinfo.end - sentV1.start],
                                                      notfound_lineinfo_list))

                    #if debug_mode:
                    #    if lineinfo.start == 192079:
                    #        print("helllo 23534")
                    #    print("adding jjj {}".format(lineinfo))
                    sentV2_list.append(SentV2(lineinfo.start, lineinfo.end, lineinfo.category,
                                              lineinfo.text, [lineinfo]))
                    latest_lineinfo = lineinfo
                    lineinfo_index += 1
                    # check if last one
                    if lineinfo_index < max_lineinfo_index:
                        # there is no way lineinfo_indx will exceed max_lineinfo_index
                        lineinfo = found_lineinfo_list[lineinfo_index]

                        
                # now take whatever is left
                if latest_lineinfo.end == sentV1.end or is_ends_off_by_1(sentV1, latest_lineinfo):
                    # print("lastest_lininfoxxx: {}".format(latest_lineinfo))
                    # print("sentv1oxxx: {}".format(sentV1))
                    sentV1_index += 1
                    if sentV1_index < max_sentV1_index:
                        sentV1 = sentV1_list[sentV1_index]
                # handle case at end of file
                #   sentv1 = "sdfa   "
                # lineinfo = "sdfa"
                elif latest_lineinfo.end < sentV1.end and line_matched_sentv1_gt(sentV1, latest_lineinfo):
                    sentV1_index += 1
                    if sentV1_index < max_sentV1_index:
                        sentV1 = sentV1_list[sentV1_index]
                elif latest_lineinfo.end < sentV1.end:
                    next_notfound_lineinfo = lineinfo_list[latest_lineinfo.sid + 1]
                    sentV1 = SentV1(next_notfound_lineinfo.start, sentV1.end,
                                    sentV1.text[next_notfound_lineinfo.start - sentV1.start:])
                else:
                    print("*************************** Error here 223*********")
                    print("sentV1: {}".format(sentV1))
                    print("lineInfo: {}".format(lineinfo))
            elif sentV1.start == lineinfo.start and sentV1.end <= lineinfo.end and line_matched_x4(sentV1, lineinfo):
                sentV2_list.append(SentV2(lineinfo.start, lineinfo.end, lineinfo.category,
                                          lineinfo.text, [lineinfo]))
                latest_lineinfo = lineinfo
                lineinfo_index += 1
                # check if last one
                if lineinfo_index < max_lineinfo_index:
                    # there is no way lineinfo_indx will exceed max_lineinfo_index
                    lineinfo = found_lineinfo_list[lineinfo_index]
                sentV1_index += 1
                if sentV1_index < max_sentV1_index:
                    sentV1 = sentV1_list[sentV1_index]
                    # print("added pad: {}".format(latest_lineinfo.end - sentV1.end))
                    # extra_pad = latest_lineinfo.end - sentV1.end
                    # sentV1 = sentV1_list[sentV1_index]
                    # sentV1 = SentV1(sentV1.start + extra_pad, sentV1.end,
                    #                 sentV1.text[extra_pad:])
            else:
                print("*************************** Error here 2244*********")
                print("sentV1: {}".format(sentV1))
                print("lineInfo: {}".format(lineinfo))

    # sentV1 might be in an intermediate state (half copied)
    if sentV1_index < max_sentV1_index:
        last_pagenum = lineinfo_list[-1].page
        sentV2_list.append(SentV2(sentV1.start, sentV1.end, "---", sentV1.text, []))
        for sentV1 in sentV1_list[sentV1_index+1:]:
            sentV2_list.append(SentV2(sentV1.start, sentV1.end, "---", sentV1.text, []))

    return sentV2_list


# because of following situation
#   sentv1: 'xxx'
# lineinfo: 'xxx\n '
def line_matched_x4(sentv1, lineinfo):
    eo_spaces = lineinfo.text[sentv1.end - lineinfo.end:]
    return strutils.is_all_spaces(eo_spaces)

def line_matched_sentv1_gt(sentv1, lineinfo):
    eo_spaces = sentv1.text[lineinfo.end - sentv1.end:]
    return strutils.is_all_spaces(eo_spaces)

def to_page_sentV2s_list(sentV2_list):

    pagenum_sentV2s_map = defaultdict(list)
    max_pagenum = -1
    for sentV2 in sentV2_list:
        sentV2_pagenum = sentV2.lineinfo_list[0].page
        sentV2.pagenum = sentV2_pagenum
        pagenum_sentV2s_map[sentV2_pagenum].append(sentV2)
        if sentV2_pagenum > max_pagenum:
            max_pagenum = sentV2_pagenum
    result = []
    for pagenum in range(1, max_pagenum+1):
        result.append(pagenum_sentV2s_map[pagenum])
    return result


def to_page_sentV2s_list_has_gap(sentV2_list):
    curpage_sentV2s = []
    result = [curpage_sentV2s]  # list of list
    prev_page_num = -1
    for sentV2 in sentV2_list:
        lineinfo_list_0 = sentV2.lineinfo_list[0]
        if prev_page_num != -1 and lineinfo_list_0.page != prev_page_num:
            curpage_sentV2s = []
            result.append(curpage_sentV2s)
        curpage_sentV2s.append(sentV2)
        sentV2.pagenum = lineinfo_list_0.page
        prev_page_num = lineinfo_list_0.page
    return result


def sentV2_between_start_end(start, end, sentV2_list):
    result = []
    for sentV2 in sentV2_list:
        if mathutils.start_end_overlap((start, end), (sentV2.start, sentV2.end)):
            result.append(sentV2)
    return result


#def fix_sentV2_list_in_segment(sentV2_list, segment_name, start, end):
#    for sentV2 in sentV2_list:
#        if mathutils.start_end_overlap((start, end), (sentV2.start, sentV2.end)):
#            # sentV2.category = 'segment-' + segment_name
#            sentV2.category = 's2222222egment-' + segment_name


def init_sentV2s(text: str,
                 lineinfo_list: List[LineInfo],
                 pagenum_list: List[LineInfo],
                 footer_list: List[LineInfo],
                 tocline_list: List[LineInfo],
                 sechead_lineinfo_list: List[LineInfo],
                 segment_lineinfo_list: List[LineInfo]) -> List[SentV2]:

    found_lineinfo_list = list(pagenum_list)
    found_lineinfo_list.extend(footer_list)
    found_lineinfo_list.extend(tocline_list)
    found_lineinfo_list.extend(sechead_lineinfo_list)
    found_lineinfo_list.extend(segment_lineinfo_list)

    sentV2_list = create_partial_sentV2s(text,
                                         sorted(found_lineinfo_list),
                                         lineinfo_list)

    categorized_lineinfo_set = set(found_lineinfo_list)
    uncategorized_lineinfo_list = [lineinfo for lineinfo in lineinfo_list
                                   if lineinfo not in categorized_lineinfo_set]

    # print("len(categorized_lineinfo_set) = ", len(categorized_lineinfo_set))
    # print("len(uncategorized_lineinfo_set) = ", len(uncategorized_lineinfo_list))
    # print("len(sentV2_list) = ", len(sentV2_list))
    
    # destructive movify sentV2_list
    enrich_style(sentV2_list, uncategorized_lineinfo_list)

    paged_sentV2s_list = to_page_sentV2s_list(sentV2_list)

    return sentV2_list, paged_sentV2s_list


def save_paged_sentv2s(file_name, paged_sentV2s_list):
    para_counter = 1
    is_last_sent_english = True
    is_last_sent_incomplete = False
    
    with open(file_name, 'wt') as pv2_out:
        for pagenum, page_sentV2s in enumerate(paged_sentV2s_list, 1):

            print('\n\n==== page #%d: ==========' % (pagenum,), file=pv2_out)
            is_first_para_in_page = True 
            for sentV2 in page_sentV2s:
                category = sentV2.category

                if category == '---':
                    if is_first_para_in_page and is_last_sent_incomplete:
                        para_counter -= 1
                    is_first_para_in_page = False
                        
                    print("para #{} {}".format(para_counter, sentV2.category), file=pv2_out)
                    for j, linfo in enumerate(sentV2.lineinfo_list, 1):
                        print("\t\t| {}.{} |\t{}".format(para_counter, j, linfo.text), file=pv2_out)
                        is_last_sent_english = linfo.is_english
                        if not linfo.text.strip():
                            print("wow, empty string......", file=pv2_out)
                        if linfo.text.strip() and linfo.text.strip()[-1] in set(['.', ':', ';', '!', '?']):
                            is_last_sent_incomplete = False
                        else:
                            is_last_sent_incomplete = True
                    #if is_last_sent_incomplete:
                    #    print("\t\t| incomplete |", file=pv2_out)
                    #else:
                    #    print("\t\t| para_complete |", file=pv2_out)
                    
                    #if len(sentV2.lineinfo_list) > 2 and not is_last_sent_incomplete:
                    para_counter += 1
                elif category == 'signature':
                    print("{}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)
                elif category == 'toc':
                    print("{}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)
                elif category == 'table':
                    print("{}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)
                elif category == 'graph':
                    print("{}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)                    
                elif category == 'footer' or category == 'pagenum':
                    print("{}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)
                elif category == 'exhibit':
                    print("\n{}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)
                elif category == 'sechead':
                    print("\n{}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)                    
                elif category is None:
                    print("None???\t\t{}".format(sentV2.text), file=pv2_out)
                else:
                    print("unknown1 {}\t\t{}".format(sentV2.category, sentV2.text), file=pv2_out)
                    
        print("wrote {}".format('paged_sentv2cs.tsv'))
                        
                        

