from kirke.docstruct import lxlineinfo
from kirke.docstruct.doctree import DTreeSegment

from kirke.utils import strutils

debug_mode = True

max_num_word = 6


def find_adjacent_short_lines(lineinfo_list, skip_lineinfo_set, min_line_size=4):
    result = []

    linfo_index = 0
    max_linfo_index = len(lineinfo_list)
    short_group_list = []
    cur_short_list = []
    is_prev_line_short = False
    while linfo_index < max_linfo_index:
        lineinfo = lineinfo_list[linfo_index]

        alpha_words = [word for word in lineinfo.words if strutils.is_alpha_word(word)]

        if lineinfo in skip_lineinfo_set:
            if cur_short_list and len(cur_short_list) > min_line_size:
                short_group_list.append(cur_short_list)
            cur_short_list = []
            is_prev_line_short = False
        elif len(alpha_words) <= max_num_word:
            cur_short_list.append(lineinfo)
            is_prev_line_short = True
        elif is_prev_line_short and len(alpha_words) <= max_num_word + 2:
            cur_short_list.append(lineinfo)
            is_prev_line_short = False
        else:    # more than 6 words
            if is_prev_line_short:
                if len(cur_short_list) >= min_line_size:
                    short_group_list.append(cur_short_list)
            else:
                if len(cur_short_list) >= min_line_size+1:
                    short_group_list.append(cur_short_list[:-1])  # the last line is not short
            cur_short_list = []
            is_prev_line_short = False
        linfo_index += 1

    for group_seq, short_group in enumerate(short_group_list, 1):
        debug_mode = False
        
        centered_total = 0
        for lineinfo in short_group:
            if lineinfo.align_label == 'CN':
                centered_total += 1

        x_start_1 = int(short_group[0].xStart)
        x_start_set = set(range(x_start_1-5, x_start_1+5))
        # print("x_start_set = {}".format(x_start_set))
        num_around_first = 0
        for lineinfo in short_group[1:]:
            if int(lineinfo.xStart) in x_start_set:
                num_around_first += 1
        if num_around_first / (len(short_group)-1) >= 0.8:
            if debug_mode:
                print('skipping group because same xStart')
                print("group #{}".format(group_seq))
                for lineinfo in short_group:
                    print("    {}".format(lineinfo))            
            continue

        if lxlineinfo.is_itemized_list(short_group):
            if debug_mode:
                print('skipping group because itemized list')
                print("group #{}".format(group_seq))
                for lineinfo in short_group:
                    print("    {}".format(lineinfo))
            continue

        if is_section_header(short_group):
            if debug_mode:
                print('skipping group because section headers')
                print("group #{}".format(group_seq))
                for lineinfo in short_group:
                    print("    {}".format(lineinfo))
            continue

                # 50% because of 4 lines, sometime centered text "article + heading, prev + post"
        if centered_total / len(short_group) >= .5:   # too centered, not graph
            if debug_mode:
                print('skipping group because too centered, probably title')
                print("group #{}".format(group_seq))
                for lineinfo in short_group:
                    print("    {}".format(lineinfo))
            continue

        if debug_mode:
            print("group #{}".format(group_seq))
            for lineinfo in short_group:
                print("    {}".format(lineinfo))
        result.append(short_group)

    return result

# take the last of tocline and secheadx_list
def extract_segment_start_end(segment_name, lineinfo_list, text):
    start = lineinfo_list[0].start
    pagenum = lineinfo_list[0].page
    for i, lineinfo in enumerate(lineinfo_list[1:]):
        if lineinfo.start < start:
            start = lineinfo.start
    end = lineinfo_list[-1].end
    for lineinfo in lineinfo_list[:-1]:
        if lineinfo.end > end:
            end = lineinfo.end
    obj = DTreeSegment(segment_name, start, end, pagenum, text[start:end])
    return obj, lineinfo_list


def extract_signature(lineinfo_list, text):
    segment_obj, out_linfo_list = extract_segment_start_end('signature', lineinfo_list, text)
    for lineinfo in out_linfo_list:
        lineinfo.category = 'signature'
    return segment_obj, out_linfo_list

def is_exhibit(lineinfo_list):
    count_exhibit_name = 0
    for lineinfo in lineinfo_list:
        if lineinfo and lineinfo.words and (lineinfo.words[0].lower() in ['exhibit', 'exmllit'] or
            'exhibit' in lineinfo.text.lower()):
            count_exhibit_name += 1
    return count_exhibit_name / len(lineinfo_list) >= 0.8

def extract_exhibit(lineinfo_list, text):
    segment_obj, out_linfo_list = extract_segment_start_end('exhibit', lineinfo_list, text)
    for lineinfo in out_linfo_list:
        lineinfo.category = 'exhibit'
    return segment_obj, out_linfo_list    

    
# take the last of tocline and secheadx_list
def extract_graph(lineinfo_list, text):
    # print("extract_graph:")
    # for lineinfo in lineinfo_list:
    #    print("\tlinfo: {}".format(lineinfo.text))
        
    first_i = 0
    for i, lineinfo in enumerate(lineinfo_list):
        if lineinfo.words and lineinfo.words[0].lower() == 'exhibit':
            first_i = i + 1
    if first_i < len(lineinfo_list)-1:  # minimal 2 lines
        lineinfo_list = lineinfo_list[first_i:]

    segment_obj, out_linfo_list =  extract_segment_start_end('graph', lineinfo_list, text)
    for lineinfo in out_linfo_list:
        lineinfo.category = 'graph'
    return segment_obj, out_linfo_list


# take the last of tocline and secheadx_list
def extract_whole_page(lineinfo_list, text):
    first_i = 0
    prev_y_start = 0
    for i, lineinfo in enumerate(lineinfo_list):
        # print("xxx ystart_diff = {}".format(lineinfo.yStart - prev_y_start))
        if lineinfo.yStart - prev_y_start >= 150.0:  # there is case, 'exhibit page-8'
            break

        if (lineinfo.words and lineinfo.words[0].lower() == 'exhibit' and
            i+1 < len(lineinfo_list) and lineinfo_list[i+1].align_label == 'CN'):
            first_i = i + 2
        elif lineinfo.words and lineinfo.words[0].lower() == 'exhibit':
            first_i = i + 1
        prev_y_start = lineinfo.yStart
    lineinfo_list = lineinfo_list[first_i:]

    # segment_obj, out_linfo_list =  extract_segment_start_end('whole-page', lineinfo_list, text)
    segment_obj, out_linfo_list =  extract_segment_start_end('table', lineinfo_list, text)
    for lineinfo in out_linfo_list:
        if lineinfo.category not in set(['pagenum', 'footer']):
            lineinfo.category = 'table'
    return segment_obj, out_linfo_list


def is_signature(lineinfo_list):
    count_title_name = 0
    for lineinfo in lineinfo_list:
        if (lineinfo.words and   # must have word
            lineinfo.words[0].lower() in ['name', 'by', 'title']):
            count_title_name += 1

    return count_title_name >= 2

def is_section_header(lineinfo_list):
    count_header_prefix = 0
    # count_lc_start = 0
    for lineinfo in lineinfo_list:
        if (lineinfo.words and   # must have a word
            lineinfo.words[0].lower() in ['article', 'section']):
            count_header_prefix += 1
        #if strutils.is_lc(lineinfo.words[0]):
        #    count_lc_start += 1

    return count_header_prefix >= 2 and len(lineinfo_list) <= 5



# all the checks have been done before
# such as distinct xStart values
def is_graph(lineinfo_list):
    return True


def find_graph_spans(page_lineinfos_list, skip_lineinfo_set, text):
    graphspan_results = []
    graphspan_lineinfo_results = []

    cur_graphspan = None
    for page_num, page_lineinfos in enumerate(page_lineinfos_list, 1):
        # if page_num == 44:
        #   print("hellolllll")

        # if page_num == 86:
        #    print("hellolllll")

        page_linfo_list = list(page_lineinfos)

        group_list = find_adjacent_short_lines(page_linfo_list, skip_lineinfo_set)

        # check if the group_list basically covered the whole page
        if group_list and lxlineinfo.is_word_overlap(group_list, page_linfo_list, skip_lineinfo_set, 0.75):
            # print("checking group overlap, pagenum = {}, {}".format(page_num, group_list))
            segment_obj, whole_lineinfo_list = extract_whole_page(page_linfo_list, text)
            graphspan_results.append(segment_obj)
            graphspan_lineinfo_results.append(whole_lineinfo_list)
            continue

        for lineinfo_group in group_list:
            if not lineinfo_group:
                print("empty group in page_num: {}".format(page_num))
                
            if is_signature(lineinfo_group):
                segment_obj, signature_lineinfo_list = extract_signature(lineinfo_group,
                                                                         text)
                graphspan_results.append(segment_obj)
                graphspan_lineinfo_results.append(signature_lineinfo_list)
            elif is_exhibit(lineinfo_group):
                segment_obj, exhibit_lineinfo_list = extract_exhibit(lineinfo_group,
                                                                     text)
                graphspan_results.append(segment_obj)
                graphspan_lineinfo_results.append(exhibit_lineinfo_list)                
            elif is_graph(lineinfo_group):
                segment_obj, graph_lineinfo_list = extract_graph(lineinfo_group,
                                                                 text)                
                graphspan_results.append(segment_obj)
                graphspan_lineinfo_results.append(graph_lineinfo_list)

    return graphspan_results, graphspan_lineinfo_results



"""
        # ybased_lineinfo_list = sorted(page_linfo_list, key=getLinfoYXstart)
        ybased_lineinfo_list = sorted(page_linfo_list, key=cmp_to_key(y_comparator))
        linfo_index = 0
        max_linfo_index = len(ybased_lineinfo_list)
        prevYStart = -1
        while linfo_index < max_linfo_index:
            lineinfo = ybased_lineinfo_list[linfo_index]
        
            if not lineinfo in skip_lineinfo_set:

                if lineinfo.is_close_prev_line:
                    linfo_index += 1
                    prevYStart = lineinfo.yStart
                    if cur_graphspan:
                        cur_graphspan.append_lineinfo(lineinfo)
                    continue

                # mainly for debugging purpose
                if linfo_index+1 < max_linfo_index:
                    next_lineinfo =  ybased_lineinfo_list[linfo_index+1]

                if (is_startswith_exhibit(lineinfo.text) and linfo_index+1 < max_linfo_index and
                    ybased_lineinfo_list[linfo_index + 1] not in skip_lineinfo_set):

                    maybe_text = lineinfo.text + '  ' + ybased_lineinfo_list[linfo_index + 1].text
                    guess_label, prefix, head_text = parse_sec_head(maybe_text)
                    is_top_graphspan, top_graphspan_num = verify_graphspan_prefix(prefix)
                    # we don't want '(a)'
                    if guess_label and '(' not in prefix and is_top_graphspan:
                        if is_startswith_exhibit(prefix):
                            guess_label = 'exhibit'
                        else:
                            guess_label = 'graphspan'
                        lineinfo.category = guess_label
                        ybased_lineinfo_list[linfo_index + 1].category = guess_label
                        cur_graphspan = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  ybased_lineinfo_list[linfo_index + 1].end,
                                                  prefix,
                                                  head_text,
                                                  maybe_text,
                                                  [lineinfo, ybased_lineinfo_list[linfo_index + 1]])
                        # print("helllo2222 {}".format(graphspan))
                        graphspan_results.append(cur_graphspan)
                        graphspan_lineinfo_results.append(lineinfo)
                        graphspan_lineinfo_results.append(ybased_lineinfo_list[linfo_index + 1])
                        linfo_index += 1  # we already used up one extra
                    else:
                        if cur_graphspan:
                            cur_graphspan.append_lineinfo(lineinfo)
                elif (lineinfo.is_center() and linfo_index+1 < max_linfo_index and
                    ybased_lineinfo_list[linfo_index+1].is_center() and
                    ybased_lineinfo_list[linfo_index + 1] not in skip_lineinfo_set):

                    maybe_text = lineinfo.text + '  ' + ybased_lineinfo_list[linfo_index + 1].text
                    guess_label, prefix, head_text = parse_sec_head(maybe_text)
                    is_top_graphspan, top_graphspan_num = verify_graphspan_prefix(prefix)
                    # we don't want '(a)'
                    if guess_label and '(' not in prefix and  is_top_graphspan:
                        if is_startswith_exhibit(prefix):
                            guess_label = 'exhibit'
                        else:
                            guess_label = 'graphspan'
                        lineinfo.category = guess_label
                        ybased_lineinfo_list[linfo_index + 1].category = guess_label
                        cur_graphspan = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  ybased_lineinfo_list[linfo_index + 1].end,
                                                  prefix,
                                                  head_text,
                                                  maybe_text,
                                                  [lineinfo, ybased_lineinfo_list[linfo_index + 1]])
                        # print("helllo2222 {}".format(graphspan))
                        graphspan_results.append(cur_graphspan)
                        graphspan_lineinfo_results.append(lineinfo)
                        graphspan_lineinfo_results.append(ybased_lineinfo_list[linfo_index + 1])
                        linfo_index += 1  # we already used up one extra
                    else:
                        if cur_graphspan:
                            cur_graphspan.append_lineinfo(lineinfo)

                            #else:
                    #    print("skipping hhhello: %s" %
                    #          lineinfo.text + ybased_lineinfo_list[linfo_index+1].text)

                else:   # at this point, we know it is not close to previous line
                    guess_label, prefix, head_text = parse_sec_head(lineinfo.text)
                    # for "toc", verify_graphspan_prefix will fail
                    is_top_graphspan, top_graphspan_num = verify_graphspan_prefix(prefix)
                    if guess_label == 'toc':
                        cur_graphspan = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  lineinfo.end,
                                                  prefix,
                                                  head_text,
                                                  lineinfo.text,
                                                  [lineinfo])
                        graphspan_results.append(cur_graphspan)
                        graphspan_lineinfo_results.append(lineinfo)
                    elif guess_label and '(' not in prefix and is_top_graphspan:
                        if is_startswith_exhibit(prefix):
                            guess_label = 'exhibit'
                        else:
                            guess_label = 'graphspan'
                        lineinfo.category = guess_label
                        cur_graphspan = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  lineinfo.end,
                                                  prefix,
                                                  head_text,
                                                  lineinfo.text,
                                                  [lineinfo])
                        graphspan_results.append(cur_graphspan)
                        graphspan_lineinfo_results.append(lineinfo)
                    else:
                        if cur_graphspan:
                            cur_graphspan.append_lineinfo(lineinfo)
            linfo_index += 1
            prevYStart = lineinfo.yStart
            prevPageNum = lineinfo.page
    return graphspan_results, graphspan_lineinfo_results
"""    
