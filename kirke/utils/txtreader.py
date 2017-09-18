
import os
import re
from typing import List

# change all nbsp to regular spaces
def loads(file_name):
    with open(file_name, 'rt', newline='') as fin:
        doc = fin.read().replace('\xa0', ' ')
    return doc


def dumps(doc_text, file_name):
    with open(file_name, 'wt') as fout:
        fout.write(doc_text)


def load_lines_with_offsets(file_name: str):
    offset = 0
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            orig_length = len(line)
            # remove the eoln char            
            # replace non-breaking space with space for regex benefit
            new_line = line[:-1].replace('\xa0', ' ')
            end = offset + orig_length - 1   # remove eoln

            yield offset, end, new_line
                
            offset += orig_length


# return list of page offsets, and list of list of line offsets
def load_page_lines_with_offsets(file_name: str):
    doc_text = loads(file_name)
    paged_text_list = doc_text.split(chr(12))  # pdftotext use ^L as page marker

    # if the last one is empty, remove it
    if len(paged_text_list[-1]) == 0:
        paged_text_list = paged_text_list[:-1]

    page_offsets = []
    offset = 0
    for paged_text in paged_text_list:
        page_len = len(paged_text)
        end = offset + page_len
        page_offsets.append((offset, end))
        offset = end + 1  # for ^L

    page_list = []
    for (page_start, page_end), paged_text in zip(page_offsets, paged_text_list):
        offset = page_start
        paged_line_list = []
        for line in paged_text.split('\n'):
            line_len = len(line)
            end = offset + line_len
            paged_line_list.append((offset, end, line))

            offset = end + 1  # for eoln
        page_list.append(paged_line_list)
    return page_offsets, page_list


BE_SPACE_PAT = re.compile('^(\s*)(.*)$')

# remove all begin and end spaces for lines
# 'be' = begin_end
def load_normalized_lines_with_offsets(file_name: str):
    from_offset = 0
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            orig_length = len(line)
            # remove the eoln char            
            # replace non-breaking space with space for regex benefit
            new_line = line[:-1].replace('\xa0', ' ')

            if new_line.strip():
                mat = BE_SPACE_PAT.match(new_line)
                no_be_space_line = mat.group(2).strip()
                len_no_be_space_line = len(no_be_space_line)
                from_start = from_offset + len(mat.group(1))
                from_end = from_offset + len_no_be_space_line
                yield from_start, from_end, no_be_space_line
            else:
                yield from_offset, from_offset, ''

            from_offset += orig_length
            

# remove all begin and end spaces for lines
# 'be' = begin_end
def load_lines_with_fromto_offsets(file_name: str):
    from_offset = 0
    to_offset = 0
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            orig_length = len(line)
            # remove the eoln char            
            # replace non-breaking space with space for regex benefit
            new_line = line[:-1].replace('\xa0', ' ')

            if new_line.strip():
                mat = BE_SPACE_PAT.match(new_line)
                no_be_space_line = mat.group(2).strip()
                len_no_be_space_line = len(no_be_space_line)
                from_start = from_offset + len(mat.group(1))
                from_end = from_offset + len_no_be_space_line
                to_end = to_offset + len_no_be_space_line
                yield (from_start, from_end), (to_offset, to_end), no_be_space_line
                to_offset += len_no_be_space_line
            else:
                yield (from_offset, from_offset), (to_offset, to_offset), ''

            from_offset += orig_length
            to_offset += 1  # eoln
            

def load_str_list(file_name):
    st_list = []
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            st_list.append(line[:-1])
    return st_list


def save_str_list(str_list: List[str], file_name: str) -> None:
    with open(file_name, 'wt') as fout:
        for line in str_list:
            fout.write(line)
            fout.write(os.linesep)

EO_SENT_CHARS = set(['.', '?', '!', ':', ';', '_'])


def is_separate_line_text(doc_text: str) -> bool:
    lines = doc_text.split('\n')
    num_lines = len(lines)
    num_zerio, num_gt_110, num_gt_120, num_le_110 = 0, 0, 0, 0
    num_empty_line, num_start_lc, num_end_period = 0, 0, 0
    for line in lines:
        line = line.strip()

        if not line:
            num_empty_line += 1
        elif len(line) > 110:
            num_gt_110 += 1
        elif len(line) > 120:
            num_gt_120 += 1
        else:
            num_le_110 += 1

        if line:
            if line[0].islower():
                num_start_lc += 1
            if line[-1] in EO_SENT_CHARS:
                num_end_period += 1

    num_not_empty_line = num_lines - num_empty_line
    # print('num_start_lc= {}, perc= {}'.format(num_start_lc, num_start_lc / num_not_empty_line))
    # print('num_end_period= {}, perc= {}'.format(num_end_period, num_end_period / num_not_empty_line))

    if num_empty_line / num_lines > 0.4:
        return True

    if num_le_120 == 0:
        return True

    if num_start_lc / num_not_empty_line >= 0.5:
        return True

    if num_end_period / num_not_empty_line < 0.2:
        return True

    return False

# this is not deployed in the code yet
"""
atext = txtreader.loads('seplines.txt')
print("is_separate_line_text = {}".format(txtreader.is_separate_line_text(atext)))

blines, doc_text = txtreader.de_separate_lines(atext)
txtreader.dumps(doc_text, 'seplines.not.txt')
for xline in blines:
    xstart, xend, line = xline
    if line:
        print("{}\t{}\t[{}]".format(xstart, xend, line))
    else:
        print()
"""
def de_separate_lines(atext: str):
    from_offset = 0
    to_offset = 0
    lines = atext.split('\n')
    all_lines = []
    for line in lines:
        orig_length = len(line)
        # assume .replace('\xa0', ' ')
        # remove oeln
        new_line = line[:-1]

        if new_line.strip():
            mat = BE_SPACE_PAT.match(new_line)
            no_be_space_line = mat.group(2).strip()
            len_no_be_space_line = len(no_be_space_line)
            from_start = from_offset + len(mat.group(1))
            from_end = from_offset + len_no_be_space_line
            all_lines.append((from_start, from_end, no_be_space_line))
        else:
            all_lines.append((from_offset, from_offset, ''))

        from_offset += orig_length

        # remove empty lines

    result = []
    prev_line = ''
    prev_non_empty_line = 'This is NOT empty'
    to_offset = 0
    for xstart, xend, line in all_lines:
        if line:
            result.append(((xstart, xend), (to_offset, to_offset + len(line)), line))
            to_offset += len(line) + 1
        else:  # this is an empty empty line
            # previous line is also empty line, add the break
            # if last char is a period, a sentence break
            if prev_non_empty_line[-1] in EO_SENT_CHARS:
                result.append(((xstart, xend), (to_offset, to_offset + len(line)), line))
                to_offset += 1
            # prev_line is long and ends in lc
            elif len(prev_non_empty_line) > 70 and prev_non_empty_line[-1].islower():
                pass
            # maybe a section header
            elif len(prev_non_empty_line) < 50 and prev_non_empty_line[-1].isupper():
                result.append(((xstart, xend), (to_offset, to_offset + len(line)), line))
                to_offset += 1
            elif not prev_line:
                result.append(((xstart, xend), (to_offset, to_offset + len(line)), line))
                to_offset += 1
            else:
                # don't bother do line break
                pass
        prev_line = line
        if line:
            prev_non_empty_line = line

    lines_only = [line for _, _, line in result]
    return result, '\n'.join(lines_only)
