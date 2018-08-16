import json
import re
from typing import List, Tuple

from kirke.utils import strutils
from kirke.utils.textoffset import TextCpointCunitMapper


def get_offsets_file_name(file_name: str):
    return file_name.replace('.txt', '.offsets.json')


# lineinfo has its data structure
# lineOffsets are just integers
# blockOffsets has 'start' and 'end' attribute
# pageOffsets has 'start' and 'end' attribute
def load_pdf_offsets(file_name: str, cpoint_cunit_mapper: TextCpointCunitMapper):
    atext = strutils.loads(file_name)
    ajson = json.loads(atext)

    # in-place update the offsets
    # print("cpoint_cunit_mapper.max_cunit = {}, docLen = {}".format(cpoint_cunit_mapper.max_cunit,
    #                                                                ajson['docLen']))
    # ajson['docLen'] = cpoint_cunit_mapper.to_codepoint_offset(ajson['docLen'])
    if not ajson.get('docLen'):
        ajson['docLen'] = len(atext)
    else:
        ajson['docLen'] = cpoint_cunit_mapper.to_codepoint_offset(ajson['docLen'])
    for adict in ajson.get('blockOffsets'):
        adict['start'], adict['end'] = cpoint_cunit_mapper.to_codepoint_offsets(adict['start'],
                                                                                adict['end'])
    for adict in ajson.get('lineOffsets'):
        adict['offset'] = cpoint_cunit_mapper.to_codepoint_offset(adict['offset'])
    for adict in ajson.get('pageOffsets'):
        adict['start'], adict['end'] = cpoint_cunit_mapper.to_codepoint_offsets(adict['start'],
                                                                                adict['end'])
    for adict in ajson.get('strOffsets'):
        adict['start'], adict['end'] = cpoint_cunit_mapper.to_codepoint_offsets(adict['start'],
                                                                                adict['end'])

    return (ajson.get('docLen'), ajson.get('strOffsets'), ajson.get('lineOffsets'),
            ajson.get('blockOffsets'), ajson.get('pageOffsets'))


# pylint: disable=too-many-branches, too-many-locals, too-many-return-statements
def para_to_para_list(line: str) -> Tuple[str, bool, List[int]]:
    """Convert a multi-line into one line or keep as is.

    Input: line is a line with new-line breaks, if there are new line break in the input text.

    returns text: a non-break line, or a multi-line
            is_multi_lines: bool
            not-linebreak_offsets: [] if is_multi-lines is True, else
                                   offsets in line that should not be '\n'
    """

    # TODO, jshaw, but not urgent because this change will invalid all corenlp cache.
    # logically, converting 3+ new lines below to spaces is not really correct.
    # If there are more than 2 nlb (new line breaks), that should mean a new paragraph.
    # Should preserve those instead of 3+.
    # The best transformation would be \s+\n+ to put all the nlb together.  Now, it
    # might not distribute them in easy detectable manner since we take them as they are.
    if re.search(r'\n\s*\n\s*\n', line):  # there must be a reason for 3 nl-breaks
        mat_list = list(re.finditer(r'\n\s*\n\s*\n', line))
        # replace all of those double new lines with space to preserve them
        ch_list = list(line)
        for mat in mat_list:
            len_mat = len(mat.group())
            mat_start = mat.start()
            for i in range(len_mat):
                ch_list[mat_start + i] = ' '
        triple_line = ''.join(ch_list)
        # print("tmp_line: [{}]".format(tmp_line))
        not_linebreak_offsets = strutils.find_all_indices('\n', triple_line)

        # create the tmp_line
        ch_list = list(line)
        for i in not_linebreak_offsets:
            ch_list[i] = ' '
        tmp_line = ''.join(ch_list)

        if not_linebreak_offsets:
            return tmp_line, False, not_linebreak_offsets
        return tmp_line, True, []

    line_list = line.split('\n')
    max_line_len = 0
    num_notempty_line = 0
    for lxx in line_list:
        lx_len = len(lxx)
        if lx_len > max_line_len:
            max_line_len = lx_len
        if lx_len != 0:
            num_notempty_line += 1
    if num_notempty_line <= 1:
        return line, False, []  # not multi-line
    # print("line = [{}]".format(line))
    # print("max_line = {}".format(max_line_len))
    not_linebreak_offsets = strutils.find_all_indices('\n', line)
    if max_line_len > 60:
        line = line.replace('\n', ' ')
        return line, False, not_linebreak_offsets

    # num_period = line.count('.')
    nonl_line = line.replace('\n', ' ')
    words = nonl_line.split(' ')
    num_cap, num_lc, num_other = 0, 0, 0
    num_words = len(words)
    for word in words:
        if word:
            if word[0].islower():
                num_lc += 1
            elif word[0].isupper():
                num_cap += 1
            else:
                num_other += 1
    # print("num_lc {} / num_words {} = {}".format(num_lc, num_words, num_lc / float(num_words)))
    if num_words >= 8 and num_lc / float(num_words) >= 0.7:
        line = nonl_line
        return line, False, not_linebreak_offsets

    if num_notempty_line > 1:
        # is_multi_line is True, not_linebreak_offset is []
        return line, True, []
    return line, False, not_linebreak_offsets
