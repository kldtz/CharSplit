import json
import re

from kirke.utils import strutils


def get_offsets_file_name(file_name: str):
    return file_name.replace('.txt', '.offsets.json')


# lineinfo has its data structure
# lineOffsets are just integers
# blockOffsets has 'start' and 'end' attribute
# pageOffsets has 'start' and 'end' attribute
def load_pdf_offsets(file_name: str):
    atext = strutils.loads(file_name)
    ajson = json.loads(atext)
    return (ajson.get('docLen'), ajson.get('strOffsets'), ajson.get('lineOffsets'),
            ajson.get('blockOffsets'), ajson.get('pageOffsets'))


# return the tuple (text, is_multi_lines)
def para_to_para_list(line):
    fake_line = ''
    if re.search("\n\s*\n", line):
        # print("weird double line in para..........")
        fake_line = re.sub('([\n\s]+)([\n\s][\n\s][\n\s])', r'\1'.replace('\n', ' ') + " XX253x", line)
        fake_line = fake_line.replace('\n', ' ').replace(' XX253x', ' \n\n')
        # print("fake line = {}".format(fake_line))

        # print("len(line) = {}, len(fake_line)= {}".format(len(line), len(fake_line)))
        return fake_line, True

    line_list = line.split('\n')
    max_line_len = 0
    num_notempty_line = 0
    for lx in line_list:
        lx_len = len(lx)
        if lx_len > max_line_len:
            max_line_len = lx_len
        if lx_len != 0:
            num_notempty_line += 1
    if num_notempty_line <= 1:
        return line, False  # not multi-line
    # print("line = [{}]".format(line))
    # print("max_line = {}".format(max_line_len))
    if max_line_len > 60:
        line = line.replace('\n', ' ')
        return line, False

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
        return line, False
    return line, num_notempty_line > 1
