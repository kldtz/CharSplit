import re
import logging

from kirke.ebrules import addresses
from kirke.ebrules import titles


DATA_DIR = './dict/party_islands/'


"""Regexes."""


def word_regex_title_upper(words):
    """Returns general regular expression for matching any word in words."""
    words = [w.title() for w in words] + [w.upper() for w in words]
    return re.compile(r'\b(?:{})\b'.format('|'.join(words)))


months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
month = word_regex_title_upper(months)


# Source: https://stackoverflow.com/questions/123559/
phone = r'(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9]' \
        r'[02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))' \
        r'\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*' \
        r'(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'
phone = re.compile(r'\b{}\b'.format(phone), re.I)


email = re.compile(r'@')
currency = re.compile(r'\$.*\d|\d.*\bdollars\b', re.I)


with open(DATA_DIR + 'business_suffixes.list') as f:
    suffixes = f.read().splitlines()
suffixes = [s.title() for s in suffixes] + [s.upper() for s in suffixes]
suffix_regex = re.compile(r'^.*?\b(?:{})\b'.format('|'.join(suffixes)))


with open(DATA_DIR + '538_names.list') as f:
    name_regex = word_regex_title_upper(f.read().splitlines())
with open(DATA_DIR + 'agents.list') as f:
    agent_regex = word_regex_title_upper(f.read().splitlines())


"""Extract lines from a file."""


def extract_lines(filepath):
    # Get title
    title = titles.extract_title(filepath)
    title_start, title_end, title_end_char = title if title else (-1, -1, -1)

    # Initial pass of lines before party_line (if no p_l, before first_eng_para)
    lines = []
    num_lines_before_first_eng_para = 0
    with open(filepath, encoding='utf-8') as f:
        for i, line in enumerate(f):
            tags = line.split('\t')[0].split('|')
            if 'party_line' in tags:
                break
            if 'first_eng_para' in tags:
                num_lines_before_first_eng_para = len(lines)
            if 'toc' in tags:
                break
            if 'skip_as_template' in tags:
                continue

            # Isolate the text and check that it is not whitespace
            after_first_bracket = ''.join(line.split('[')[1:])
            line = ''.join(after_first_bracket.split(']')[:-1])
            if not line.strip():
                continue

            # Don't consider titles
            start_char = 0
            if title_start <= i < title_end:
                continue
            if i == title_end:
                if title_end_char == -1:
                    # Entire line is the title (which will not be considered)
                    continue
                line = line[title_end_char:]
                start_char = title_end_char

            # Don't consider dates, phone numbers, emails, addresses, & currency
            if (month.search(line) or phone.search(line) or email.search(line)
                or addresses.classify(line) > 30 or currency.search(line)):
                continue

            # Line is of interest; append to lines
            line = (line, i, start_char)
            lines.append(line)

        else:
            # No party line was found (loop did not break)
            lines = lines[:num_lines_before_first_eng_para]

    # Return the lines of interest
    return lines


def extract_lines_v2(paras_attr_list):
    # Get title
    # title = titles.extract_title(filepath)
    # title_start, title_end, title_end_char = title if title else (-1, -1, -1)

    # Initial pass of lines before party_line (if no p_l, before first_eng_para)
    lines = []
    num_lines_before_first_eng_para = 0
    offset = 0
    start_end_list = []

    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        line_st_len = len(line_st)
        start_end_list.append((offset, offset + line_st_len))
        offset += line_st_len + 1
    
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        if 'party_line' in para_attrs:
            break
        if 'first_eng_para' in para_attrs:
            num_lines_before_first_eng_para = len(lines)
        if 'toc' in para_attrs or 'skip_as_template' in para_attrs:
            continue

        # Isolate the text and check that it is not whitespace
        line = line_st
        if not line.strip():
            continue

        # Don't consider titles
        start_char = 0
        """
        if title_start <= i < title_end:
            continue
        if i == title_end:
            if title_end_char == -1:
                # Entire line is the title (which will not be considered)
                continue
            line = line[title_end_char:]
            start_char = title_end_char
        """

        # Don't consider dates, phone numbers, emails, addresses, & currency
        if (month.search(line) or phone.search(line) or email.search(line)
            or addresses.classify(line) > 30 or currency.search(line)):
            continue

        # Line is of interest; append to lines
        line = (line, i, start_char)
        lines.append(line)
    else:
        # No party line was found (loop did not break)
        lines = lines[:num_lines_before_first_eng_para]

    # Return the lines of interest
    return lines, start_end_list



"""Extract party islands from a file."""


parens = r'\(.*?\)'
colon = r':'
between = ['among', 'amongst', 'and', 'between']
between = r'\b(?:{})\b'.format('|'.join(between))
split_patterns = re.compile(r'({}|{}|{})'.format(parens, colon, between), re.I)

def extract_party_islands_offset(paras_attr_list):
    # Grab lines from the file
    extracted_lines, start_end_list = extract_lines_v2(paras_attr_list)

    lines = []
    for (line, id, start_char) in extracted_lines:
        # Split the line
        parts = split_patterns.split(line)

        # Add indices
        new_parts = []
        curr_index = start_char
        for part in parts:
            start = curr_index
            curr_index += len(part)
            new_parts.append((part, start, curr_index))

        # Strip spaces from each new_part, then add to lines
        for (text, start, end) in new_parts:
            if text.strip():
                while text[0] == ' ':
                    text = text[1:]
                    start += 1
                while text[-1] == ' ':
                    text = text[:-1]
                    end -= 1
                lines.append({'text': text, 'id': id,
                              'start': start, 'end': end})

    # For each line, keep if has a suffix (truncate), name, or agent
    parties = []
    for line in lines:
        if len(line['text'].split()) < 2:
            continue
        suffix_matches = agent_regex.finditer(line['text'])
        for match in suffix_matches:
            parties.append((line['id'], line['start'] + match.start(),
                            line['start'] + match.end()))
        agent_matches = agent_regex.finditer(line['text'])
        for match in agent_matches:
            parties.append((line['id'], line['start'] + match.start(),
                            line['start'] + match.end()))
        if name_regex.search(line['text']):
            parties.append((line['id'], line['start'], line['end']))

    out_list = []
    for line_num, start, end in parties:
        out_list.append((start_end_list[line_num][0] + start, start_end_list[line_num][0] + end))

    # Return parties list. Each party: (line id, start char, exclusive end char)
    return out_list


def extract_party_islands(filepath):
    lines = []
    for (line, id, start_char) in extract_lines(filepath):
        # Split the line
        parts = split_patterns.split(line)

        # Add indices
        new_parts = []
        curr_index = start_char
        for part in parts:
            start = curr_index
            curr_index += len(part)
            new_parts.append((part, start, curr_index))

        # Strip spaces from each new_part, then add to lines
        for (text, start, end) in new_parts:
            if text.strip():
                while text[0] == ' ':
                    text = text[1:]
                    start += 1
                while text[-1] == ' ':
                    text = text[:-1]
                    end -= 1
                lines.append({'text': text, 'id': id,
                              'start': start, 'end': end})

    # For each line, keep if has a suffix (truncate), name, or agent
    parties = []
    for line in lines:
        text = line['text']
        if len(text.split()) < 2 or any(c.isdigit() for c in text):
            continue
        suffix_matches = suffix_regex.findall(text)
        if suffix_matches:
            parties.append((line['id'], line['start'],
                            line['start'] + len(suffix_matches[0])))
            continue
        # agent_matches = agent_regex.finditer(line['text'])
        # for match in agent_matches:
        #     parties.append((line['id'], line['start'] + match.start(),
        #                     line['start'] + match.end()))
        if name_regex.search(text):
            parties.append((line['id'], line['start'], line['end']))

    # Return parties list. Each party: (line id, start char, exclusive end char)
    return parties

