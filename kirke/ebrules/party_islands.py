import re
from kirke.ebrules import addresses
from kirke.ebrules import titles


DATA_DIR = './dict/party_islands/'


"""Regexes."""


def word_regex(words):
    """Returns general regular expression for matching any word in words."""
    return re.compile(r'\b(?:{})\b'.format('|'.join(words)), re.I)


months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
month = word_regex(months)


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
suffix_regex = re.compile(r'(^.*?\b(?:{})\b)'.format('|'.join(suffixes)), re.I)
with open(DATA_DIR + 'first_names.list') as f:
    name_regex = word_regex(f.read().splitlines())
with open(DATA_DIR + 'agents.list') as f:
    agent_regex = word_regex(f.read().splitlines())


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
            if 'toc' in tags or 'skip_as_template' in tags:
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


"""Extract party islands from a file."""


parens = r'\(.*?\)'
colon = r':'
between = ['among', 'amongst', 'and', 'between']
between = r'\b(?:{})\b'.format('|'.join(between))
split_patterns = re.compile(r'({}|{}|{})'.format(parens, colon, between), re.I)


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
        suffix_matches = suffix_regex.findall(line['text'])
        if suffix_matches:
            parties.append((line['id'], line['start'],
                            line['start'] + len(suffix_matches[0])))
            continue
        if name_regex.search(line['text']) or agent_regex.search(line['text']):
            parties.append((line['id'], line['start'], line['end']))

    # Return parties list. Each party: (line id, start char, exclusive end char)
    return parties
