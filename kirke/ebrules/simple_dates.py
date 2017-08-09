import re


"""Substitute months with equal signs."""


months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
month_regexes = [(re.compile(r'\b{}\b'.format(m), re.IGNORECASE),
                  '=' * len(m)) for m in months]


def month_sub(s):
    for regex, equal_signs in month_regexes:
        s = regex.sub(equal_signs, s)
    return s


"""Extract dates. Dates: (line id, start, exclusive-end, effective-bool)."""


eff = re.compile(r'\beffective\b', re.IGNORECASE)
mdy = re.compile(r'={3,9}[^=\d]{,15}[\D12]\d[^=\d]{,15}(?:19|20)\d{2}')
dmy = re.compile(r'[\D12]\d[^=\d]{,15}={3,9}[^=\d]{,15}(?:19|20)\d{2}')


def extract_dates(filepath):
    dates = []
    with open(filepath) as f:
        lines = f.read().splitlines()

    # Find dates before party line and first English paragraph
    for i, line in enumerate(lines):
        tags = line.split('\t')[0].split('|')
        if 'party_line' in tags or 'first_eng_para' in tags:
            break

        # Substitute months
        after_first_bracket = ''.join(line.split('[')[1:])
        between_brackets = ''.join(after_first_bracket.split(']')[:-1])
        line = month_sub(between_brackets)

        # Record whether this is an effective date
        effective = (bool(eff.search(lines[i]))
                     or (i > 1
                         and bool(eff.search(' '.join(lines[i - 2:i]))))
                     or (i < len(lines) - 1
                         and bool(eff.search(' '.join(lines[i + 1:i + 3])))))

        # Find dates
        for match in mdy.finditer(line):
            dates.append((i, match.start(), match.end(), effective))
        for match in dmy.finditer(line):
            dates.append((i, match.start(), match.end(), effective))

    # Return result
    return dates if dates else None

