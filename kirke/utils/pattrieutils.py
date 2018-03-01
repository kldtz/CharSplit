import re
from pattrie import pattrie3

# xxx = match_approximate('with a copy', 'With a copy to', 2)
def startswith_approximate(term: str, line: str, edist: int):
    # we intentionally also don't match "w ith a copy" because of
    # first word is 1 char
    # First try adding spaces
    tmp_str = r'\s*'.join(map(re.escape, list(term)))
    found_starts = list(re.finditer(tmp_str, line.lower()))
    # print("jjjjjjjjj234234234")
    # print("tmp_str = [{}]".format(tmp_str))
    # print("found_starts: {}".format(found_starts))

    if found_starts:
        # print('find_approximate([{}], [{}], {}, returned True1'.format(term, line, edist))
        return True

    term_defn = '{}_defn'.format(term)
    term_list = [(term, term_defn)]
    term_pattrie = pattrie3.PatTrie()
    term_pattrie.load_with_tuple2(term_list)

    # mat = term_pattrie.find_approximate(line.lower(), edist)
    mat = term_pattrie.substr_search(line.lower(), edist)

    if mat:
        print('find_approximate([{}], [{}], {}, returned True2'.format(term, line, edist))

    return mat
