
import re

# ?= lookahead assertion, doesn't consume
# ?: non-capturing
# checking if spaces followed by capitalized letter or uncap letter by cap letter
SENTENCE_DOES_NOT_CONTINUE = r'(?=\s+(?:[A-Z0-9].|[a-z][A-Z0-9]))'
# ?<! negative lookahead assertion, not preceded by \w\. or \w\w\.
NOT_FEW_LETTERS = r'(?<!\b[A-Za-z]\.)(?<!\b[A-Za-z]{2}\.)'
# not preceded by No\. or Nos\.
NOT_NUMBER = r'(?<!\bN(O|o)\.)(?<!\bN(O|o)(S|s)\.)'
# period followed by above patterns
REAL_PERIOD = r'\.' + SENTENCE_DOES_NOT_CONTINUE + NOT_FEW_LETTERS + NOT_NUMBER
FIRST_SENT = re.compile(r'(.*?' + REAL_PERIOD + ')')

# print("first_sent regex = {}".format(r'(.*?' + REAL_PERIOD + ')'))

def first_sentence(astr: str) -> str:
    """Trying to avoid sentence tokenizing since occurs before CoreNLP."""
    match = FIRST_SENT.search(astr)
    return match.group() if match else astr

