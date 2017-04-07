
import re
from collections import OrderedDict

import sys

def annotate_for_enhanced_ner(text_as_string):
    
    def transform_text(raw_text, hash_patterns):
        regex = re.compile("(%s)" % "|".join(hash_patterns.keys()), re.IGNORECASE)

        def inplace_str_sub(match):
            match_str = match.string[match.start():match.end()]
            for pat in hash_patterns:
                tmp_regex = re.compile(pat, re.IGNORECASE)
                if tmp_regex.search(match_str):
                    return tmp_regex.sub(hash_patterns[pat], match_str)
                
            return ""
        
        return regex.sub(inplace_str_sub, raw_text) 
    
    expressions = OrderedDict(
        [(r",(\W*)(inc)(\W)", r" \1Inc\3"), 
         (r"(\W)(inc)(\W)", r"\1Inc\3"), 
         (r"(\W)(ltd)(\W)", r"\1Ltd\3"),
         (r",(\W*)(llc)(\W)", r" \1Llc\3"),
         (r"(\W)(llc)(\W)", r"\1Llc\3"),
         (r",(\W*)(corp)(\W)", r" \1Corp\3"),
         (r"(\W)(corp)(\W)", r"\1Corp\3")])
    
    return transform_text(text_as_string, expressions)


EXPRESSIONS = OrderedDict(
    [(r",(\W*)(inc)\b", r" \1Inc"), 
     (r"(\W)(inc)\b", r"\1Inc"), 
     (r"(\W)(ltd)\b", r"\1Ltd"),
     (r",(\W*)(llc)\b", r" \1Llc"),
     (r"(\W)(llc)\b", r"\1Llc"),
     (r",(\W*)(corp)\b", r" \1Corp"),
     (r"(\W)(corp)\b", r"\1Corp")])

def annotate_for_enhanced_ner2(text_as_string):
    
    def transform_text(raw_text, hash_patterns):
        regex = re.compile("(%s)" % "|".join(hash_patterns.keys()), re.IGNORECASE)

        def inplace_str_sub(match):
            match_str = match.string[match.start():match.end()]
            for pat in hash_patterns:
                tmp_regex = re.compile(pat, re.IGNORECASE)
                if tmp_regex.search(match_str):
                    return tmp_regex.sub(hash_patterns[pat], match_str)
                
            return ""
        
        return regex.sub(inplace_str_sub, raw_text) 
    
    return transform_text(text_as_string, EXPRESSIONS)


INC_PAT = re.compile(r"(,\W*|\W)(inc|llc|corp|ltd)\b" , re.IGNORECASE)

def inplace_str_sub(match):
    if match.group(2).lower() == 'ltd':
        return match.group(1) + match.group(2).capitalize()
    else:
        return match.group(1).replace(',', ' ') + match.group(2).capitalize()

def annotate_for_enhanced_ner3(text_as_string):
    return INC_PAT.sub(inplace_str_sub, text_as_string)
