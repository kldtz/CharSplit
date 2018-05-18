from collections import namedtuple, Counter, defaultdict
from functools import total_ordering
import os
import sys
from typing import Any, DefaultDict, Dict, List, Tuple

from kirke.docstruct import jenksutils, docstructutils
from kirke.utils import engutils, strutils


class AbbyLine:

    def __init__(self,
                 text: str,
                 attr_dict: Dict) -> None:
        self.num = -1
        self.text = text
        self.attr_dict = attr_dict

        # this links to PDFBox's offset
        self.span_list = []
        self.pbox_line_ids = []

        self.infer_attr_dict = None

        
class AbbyPar:

    def __init__(self,
                 ab_lines: List[AbbyLine],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_lines = ab_lines
        self.attr_dict = attr_dict

        self.infer_attr_dict = None

    
class AbbyTextBlock:

    def __init__(self,
                 ab_pars: List[AbbyPar],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_pars = ab_pars
        self.attr_dict = attr_dict

        self.infer_attr_dict = None        

        
class AbbyTableBlock:

    def __init__(self,
                 table_block_num: int,
                 attr_dict: Dict) -> None:                 
        self.num = table_block_num
        self.attr_dict = attr_dict

class AbbyPage:

    def __init__(self,
                 ab_text_blocks: List[AbbyTextBlock],
                 ab_table_blocks: List[AbbyTextBlock],
                 attr_dict: Dict) -> None:                                  
        self.num = -1
        self.ab_text_blocks = ab_text_blocks
        self.ab_table_blocks = ab_table_blocks
        self.attr_dict = attr_dict

        self.infer_attr_dict = None
        
class AbbyXmlDoc:

    def __init__(self,
                 file_name: str,
                 ab_pages: List[AbbyPage]) -> None:
        self.file_id = file_name
        self.ab_pages = ab_pages

    def print_raw(self):
        for pnum, abby_page in enumerate(self.ab_pages):
            print("\n\npage #{} ========== {}".format(pnum, abby_page.attr_dict))
            for bid, ab_text_block in enumerate(abby_page.ab_text_blocks):
                print("\n    block #{} -------- {}".format(bid, ab_text_block.attr_dict))
                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    print("        par #{} {}".format(par_id, ab_par.attr_dict))
                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        print("            line #{} [{}] {}".format(lid, ab_line.text, ab_line.attr_dict))
                        # print("            line #{} [{}]".format(lid, ab_line.text, ab_line.attr_dict))

    def print_raw_lines(self):
        for pnum, abby_page in enumerate(self.ab_pages):
            print("\n\npage #{} ========== {}".format(pnum, abby_page.attr_dict))
            for bid, ab_text_block in enumerate(abby_page.ab_text_blocks):
                print("\n    block #{} -------- {}".format(bid, ab_text_block.attr_dict))
                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    print("        par #{} {}".format(par_id, ab_par.attr_dict))
                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        # print("            line #{} [{}] {}".format(lid, ab_line.text, ab_line.attr_dict))
                        print("            line #{} [{}]".format(lid, ab_line.text, ab_line.attr_dict))                        


    def print_text(self):
        for pnum, abby_page in enumerate(self.ab_pages):
            print("\n\npage #{} ========== {}".format(pnum, abby_page.infer_attr_dict))
            for bid, ab_text_block in enumerate(abby_page.ab_text_blocks):
                print("\n    block #{} -------- {}".format(bid, ab_text_block.infer_attr_dict))
                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    print("        par #{} {}".format(par_id, ab_par.infer_attr_dict))
                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        print("            line #{} [{}] {}".format(lid, ab_line.text, ab_line.infer_attr_dict))                        
                        
                        
        
