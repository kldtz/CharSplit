
# pylint: disable=unused-import
from typing import Any, List, Dict, Tuple, Union

from kirke.docstruct import linepos

# pylint: disable=too-many-locals
# for pdftxtparser, the 2nd argument in paras_with_attrs is a List
# for abbyxmlparser, the 2nd attribute in paras_with_attrs is a Dict
# but this doesn't work, Union[Dict, List]]]
def print_paras_with_attrs(paras_with_attrs: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
                                                        Any]],
                           doc_text: str,
                           nlp_text: str,
                           out_file_name: str) -> None:

    with open(out_file_name, 'wt') as fout:
        for para_with_attrs in paras_with_attrs:
            lnpos_pair_list, unused_attrs = para_with_attrs

            for from_lnpos, to_lnpos in lnpos_pair_list:
                from_start, from_end, from_line_num, is_gap = from_lnpos.to_tuple()
                print('From: {:5d} {:5d} {} {}: [{}]'.format(from_start,
                                                             from_end,
                                                             from_line_num,
                                                             is_gap,
                                                             doc_text[from_start:from_end]),
                      file=fout)

                to_start, to_end, to_line_num, is_gap = to_lnpos.to_tuple()
                print('  To: {:5d} {:5d} {} {}: [{}]'.format(to_start,
                                                             to_end,
                                                             to_line_num,
                                                             is_gap,
                                                             nlp_text[to_start:to_end]),
                      file=fout)

            print("\n\n", file=fout)
