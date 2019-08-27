from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from kirke.utils import osutils, strutils, textoffset


# cannot use this because in line 600 prov_annotation.start = xxx in ebtext2antdoc.py
# maybe fix in future.
# ProvisionAnnotation = namedtuple('ProvisionAnnotation', ['start', 'end', 'label'])
# pylint: disable=R0903
class ProvisionAnnotation:
    __slots__ = ['start', 'end', 'label', 'score']

    def __init__(self, start: int, end: int, label: str, score: float = -1.0):
        self.start = start
        self.end = end
        self.label = label
        self.score = score

    def __repr__(self) -> str:
        if self.score != -1.0:
            return "ProvisionAnnotation({}, {}, '{}', score = {})".format(self.start,
                                                                          self.end,
                                                                          self.label,
                                                                          self.score)
        return "ProvisionAnnotation({}, {}, '{}')".format(self.start,
                                                          self.end,
                                                          self.label)


    def __lt__(self, other) -> Any:
        return (self.start, self.end) < (other.start, other.end)

    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()

    def to_tuple(self) -> Tuple[int, int, str, float]:
        return (self.start, self.end, self.label, self.score)

    # def to_tuple(self):
    #     return (self.lable, self.start, self.end)

# pylint: disable=R0902
class EbProvisionAnnotation:
    __slots__ = ['confidence', 'correctness', 'start', 'end',
                 'label', 'text', 'pid', 'custom_text']

    def __init__(self, ajson) -> None:
        self.confidence = ajson.get('confidence')
        self.correctness = ajson.get('correctness')
        self.start = ajson.get('start')
        self.end = ajson.get('end')
        self.label = ajson.get('type')
        self.text = ajson.get('text')
        self.pid = ajson.get('id')    # string, not 'id' but 'pid'
        self.custom_text = ajson.get('customText')  # boolean

    def to_dict(self) -> Dict[str, Any]:
        return {'confidence': self.confidence,
                'correctness': self.correctness,
                'customText': self.custom_text,
                'start': self.start,
                'end': self.end,
                'id': self.pid,
                'text': self.text,
                'type': self.label}

    def __repr__(self) -> str:
        return str(self.to_dict())

    def __str__(self) -> str:
        return self.__repr__()

    def to_tuple(self) -> ProvisionAnnotation:
        return ProvisionAnnotation(self.start, self.end, self.label)


class PredProvisionAnnotation:

    def __init__(self, ajson) -> None:
        self.label = ajson['label']
        self.start = ajson['start']
        self.end = ajson['end']
        self.text = ajson['text']
        self.prob = ajson.get('prob', -1.0)

    def __repr__(self) -> str:
        out_st = []
        out_st.append('label= "{}"'.format(self.label))
        out_st.append('start= {}'.format(self.start))
        out_st.append('end= {}'.format(self.end))
        out_st.append('prob={:.4f}'.format(self.prob))
        out_st.append('text= "{}"'.format(self.text.replace('\n', ' ')))
        return "{" + ', '.join(out_st) + "}"

    def __str__(self) -> str:
        return self.__repr__()


def load_pred_prov_ant(filename: str, provision_name: str) \
    -> List[PredProvisionAnnotation]:

    # logging.info('load provision %s annotation: [%s]', provision_name, filename)
    with open(filename, 'rt') as handle:
        parsed = json.load(handle)

    # had a coding error, sometimes the result might be stored inside
    # 'ebannotations', which the 'Extractor' will receive.
    # normalize that, make 'parsed' just have the prov_antlist_dict.
    parsed = parsed.get('ebannotations', parsed)

    # print('filenname: [{}]'.format(filename))
    # print('parsed:')
    # pprint.pprint(parsed)

    label_ant_list_map = defaultdict(list)  # type: Dict[str, List]
    for label, alist in parsed.items():
        for an_ant in alist:
            label_ant_list_map[label].append(PredProvisionAnnotation(an_ant))

    return label_ant_list_map[provision_name]


def load_prov_ant(filename: str, provision_name: Optional[str] = None) \
    -> List[EbProvisionAnnotation]:

    # logging.info('load provision %s annotation: [%s]', provision_name, filename)
    with open(filename, 'rt') as handle:
        parsed = json.load(handle)

    result = [EbProvisionAnnotation(ajson) for ajson in parsed]

    # if provision_name is specified, only return that specific provision
    if provision_name:
        return [provision_se for provision_se in result if provision_se.label == provision_name]

    return result


def load_prov_ebdata(filename: str, provision_name: Optional[str] = None) \
    -> Tuple[List[EbProvisionAnnotation], bool]:
    result = []  # type: List[EbProvisionAnnotation]
    is_test_set = False
    with open(filename, 'rt') as handle:
        parsed = json.load(handle)

    for _, ajson_list in parsed['ants'].items():
        # print("ajson_map: {}".format(ajson_map))
        for ajson in ajson_list:
            result.append(EbProvisionAnnotation(ajson))

    is_test_set = parsed.get('isTestSet', False)

    # if provision_name is specified, only return that specific provision
    if provision_name:
        return [provision_se for provision_se in result
                if provision_se.label == provision_name], is_test_set

    return result, is_test_set


# the result is a list of
# (start, end, ant_name)
def load_prov_annotation_list(txt_file_name: str,
                              cpoint_cunit_mapper: textoffset.TextCpointCunitMapper,
                              provision: Optional[str] = None) \
                              -> Tuple[List[ProvisionAnnotation], bool]:
    prov_ant_fn = txt_file_name.replace('.txt', '.ant')
    prov_ant_file = Path(prov_ant_fn)
    prov_ebdata_fn = txt_file_name.replace('.txt', '.ebdata')
    prov_ebdata_file = Path(prov_ebdata_fn)

    prov_annotation_list = []  # type: List[EbProvisionAnnotation]
    is_test = False
    if os.path.exists(prov_ant_fn):
        # in is_bespoke_mode, only the annotation for a particular provision
        # is returned.
        prov_annotation_list = (load_prov_ant(prov_ant_fn, provision)
                                if prov_ant_file.is_file() else [])

    elif os.path.exists(prov_ebdata_fn):
        prov_annotation_list, is_test = (load_prov_ebdata(prov_ebdata_fn, provision)
                                         if prov_ebdata_file.is_file() else ([], False))
    # else:
    #     raise ValueError

    # in-place update offsets
    result = []  # type: List[ProvisionAnnotation]
    for eb_prov_ant in prov_annotation_list:
        eb_prov_ant.start, eb_prov_ant.end = \
            cpoint_cunit_mapper.to_codepoint_offsets(eb_prov_ant.start,
                                                     eb_prov_ant.end)
        result.append(eb_prov_ant.to_tuple())

    return result, is_test

def get_ant_fn_list(txt_fname_list_fn: str) -> List[str]:
    txt_fn_list = strutils.load_str_list(txt_fname_list_fn)
    result = []  # type: List[str]
    for txt_file_name in txt_fn_list:
        prov_ant_fn = txt_file_name.replace('.txt', '.ant')
        # prov_ant_file = Path(prov_ant_fn)
        prov_ebdata_fn = txt_file_name.replace('.txt', '.ebdata')
        # prov_ebdata_file = Path(prov_ebdata_fn)
        if os.path.exists(prov_ant_fn):
            result.append(prov_ant_fn)
        elif os.path.exists(prov_ebdata_fn):
            result.append(prov_ebdata_fn)
    return result


def replace_custid_in_ant_list(cust_id: str,
                               human_readable_provision: str,
                               ajson: List[Dict[str, Any]]) -> None:
    for adict in ajson:
        type_val = adict.get('type')
        if type_val and type_val == cust_id:
            adict['type'] = human_readable_provision


def replace_custid_in_ebdata(cust_id: str,
                             human_readable_provision: str,
                             ajson: Dict[str, Any]) -> None:
    ants_val = ajson.get('ants', [])

    if ants_val:
        custid_ant_list = ants_val.get(cust_id, [])
        # in-place replacement
        replace_custid_in_ant_list(cust_id,
                                   human_readable_provision,
                                   custid_ant_list)
        ants_val[human_readable_provision] = custid_ant_list
        # remove the old annotation, the type values are inconsistent
        # in respect to ants_val[cust_id]
        del ants_val[cust_id]


def replace_custid_in_ant_fn_list(cust_id: str,
                                  human_readable_provision: str,
                                  txt_fname_list_fn: str) \
                                  -> None:
    timestamp = osutils.get_minute_timestamp_str()
    ant_fn_list = get_ant_fn_list(txt_fname_list_fn)
    renamed_ant_fn_list = [(ant_fn,
                            ant_fn + '.orig.' + timestamp)
                           for ant_fn in ant_fn_list]
    # the replacement of the custid with provision is performed on json,
    # not ProvisionAnnotation.  We don't want to deal with character offset issue

    for ant_fn, ant_orig in renamed_ant_fn_list:

        with open(ant_fn, 'rt') as handle:
            parsed = json.load(handle)

        if ant_fn.endswith('.ebdata'):
            replace_custid_in_ebdata(cust_id,
                                     human_readable_provision,
                                     parsed)
        elif ant_fn.endswith('.ant'):
            replace_custid_in_ant_list(cust_id,
                                       human_readable_provision,
                                       parsed)
        shutil.move(ant_fn, ant_orig)
        print("shutil.move({}, {})".format(ant_fn, ant_orig))
        strutils.dumps(json.dumps(parsed), ant_fn)


def make_copy_custid_in_ant_fn_list(cust_id: str,
                                    human_readable_provision: str,
                                    txt_fname_list_fn: str,
                                    work_dir: str) \
                                    -> str:
    """Make a copy the txt_fname_list_fn, with all cust_id replace.

    All the files in txt_fname_list_fn, will be copied to
    dir-work/human_readable_provision-timestamp/

    A new file with all those copied file names will be returned.
    """
    to_dir = '{}/{}_{}'.format(work_dir,
                               human_readable_provision,
                               osutils.get_minute_timestamp_str())
    osutils.mkpath(to_dir)
    out_fn_list = []  # type: List[str]

    fn_list = strutils.load_str_list(txt_fname_list_fn)
    for txt_file_name in fn_list:
        # copy txt file to destination directory
        shutil.copy(txt_file_name, to_dir)
        print("shutil.copy({}, {})".format(txt_file_name, to_dir))
        base_fname = os.path.basename(txt_file_name)
        out_fn_list.append('{}/{}'.format(to_dir, base_fname))

        # copy the annotation file, either .ebdata or .ant
        prov_ant_fn = txt_file_name.replace('.txt', '.ant')
        # prov_ant_file = Path(prov_ant_fn)
        prov_ebdata_fn = txt_file_name.replace('.txt', '.ebdata')
        # prov_ebdata_file = Path(prov_ebdata_fn)
        if os.path.exists(prov_ant_fn):
            print("shutil.copy({}, {})".format(prov_ant_fn, to_dir))
            shutil.copy(prov_ant_fn, to_dir)
        elif os.path.exists(prov_ebdata_fn):
            print("shutil.copy({}, {})".format(prov_ebdata_fn, to_dir))
            shutil.copy(prov_ebdata_fn, to_dir)

        # copy the offsets.json
        offsets_fn = txt_file_name.replace('.txt', '.offsets.json')
        if os.path.exists(offsets_fn):
            print("shutil.copy({}, {})".format(offsets_fn, to_dir))
            shutil.copy(offsets_fn, to_dir)

        # copy the pdf.xml
        pdfxml_fn = txt_file_name.replace('.txt', '.pdf.xml')
        if os.path.exists(pdfxml_fn):
            print("shutil.copy({}, {})".format(pdfxml_fn, to_dir))
            shutil.copy(pdfxml_fn, to_dir)

    fn_list_basename = os.path.basename(txt_fname_list_fn)
    out_fn_list_fn = '{}/{}'.format(to_dir, fn_list_basename)
    strutils.save_str_list(out_fn_list, out_fn_list_fn)

    replace_custid_in_ant_fn_list(cust_id,
                                  human_readable_provision,
                                  out_fn_list_fn)
    print("returning replaced fn: [{}]".format(out_fn_list_fn))

    return out_fn_list_fn


def ebdata_to_ant_json(ebdata_fname: str) -> List[Dict]:
    prov_ant_list, unused_is_test_set = load_prov_ebdata(ebdata_fname)

    ant_list = [prov_ant.to_dict() for prov_ant in prov_ant_list]
    return ant_list

def ebdata_to_ant_file(ebdata_fname: str, ant_fname: str) -> None:
    ant_list = ebdata_to_ant_json(ebdata_fname)
    with open(ant_fname, 'wt') as fout:
        print(json.dumps(ant_list), file=fout)
