import os
import re

# pylint: disable=unused-import
from typing import Any, Dict, List, Optional, Set, Tuple


def get_model_file_names(dir_name: str) -> Set[str]:
    """Return all the model file names in dir_name.

    There is a check on the file name pattern to decide if a file is
    the model files we wanted.
    """
    fnames = [f for f in os.listdir(dir_name)
              if (os.path.isfile(os.path.join(dir_name, f))
                  and ('classifier' in f or
                       'annotator' in f)
                  and f.endswith('.pkl'))]
    return fnames

def get_custom_model_file_names(dir_name: str) -> Set[str]:
    """Return all the model file names in dir_name.

    There is a check on the file name pattern to decide if a file is
    the model files we wanted.
    """
    fnames = [f for f in os.listdir(dir_name)
              if (f.startswith('cust_') and
                  os.path.isfile(os.path.join(dir_name, f))
                  and ('classifier' in f or
                       'annotator' in f)
                  and f.endswith('.pkl'))]
    return fnames


# pylint: disable=too-few-public-methods
class ModelFileRecord:

    def __init__(self,
                 prov: str,
                 prov_ver: str,
                 lang: str,
                 model_suffix: str) -> None:
        self.prov_ver = prov_ver
        self.prov = prov
        self.lang = lang
        self.model_suffix = model_suffix

    def to_normalize_name(self) -> str:
        prov_ver_lang = self.get_prov_ver_lang()
        return '{}_{}'.format(prov_ver_lang, self.model_suffix)

    def get_prov_ver_lang(self) -> str:
        if self.lang == 'en':
            return '{}'.format(self.prov_ver)
        return '{}_{}'.format(self.prov_ver, self.lang)

    def __str__(self):
        return 'ModelFileRecord(prov=%s, prov_ver=%s, lang=%s, model_suffix=%s)' % \
            (self.prov, self.prov_ver, self.lang, self.model_suffix)


CUSTOM_MODEL_PREFIX_PAT = re.compile(r'((cust_\d+)(\.(\d+))?)_(..)(.)')

def parse_custom_model_file_name(file_name: str) -> Optional[ModelFileRecord]:
    mat = CUSTOM_MODEL_PREFIX_PAT.match(file_name)
    if mat:
        prov_ver = mat.group(1)
        prov_name = mat.group(2)
        lang = mat.group(5)
        after_lang = mat.group(6)
        if after_lang != '_':
            lang = 'en'
            model_suffix = file_name[mat.start(5):]
        else:
            model_suffix = file_name[mat.start(6)+1:]

        return ModelFileRecord(prov_name,
                               prov_ver,
                               lang,
                               model_suffix)
    return None


def get_all_custom_prov_versions(dir_name: str) -> Set[str]:
    """This returns all provision names with version."""

    model_fnames = get_custom_model_file_names(dir_name)

    cust_prov_set = set([])  # type: Set[str]
    for model_fname in model_fnames:
        model_rec = parse_custom_model_file_name(model_fname)
        if model_rec:
            cust_prov_set.add(model_rec.prov_ver)
            # print("found: " + str(model_rec))
    return cust_prov_set


def get_all_custom_prov_ver_langs(dir_name: str) -> Set[str]:
    """This returns all provision names with version."""

    model_fnames = get_custom_model_file_names(dir_name)

    cust_prov_set = set([])  # type: Set[str]
    for model_fname in model_fnames:
        model_rec = parse_custom_model_file_name(model_fname)
        if model_rec:
            cust_prov_set.add('{}'.format(model_rec.get_prov_ver_lang()))
            # print("found: " + str(model_rec))
    return cust_prov_set


# pylint: disable=too-many-locals, too-many-branches
def get_custom_model_files(dir_name: str,
                           cust_idverlg_set: Set[str]) \
                           -> List[Tuple[str, str]]:
    """Return the list file names that matched cust_idverlg_set.

    Return a list of tuples, with cust_idverlg, model_file_name.
    The provision name is exactly the same as those from cust_idverlg_set.
    """
    model_fnames = get_custom_model_file_names(dir_name)

    # print("cust_idverlg_set: {}".format(cust_idverlg_set))

    cust_idverlg_fname_list = []  # type: List[Tuple[str, str]]
    for model_fname in model_fnames:
        model_rec = parse_custom_model_file_name(model_fname)
        if model_rec:
            # print("found: " + str(model_rec))

            model_idverlg = model_rec.get_prov_ver_lang()
            if model_idverlg in cust_idverlg_set:
                cust_idverlg_fname_list.append((model_idverlg, model_fname))

    print("cust_idverlg_fname_list: {}".format(cust_idverlg_fname_list))
    return cust_idverlg_fname_list


# pylint: disable=invalid-name
def get_provision_custom_model_files(dir_name: str,
                                     cust_id_ver: str) \
                                     -> List[Tuple[str, str]]:
    """Return the list file names that matched cust_id_ver.

    This include all languages.

    Return a list of tuples, with provision_name, model_file_name.
    The provision name is exactly the same as those from cust_prov_set.
    """
    model_fnames = get_custom_model_file_names(dir_name)

    # print("cust_prov_set: {}".format(cust_prov_set))

    cust_idverlg_fname_list = []  # type: List[Tuple[str, str]]
    for model_fname in model_fnames:
        model_rec = parse_custom_model_file_name(model_fname)
        if model_rec:
            # print("found: " + str(model_rec))

            if model_rec.prov_ver == cust_id_ver:
                cust_idverlg_fname_list.append((model_rec.get_prov_ver_lang(), model_fname))

    print("cust_idverlg_fname_list: {}".format(cust_idverlg_fname_list))
    return cust_idverlg_fname_list
