import os
import re

# pylint: disable=unused-import
from typing import Any, Dict, List, Optional, Set, Tuple


# pylint: disable=too-few-public-methods
class ModelFileRecord:

    # pylint: disable=too-many-arguments
    def __init__(self,
                 file_name: str,
                 prov: str,
                 prov_ver: str,
                 version: int,
                 lang: str,
                 model_suffix: str) -> None:
        self.file_name = file_name
        self.prov_ver = prov_ver
        self.version = version
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


DEFAULT_MODEL_PREFIX_PAT = re.compile(r'^(.*)_(scut)')
# default provision names have no version information
MODEL_VERSION_PAT = re.compile(r'\.(\d+)')
SUFFIX_LANG = re.compile('^(.*)_([a-z][a-z])$')

def parse_default_model_file_name(file_name: str) -> Optional[ModelFileRecord]:
    mat = DEFAULT_MODEL_PREFIX_PAT.search(file_name)
    if mat:
        prov_name = mat.group(1)
        if prov_name.startswith('cust_') or \
           MODEL_VERSION_PAT.search(prov_name):
            return None
        model_suffix = file_name[mat.start(2):]

        lang_mat = SUFFIX_LANG.search(prov_name)
        found_lang = 'en'  # default
        # specific case for ea_reasonableness_rc
        if lang_mat and lang_mat.group(2) != 'rc':
            prov_name = lang_mat.group(1)
            found_lang = lang_mat.group(2)
        return ModelFileRecord(file_name,
                               prov_name,
                               prov_name,
                               -1,
                               found_lang,
                               model_suffix)
    return None


def get_model_file_names(dir_name: str) -> List[str]:
    """Return all the model file names in dir_name.

    There is a check on the file name pattern to decide if a file is
    the model files we wanted.
    """

    # it is intentional that the custom model file check is NOT done
    # here.  The logic for that is more complicated, so
    # simply return all files that can be a model, not just
    # default models.
    fnames = [f for f in os.listdir(dir_name)
              if (os.path.isfile(os.path.join(dir_name, f))
                  and not 'docclassifier' in f
                  and ('classifier' in f or
                       'annotator' in f)
                  and f.endswith('.pkl'))]
    return fnames


def get_default_model_records(dir_name: str) -> List[ModelFileRecord]:
    model_fnames = get_model_file_names(dir_name)

    record_list = []  # type: List[ModelFileRecord]
    for model_fname in model_fnames:
        model_rec = parse_default_model_file_name(model_fname)
        if model_rec:
            record_list.append(model_rec)
    return record_list


def get_default_model_file_names(dir_name: str) -> List[str]:
    model_records = get_default_model_records(dir_name)
    result = [model_rec.file_name for model_rec in model_records]  # type; List[str]
    return result


def get_default_provisions(dir_name: str) -> Set[str]:
    """This returns all provision names, without custom models."""

    model_records = get_default_model_records(dir_name)
    prov_set = set([model_rec.prov for model_rec in model_records])  # type: Set[str]
    return prov_set


CUSTOM_MODEL_PREFIX_PAT = re.compile(r'((cust_\d+)(\.(\d+))?)_(..)(.)')

def parse_custom_model_file_name(file_name: str) -> Optional[ModelFileRecord]:
    mat = CUSTOM_MODEL_PREFIX_PAT.match(file_name)
    if mat:
        prov_ver = mat.group(1)
        prov_name = mat.group(2)
        version_st = mat.group(4)
        lang = mat.group(5)
        after_lang = mat.group(6)
        if after_lang != '_':
            lang = 'en'
            model_suffix = file_name[mat.start(5):]
        else:
            model_suffix = file_name[mat.start(6)+1:]

        if version_st:
            version = int(version_st)
        else:
            version = -1

        return ModelFileRecord(file_name,
                               prov_name,
                               prov_ver,
                               version,
                               lang,
                               model_suffix)
    return None


def remove_custom_provision_version(prov_version: str) -> str:
    chunks = prov_version.split('.')
    return chunks[0]


def get_custom_model_file_names(dir_name: str) -> List[str]:
    """Return all the model file names in dir_name.

    There is a check on the file name pattern to decide if a file is
    the model files we wanted.
    """
    fnames = [f for f in os.listdir(dir_name)
              if (f.startswith('cust_') and
                  os.path.isfile(os.path.join(dir_name, f))
                  and not 'docclassifier' in f
                  and ('classifier' in f or
                       'annotator' in f)
                  and f.endswith('.pkl'))]
    return fnames


def get_custom_model_records(dir_name: str) -> List[ModelFileRecord]:
    model_fnames = get_custom_model_file_names(dir_name)

    record_list = []  # type: List[ModelFileRecord]
    for model_fname in model_fnames:
        model_rec = parse_custom_model_file_name(model_fname)
        if model_rec:
            record_list.append(model_rec)
    return record_list


# it seems that nobody is calling this
def get_custom_prov_versions(dir_name: str) -> Set[str]:
    """This returns all provision names with version."""

    model_records = get_custom_model_records(dir_name)
    cust_prov_set = set([model_rec.prov_ver
                         for model_rec in model_records])  # type: Set[str]
    return cust_prov_set


def get_custom_prov_ver_langs(dir_name: str) -> Set[str]:
    """This returns all provision names with version."""

    model_records = get_custom_model_records(dir_name)
    cust_prov_set = set([model_rec.get_prov_ver_lang()
                         for model_rec in model_records])  # type: Set[str]
    return cust_prov_set


# pylint: disable=too-many-locals, too-many-branches
def get_custom_model_files(dir_name: str,
                           cust_lang_provision_set: Set[str]) \
                           -> List[Tuple[str, str]]:
    """Return the list file names that matched cust_lang_provision_set.

    Return a list of tuples, with cust_lang_provision, model_file_name.
    The provision name is exactly the same as those from cust_lang_provision_set.
    """
    model_records = get_custom_model_records(dir_name)

    # print("cust_lang_provision_set: {}".format(cust_lang_provision_set))
    cust_lang_provision_fname_list = []  # type: List[Tuple[str, str]]
    for model_rec in model_records:
        model_lang_provision = model_rec.get_prov_ver_lang()
        if model_lang_provision in cust_lang_provision_set:
            cust_lang_provision_fname_list.append((model_lang_provision,
                                                   model_rec.file_name))
    # print("cust_lang_provision_fname_list: {}".format(cust_lang_provision_fname_list))
    return cust_lang_provision_fname_list


# pylint: disable=invalid-name
def get_provision_custom_model_files(dir_name: str,
                                     cust_provision: str) \
                                     -> List[Tuple[str, str]]:
    """Return the list file names that matched cust_provision.

    cust_provision does not have language specified, i.e., 'cust_12345.9393'
    This include all languages.

    Return a list of tuples, with lang_provision name, model_file_name.
    Please notice that the first str in the tuple has language id.
    """
    model_records = get_custom_model_records(dir_name)
    # print("cust_prov_set: {}".format(cust_prov_set))

    cust_lang_provision_fname_list = []  # type: List[Tuple[str, str]]
    for model_rec in model_records:
        if model_rec.prov_ver == cust_provision:
            cust_lang_provision_fname_list.append((model_rec.get_prov_ver_lang(),
                                                   model_rec.file_name))
    # print("cust_lang_provision_fname_list: {}".format(cust_lang_provision_fname_list))
    return cust_lang_provision_fname_list


def update_custom_prov_with_version(prov_list: List[str],
                                    dir_name) \
                                    -> List[str]:
    """Update a provision list with incomplete cust_* specification
       with version information for custom provisions.

    For example, "cust_9" will have the highest version in the dir_name.
    """

    custom_model_records = get_custom_model_records(dir_name)
    result = []  # type: List[str]

    for prov in prov_list:
        if prov.startswith('cust_'):
            prov_ver = find_highest_version(prov, custom_model_records)
            if prov_ver:
                result.append(prov_ver)
            else:
                # not found, kept the old provision
                result.append(prov)
        else:
            result.append(prov)
    return result


def find_highest_version(prov: str,
                         model_records: List[ModelFileRecord]) -> Optional[str]:
    top_ver = 0
    top_model_rec = None  # type: Optional[ModelFileRecord]
    for model_rec in model_records:
        if model_rec.prov == prov and \
           model_rec.version > top_ver:
            top_ver = model_rec.version
            top_model_rec = model_rec
    if top_model_rec:
        return top_model_rec.prov_ver
    return None
