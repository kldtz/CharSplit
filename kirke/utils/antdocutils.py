
import json
from operator import itemgetter
from typing import Any, List


def get_ant_out_json(json_fn: str) -> Any:
    with open(json_fn, 'rt') as handle:    
        ajson = json.load(handle)
    return ajson

def get_ant_out_file_prov_list(json_fn: str,
                               provision: str) -> List:
    ajson = get_ant_out_json(json_fn)
    return get_ant_out_json_prov_list(ajson, provision)


def get_ant_out_json_prov_list(ajson,
                               provision: str) -> List:
    prov_list = ajson['ebannotations'].get(provision, [])
    sorted_list = sorted(prov_list, key=itemgetter('start'))
    return sorted_list


def get_ant_out_file_lang(json_fn: str):
    ajson = get_ant_out_json(json_fn)    
    return get_ant_out_json_lang(ajson)


def get_ant_out_json_lang(ajson) -> str:
    return ajson.get('lang', 'None')


def get_ant_out_file_doccat(json_fn: str) -> str:
    ajson = get_ant_out_json(json_fn)        
    return get_ant_out_json_doccat(ajson)


def get_ant_out_json_doccat(ajson) -> str:
    return ajson.get('tags', 'None')
