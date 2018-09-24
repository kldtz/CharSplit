CORENLP_JSON_VERSION = '1.12'

def get_corenlp_json_fname(txt_basename: str, work_dir: str) -> str:
    base_fn = txt_basename.replace('.txt',
                                   '.corenlp.v{}.json'.format(CORENLP_JSON_VERSION))
    return '{}/{}'.format(work_dir, base_fn)


def get_nlp_file_name(txt_basename: str, work_dir: str) -> str:
    base_fn = txt_basename.replace('.txt', '.nlp.v{}.txt'.format(CORENLP_JSON_VERSION))
    return '{}/{}'.format(work_dir, base_fn)
