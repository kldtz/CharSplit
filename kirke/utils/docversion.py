CORENLP_JSON_VERSION = '1.12'

def get_corenlp_json_fname(doc_id: str,
                           *,
                           nlptxt_md5: str,
                           work_dir: str) \
                           -> str:
    base_fn = '{}-{}.corenlp.v{}.txt'.format(doc_id, nlptxt_md5, CORENLP_JSON_VERSION)
    return '{}/{}'.format(work_dir, base_fn)


def get_nlp_file_name(doc_id: str,
                      *,
                      nlptxt_md5: str,
                      work_dir: str) \
                      -> str:
    base_fn = '{}-{}.nlp.v{}.txt'.format(doc_id, nlptxt_md5, CORENLP_JSON_VERSION)
    return '{}/{}'.format(work_dir, base_fn)
