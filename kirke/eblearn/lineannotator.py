import logging

from typing import Any, List, Tuple


from kirke.utils import ebantdoc4, evalutils

from kirke.docstruct import fromtomapper, htmltxtparser, linepos
from kirke.ebrules import parties

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LineAnnotator:

    def __init__(self, provision, prov_annotator):
        self.provision_annotator = prov_annotator
        self.provision = provision
        self.eval_status = {}  # this is set after training
        self.threshold = 0.2


    def get_eval_status(self):
        return self.eval_status


    # ProvisionAnnotator does not train, it only predict
    # Training is available only for classifiers
    # def train(self):
    #    pass
    # pylint: disable=R0914
    def test_antdoc_list(self, ebantdoc_list: List[ebantdoc4.EbAnnotatedDoc4]):
        logger.debug('lineannotator.test_antdoc_list')

        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0

        for ebantdoc in ebantdoc_list:
            paras_with_attrs = ebantdoc.paras_with_attrs
            paras_text = ebantdoc.get_nlp_text()

            fromto_mapper = fromtomapper.FromToMapper('an offset mapper',
                                                      ebantdoc.get_nlp_sx_lnpos_list(),
                                                      ebantdoc.get_origin_sx_lnpos_list())

            print("test_antdoc_list(), at {}".format(ebantdoc.file_id))
            ant_list = self.annotate_antdoc(paras_with_attrs,
                                            paras_text,
                                            fromto_mapper,
                                            ebantdoc.get_nl_text())
            # print("88234 ant_list = {}".format(ant_list))
            # for ant in ant_list:
            #     print("ant: {}".format(ant))
            print('ebantdoc.fileid = {}'.format(ebantdoc.file_id))
            # print("ant_list: {}".format(ant_list))
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.provision]

            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            if self.provision == 'title':
                xtp, xfn, xfp, xtn, unused_log_json = \
                    evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                     ant_list,
                                                                     ebantdoc.file_id,
                                                                     ebantdoc.get_text())
            else:
                xtp, xfn, xfp, xtn, _, unused_log_json = \
                    evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                            ant_list,
                                                            ebantdoc.file_id,
                                                            ebantdoc.get_text(),
                                                            self.threshold,
                                                            is_raw_mode=True)

            tp += xtp
            fn += xfn
            fp += xfp
            tn += xtn

        title = "annotate_status, line-based"
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'prec': prec, 'recall': recall, 'f1': f1}}
        return tmp_eval_status


    # pylint: disable=too-many-branches
    def annotate_antdoc(self,
                        paras_with_attrs: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
                                                     List[Any]]],
                        paras_text: str,
                        fromto_mapper: fromtomapper.FromToMapper,
                        nl_text: str):
        prov_annotations = []
        if self.provision == 'party':
            paras_attr_list = htmltxtparser.lineinfos_paras_to_attr_list(paras_with_attrs, nl_text)
            party_offset_pair_list = \
                self.provision_annotator.extract_provision_offsets(paras_attr_list,
                                                                   paras_text)

            if party_offset_pair_list:
                for i, party_offset_pair in enumerate(party_offset_pair_list, 1):
                    party_ox, term_ox = party_offset_pair
                    if party_ox:
                        party_start, party_end = party_ox
                        party_st = paras_text[party_start:party_end]
                        prov_annotations.append({'end': party_end,
                                                 'label': self.provision,
                                                 'id': i,
                                                 'start': party_start,
                                                 'prob': 0.91,
                                                 'text': party_st})
                    if term_ox:
                        term_start, term_end = term_ox
                        term_st = paras_text[term_start:term_end]
                        prov_annotations.append({'end': term_end,
                                                     'label': self.provision,
                                                     'id': i,
                                                     'start': term_start,
                                                     'prob': 0.91,
                                                     'text': term_st})
            fromto_mapper.adjust_fromto_offsets(prov_annotations)

        elif self.provision == 'date':
            paras_attr_list = htmltxtparser.lineinfos_paras_to_attr_list(paras_with_attrs, nl_text)
            # prov_type can be 'date', 'effective-date', 'signature-date'
            date_list = self.provision_annotator.extract_provision_offsets(paras_attr_list,
                                                                           paras_text)

            if date_list:
                for i, date_ox in enumerate(date_list, 1):
                    start_offset, end_offset, unused_date_st, prov_type, date_norm = date_ox
                    prov_annotations.append({'end': end_offset,
                                             'label': prov_type,
                                             'start': start_offset,
                                             'prob': 0.91,
                                             'norm': date_norm,
                                             'text': paras_text[start_offset:end_offset]})
            fromto_mapper.adjust_fromto_offsets(prov_annotations)

        else:  # title
            paras_attr_list = htmltxtparser.lineinfos_paras_to_attr_list(paras_with_attrs, nl_text)
            start_offset, end_offset = \
                self.provision_annotator.extract_provision_offsets(paras_attr_list,
                                                                   paras_text)

            if start_offset is not None:
                prov_annotations = [{'end': end_offset,
                                     'label': self.provision,
                                     'start': start_offset,
                                     'prob': 0.91,
                                     'text': paras_text[start_offset:end_offset]}]

                fromto_mapper.adjust_fromto_offsets(prov_annotations)
            else:
                # didn't find title based on para_text, now try nl_text
                start_offset, end_offset = \
                    self.provision_annotator.extract_nl_provision_offsets(nl_text)

                if start_offset is not None:
                    prov_annotations = [{'end': end_offset,
                                         'label': self.provision,
                                         'start': start_offset,
                                         # span_list is normally added by fromto_mapper
                                         'span_list': [{'start': start_offset,
                                                        'end': end_offset}],
                                         'prob': 0.91,
                                         'text': nl_text[start_offset:end_offset]}]

                    # since nl_text has the original offsets, use that.
                    # DO NOT transform
                    # fromto_mapper.adjust_fromto_offsets(prov_annotations)
                else:  # still failed, last effort using regex match on paras_text
                    start_offset, end_offset = \
                        self.provision_annotator.extract_provision_offsets_not_line(paras_attr_list,
                                                                                    paras_text)

                    if start_offset is not None:
                        prov_annotations = [{'end': end_offset,
                                             'label': self.provision,
                                             'start': start_offset,
                                             'prob': 0.91,
                                             'text': paras_text[start_offset:end_offset]}]

                        fromto_mapper.adjust_fromto_offsets(prov_annotations)


        return prov_annotations
