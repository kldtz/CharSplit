import logging
import time

from kirke.eblearn import ebpostproc
from kirke.utils import evalutils

from kirke.docstruct import htmltxtparser
from kirke.ebrules import titles


class LineAnnotator:

    def __init__(self, provision, prov_annotator):
        self.provision_annotator = prov_annotator
        self.provision = provision
        self.eval_status = {}  # this is set after training

        
    def get_eval_status(self):
        return self.eval_status
    

    # ProvisionAnnotator does not train, it only predict
    # Training is available only for classifiers
    # def train(self):
    #    pass
    # pylint: disable=R0914
    def test_antdoc_list(self, paras_with_attrs_list, paras_text_list, ebantdoc_list):
        logging.debug('test_antdoc_list')

        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0

        for ebantdoc, paras_with_attrs, paras_text in zip(ebantdoc_list, paras_with_attrs_list, paras_text_list):
            ant_list = self.annotate_antdoc(paras_with_attrs, paras_text)
            print('ebantdoc.fileid = {}'.format(ebantdoc.file_id))
            # print("ant_list: {}".format(ant_list))
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.provision]

            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            xtp, xfn, xfp, xtn = \
                evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                 ant_list,
                                                                 paras_text,
                                                                 diagnose_mode=True)
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
    

    def test_antdoc(self, paras_with_attrs, doc_text, ebantdoc):
        logging.debug('test_document')

        ant_list = self.annotate_antdoc(paras_with_attrs, doc_text)
        # print("ant_list: {}".format(ant_list))
        prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                               if hant.label == self.provision]
        # print("human_list: {}".format(prov_human_ant_list))

        # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
        # pred_prob_start_end_list, txt)
        # pylint: disable=C0103
        tp, fn, fp, tn = evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                 ant_list,
                                                                 ebantdoc.get_text())

        title = "annotate_status, line-based"
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'prec': prec, 'recall': recall, 'f1': f1}}

        return tmp_eval_status


    def annotate_antdoc(self, paras_with_attrs, paras_text):
        prov_annotations = []
        if self.provision == 'party':
            paras_attr_list = htmltxtparser.lineinfos_paras_to_attr_list(paras_with_attrs)
            party_offset_pair_list = self.provision_annotator.extract_provision_offsets(paras_attr_list, paras_text)
            
            if party_offset_pair_list:
                for i, party_offset_pair in enumerate(party_offset_pair_list, 1):
                    (party_start, party_end), term_ox = party_offset_pair
                    prov_annotations.append({'end': party_end,
                                             'label': self.provision,
                                             'id': i,
                                             'start': party_start,
                                             'prob': 1.0,
                                             'text': paras_text[party_start:party_end]})
                    if term_ox:
                        term_start, term_end = term_ox
                        prov_annotations.append({'end': term_end,
                                                 'label': self.provision,
                                                 'id': i,
                                                 'start': term_start,
                                                 'prob': 1.0,
                                                 'text': paras_text[term_start:term_end]})
        elif self.provision == 'date':
            paras_attr_list = htmltxtparser.lineinfos_paras_to_attr_list(paras_with_attrs)
            # prov_type can be 'date', 'effective-date', 'signature-date'
            date_list = self.provision_annotator.extract_provision_offsets(paras_attr_list, paras_text)


            # print('title_start, end = ({}, {})'.format(start_offset, end_offset))

            #print("ebannotator({}).threshold = {}".format(self.provision,
            #self.threshold))

            if date_list:
                for i, date_ox in enumerate(date_list, 1):
                    start_offset, end_offset, prov_type = date_ox
                    prov_annotations.append({'end': end_offset,
                                             'label': prov_type,
                                             'start': start_offset,
                                             'prob': 1.0,
                                             'text': paras_text[start_offset:end_offset]})
        else:
            paras_attr_list = htmltxtparser.lineinfos_paras_to_attr_list(paras_with_attrs)
            start_offset, end_offset = self.provision_annotator.extract_provision_offsets(paras_attr_list, paras_text)

            # print('title_start, end = ({}, {})'.format(start_offset, end_offset))

            #print("ebannotator({}).threshold = {}".format(self.provision,
            #self.threshold))

            if start_offset is not None:
                prov_annotations = [{'end': end_offset,
                                     'label': self.provision,
                                     'start': start_offset,
                                     'prob': 1.0,
                                     'text': paras_text[start_offset:end_offset]}]

        return prov_annotations