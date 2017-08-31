
from kirke.utils import strutils

from kirke.docstruct.sent4nlp import FromToSpan

class FromToSpanText(FromToSpan):

    def __init__(self, text, from_start, from_end, to_start, to_end):
        super().__init__(from_start, from_end, to_start, to_end)
        self.text = text
        

class Sent4Nlp:

    def __init__(self, fromTo_span_list):
        self.from_start = fromTo_span_list[0].from_start
        self.from_end = fromTo_span_list[-1].from_end
        self.to_start = fromTo_span_list[0].to_start
        self.to_end = fromTo_span_list[-1].to_end
        
        self.fromTo_span_list = fromTo_span_list

    def get_text(self):
        st_list = [span.text for span in self.fromTo_span_list]
        return ''.join(st_list)

        
class Text4NlpReader:

    def __init__(self, filename):
        nlp_text = strutils.loads(filename)
        self.text = nlp_text

        raw_text = strutils.loads(filename.replace('.nlp.txt', '.txt'))

        sent_st_list = self.text.split('\n')
        sent_st_list = [sent_st for sent_st in sent_st_list if sent_st]  # remove the empty lines
        offsets_list = strutils.load_json_list(filename.replace('.nlp.txt', '.offsets.nlp.json'))

        print("len(sent_st_list:", len(sent_st_list))
        print("len(offsets_lsit:", len(offsets_list))

        raw_sent_st_list = []
        nlp_sent_st_list = []
        
        self.sent4nlp_list = []
        for offsets in offsets_list:
            span_list = []
            raw_span_st_list = []
            nlp_span_st_list = []
            for quad in offsets:
                from_start = quad['from_start']
                from_end = quad['from_end']
                to_start = quad['to_start']
                to_end = quad['to_end']                
                span_list.append(FromToSpanText(nlp_text[from_start:from_end], from_start, from_end, to_start, to_end))
                raw_span_st_list.append(raw_text[to_start:to_end])
                nlp_span_st_list.append(nlp_text[from_start:from_end])
            self.sent4nlp_list.append(Sent4Nlp(span_list))
            raw_sent_st_list.append(''.join(raw_span_st_list))
            nlp_sent_st_list.append(''.join(nlp_span_st_list))            

        for sent_st, sent4nlp, raw_sent_st, nlp_sent_st in zip(sent_st_list, self.sent4nlp_list, raw_sent_st_list, nlp_sent_st_list):
            #print("\n sent_st: {}".format(sent_st))
            #print("\nsent4nlp: {}".format(sent4nlp.get_text()))
            #print("\nraw_sent: {}".format(raw_sent_st))
            #print("\nnlp_sent: {}".format(nlp_sent_st))
            if sent_st != sent4nlp.get_text():
                print("**********************************************mismatch...")
            else:
                print("good match...")

        
def read_nlpdoc(filename):

    reader = Text4NlpReader(filename)


