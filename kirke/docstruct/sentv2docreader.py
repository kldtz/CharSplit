
from kirke.utils import strutils

        
class SentV2DocReader:

    def __init__(self, filename):
        doc_text = strutils.loads(filename)
        self.text = doc_text

        sentv2_text = strutils.loads(filename.replace(".txt", ".sentv2.txt"))

        sentv2_st_list = sentv2_text.split('\n')

        sentv2_st_list = [sentv2_st for sentv2_st in sentv2_st_list if sentv2_st]
                                    
        offsets_list = strutils.load_json_list(filename.replace('.txt', '.offsets.sentv2.json'))

        print("len(sentv2_st_list:", len(sentv2_st_list))
        print("len(offsets_lsit:", len(offsets_list))

        for offsets, sentv2_st in zip(offsets_list, sentv2_st_list):
            start = offsets['start']
            end = offsets['end']                

            # print("\n sentv2_st: [{}]".format(sentv2_st))
            # print("\n      text: [{}]".format(doc_text[start:end]))
            if sentv2_st != doc_text[start:end]:
                print("**********************************************mismatch...")
            else:
                print("good match...")

        
def read_sentv2doc(filename):
    reader = SentV2DocReader(filename)



