from kirke.docstruct import tokenizer

# DTree = DocumentTree, or DocTree
class DTreeAnnotation:

    def __init__(self, start, end):
        self.start = start
        self.end = end
        
        
class DTreeDoc(DTreeAnnotation):

    def __init__(self, fname, start, end, text):
        super().__init__(start, end)
        self.file_name = fname
        self.text = text
        self.toc = []
        self.section_list = []
        self.misc_content = []
        self.pagenum_list = []
        self.pheader_list = []
        self.pfooter_list = []
        self.table_list = []
        self.page_list = []

        
class DTreeSection(DTreeAnnotation):

    def __init__(self, start, end, text):
        super().__init__(start, end)        
        self.header = None
        self.paragraph = []
        self.text = text

        
class DTreeParagraph(DTreeAnnotation):
    
    def __init__(self, start, end, text):
        super().__init__(start, end)
        self.sent_header = None
        self.text = text
        self.words = tokenizer.word_tokenize(text)

class DTreeSegment(DTreeAnnotation):
    
    def __init__(self, category, start, end, pagenum, text):
        super().__init__(start, end)
        self.category = category
        self.pagenum = pagenum
        self.text = text

    def __repr__(self):
        return '(Segment-%s page= %d, (%d, %d): [%s...])' % (self.category,
                                                             self.pagenum,
                                                             self.start,
                                                             self.end,
                                                             self.text[:20])
    
        
class DTreePageNumber(DTreeAnnotation):

    def __init__(self, start, end, page_number, text):
        super().__init__(start, end) 
        self.page_number = page_number
        self.text = text

    def __repr__(self):
        return "PageNumber({}, {}, '{}', '{}')".format(self.start, self.end, self.page_number,
                                                       self.text)
