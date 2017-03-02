#!/usr/bin/env python

class AnnotationType(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '{}'.format(self.name)
        
# Based on UIMA's approach, with 'start' and 'end',
# but without the efficiency.
class Annotation(object):

    def __init__(self, atype, start, end):
        self.atype = atype  # AnnotationType
        self.start = start
        self.end = end

    def __repr__(self):
        return '({} s={} e={})'.format(str(self.atype), self.start, self.end)

    def __setattr__(self, name, value):
        if name == 'pos':
            name = 'atype'
        super(Annotation, self).__setattr__(name, value)

    def __getattribute__(self, name):
        if name == 'pos':
            name = 'atype'
        return super(Annotation, self).__getattribute__(name)


class SentenceAnnotation(Annotation):

    sent_type = AnnotationType('sentence')
    
    def __init__(self, start, end, sf):
        super(self.__class__, self).__init__(SentenceAnnotation.sent_type, start, end)
        self.surface_form = sf

    def __repr__(self):
        return "({}, {}, '{}')".format(self.start, self.end,
                                       self.surface_form)

    def to_tuple(self):
        return (self.start, self.end, self.surface_form)
    
