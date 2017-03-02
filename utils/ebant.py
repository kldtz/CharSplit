#!/usr/bin/env python

import json
import argparse

class EbAnnotation:

    def __init__(self, ajson):
        self.confidence = ajson['confidence']
        self.correctness = ajson.get('correctness')
        self.start = ajson.get('start')
        self.end = ajson.get('end')
        self.type = ajson.get('type')
        self.text = ajson.get('text')
        self.id = ajson.get('id')    # string
        self.custom_text = ajson.get('customText')  # boolean

    def to_dict(self):
        return {"confidence" : self.confidence,
                "correctness" : self.correctness,
                "customText" : self.custom_text,
                "start" : self.start,
                "end" : self.end,
                "id" : self.id,
                "text" : self.text,
                "type" : self.type}

    def __str__(self):
        return str(self.to_dict())

    def to_tuple(self):
        return self.type, self.start, self.end


def load_provision_annotations(filename, provision):
    result = []
    with open(filename, "rt") as handle:
        parsed = json.load(handle)

        for ajson in parsed:
            eb_ant = EbAnnotation(ajson)
            # print("eb_ant= {}".format(eb_ant))
            # print("eb_ant= {}".format(eb_ant.to_tuple()))
            if eb_ant.type == provision:
                result.append(eb_ant.to_tuple())
    return result
        
    
if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description='normalize address v1')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument("filename", nargs='?')

    args = parser.parse_args()
    if args.verbosity:
        print("verbosity turned on")
    if args.debug:
        isDebug= True
    if args.filename:
        with open(args.filename, "rt") as handle:
            parsed = json.load(handle)
    else:
        parsed = json.load(sys.stdin)

    # print(json.dumps(parsed, indent=4, sort_keys=True))
    for ajson in parsed:
        eb_ant = EbAnnotation(ajson)
        # print("eb_ant= {}".format(eb_ant))
        print("eb_ant= {}".format(eb_ant.to_tuple()))

    print("provision = party")
    party_ants = load_provision_annotations(args.filename, 'party')
    print(party_ants)
    

        
