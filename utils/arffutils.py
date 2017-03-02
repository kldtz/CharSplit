#!/usr/bin/env python3

import arff   # pip install liac-arff
import csv
import os
import sys

class ArffObj:

    def __init__(self, relation=None):
        if relation:
            self.relation = relation
        else:
            self.relation = ''
        self.description = ''
        self.attributes = []
        self.data = []

    # Due to the limitation of liac-arff.arff.load(), This load() doesn't
    # handle commas or newlines in a String field.
    # Comma issue can be fixed by replacing
    # line 334 in arff.py in liac-arff library:
    #     values = next(csv.reader([s.strip(' ')]))
    # with
    #     values = next(csv.reader([s.strip(' ')], quotechar="'",
    #                   escapechar='\\', doublequote=False))
    # Linebreak issue, unfortunately, requires much more work because
    # by the time parsing of row is performed, liac-arff already splits
    # rows by newlines -- csv.read() only receive broken lines.
    # Will take too much effort to fix.
    def load(self, file_name):
        data = arff.load(open(file_name, 'rb'))
        self.description = data.get('description', '')
        self.relation = data.get('relation', '')
        self.attributes = data.get('attributes', [])
        self.data = data.get('data', [])

    def save_arff(self, file_name):
        adict = self.to_dict()
        st = arff.dumps(adict)
        with open(file_name, 'wt') as outs:
            outs.write(st)
            outs.write(os.linesep)

    # write to a CSV that python's standard reader can handle
    def save_csv(self, file_name):
        headers = self.get_attribute_names()
        instance_list = self.data 
        with open(file_name, 'wt') as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow(headers)
            for row in instance_list:
                wr.writerow(row)

    def set_description(self, st):
        self.description = st

    def set_relation(self, st):
        self.relation = st
        
    def add_attr(self, name, atype=None):
        if not atype:
            self.attributes.append((name, 'NUMERIC'))
        elif atype == 'STRING':
            self.attributes.append((name, 'STRING'))
        elif isinstance(atype, list):
            self.attributes.append((name, atype))

    def add_data(self, alist):
        self.data.append(alist)

    def clear_data(self):
        self.data = []

    def to_dict(self):
        obj = {
            'description': self.description,
            'relation': self.relation,
            'attributes': self.attributes,
            'data': self.data
            }
        return obj

    def __repr__(self):
        adict = self.to_dict()
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(adict)
        return arff.dumps(adict)

    def get_attributes(self):
        return self.attributes

    def get_attribute_names(self):
        return [attr_type[0] for attr_type in self.attributes]

    def get_data(self):
        return self.data


# this reads an arff file using the python CSV reader.  Arffs are basically CSV files with type information anyway
def read_arff_header_data(file_name):
    headers = []
    instance_list = []

    with open(file_name, 'rt') as ins:
        for line in ins:
            parts = line.strip().split()

            if not parts:
                continue
            if parts[0].lower() == '@attribute':
                headers.append(parts[1])
            elif parts[0].lower() == '@data':
                break        

        # Special CSV dialect for arff files
        # This handles comma and linebreak in a column
        reader = csv.reader(ins, quotechar="'", escapechar='\\', doublequote=False)
        header_size = len(headers)
        for i, instance in enumerate(list(reader)):
            # print("instance {}: {}".format(len(instance), instance))
            if len(instance) == 1 and instance[0].startswith('%'):
                continue
            if len(instance) != header_size:
                print('Error: number of column different from header, expected {}, got {}'.format(header_size, len(instance)), file=sys.stderr)
                print("  file_name: '{}', instance_id: {}".format(file_name, i), file=sys.stderr)
                print("  column: '{}'".format('|'.join(instance)), file=sys.stderr)
                continue
            instance_list.append(instance)
    return (headers, instance_list)


def arff_to_csv(arff_file_name, csv_file_name):
    headers, instance_list = read_arff_header_data(arff_file_name)
    with open(csv_file_name, 'wt') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(headers)
        for row in instance_list:
            wr.writerow(row)
