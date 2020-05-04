#!/bin/python

class Token:
    def __init__(self, word, offset_from, offset_to):
        self.word = word
        self.offset_from = int(offset_from)
        self.offset_to = int(offset_to)
        self.type = "O"  # Initialize all tokens with type O as non-drugs non-brands and non-groups

    def __repr__(self):
        return (self.word, self.offset_from, self.offset_to, self.type)

class Entity:

    def __init__(self, **kwargs):
        self.word = kwargs["text"]
        self.offset_from, self.offset_to = self.parse_offset(kwargs["charOffset"])
        self.type = kwargs["type"]

    def parse_offset(self, offset):

        # offset can be given in two ways
        # e.g.:
        #       * 9-23
        #       * 9-11;12-20;21-23
        #
        # We differenciate both cases and always save the first one and the last one

        if ";" in offset:
            offset = offset.split(";")
            offset_from = offset[0].split('-')[0]
            offset_to = offset[-1].split('-')[1]
        else:
            offset_from, offset_to = offset.split('-')

        return int(offset_from), int(offset_to)

