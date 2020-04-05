#!/usr/bin/env python

import xml.etree.ElementTree as ET
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser

import os
import sys
import re


#### FUNCTIONS ####

def analyze(s):
    mytree = my_parser.raw_parse(s)
    last_offset_end = 0
    for head_node in mytree:
        for key in sorted(head_node.nodes, key=lambda key: int(key)):

            # first parsed is no word
            if key == 0:
                continue
            # find first occurrence of substring token in sentence
            word = head_node.nodes[key]['word']
            # Deal with left and right bracket
            if head_node.nodes[key]['word'] == '-LRB-':
                word = "("
            elif head_node.nodes[key]['word'] == '-RRB-':
                word = ")"

            offset_start = s.find(word, last_offset_end)
            offset_end = offset_start + len(word) - 1 # -1 as length 1 is same start and end
            # store last offsets
            last_offset_end = offset_end
            # add start and end to the token
            head_node.nodes[key]['start'] = offset_start
            head_node.nodes[key]['end'] = offset_end
    return mytree


#### VARIABLES ####
inputdir = sys.argv[1]
outputfilename = "./task9.2_TrainGianMarc_1.txt"
outputfile = open(outputfilename, "w")

# connect to CoreNLP server 
my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

#### MAIN ####

# process each file in directory
for filename in os.listdir(inputdir):
    # parse XML file, obtaining a DOM tree
    file_path = os.path.join(inputdir, filename)
    tree = ET.parse(file_path)
    sentences = tree.findall("sentence")

    for sentence in sentences:
        (sid, stext) = (sentence.attrib["id"], sentence.attrib["text"])

        # load sentence entities into a dictionary
        entities = {}
        ents = sentence.findall("entity")
        for e in ents:
            ent_id = e.attrib["id"]
            offs = e.attrib["charOffset"].split("-")
            entities[ent_id] = offs

        # Tokenize, tag, and parse sentence
        if not stext:
            continue
        analysis = analyze(stext)
        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = sentence.findall("pair")
        for pair in pairs:
            id_e1 = pair.attrib["e1"]
            id_e2 = pair.attrib["e2"]
            # TODO check_interaction function
            # (is_ddi, ddi_type) = check_interaction(analysis, entities, id_e1, id_e2)
            line = "|".join([sid, id_e1, id_e2])  # is_ddi, ddi_type])
            outputfile.write(line + "\n")

#TODO define evaluate
#evaluate(inputdir, outputfilename)
outputfile.close()
