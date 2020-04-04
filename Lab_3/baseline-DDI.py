#!/usr/bin/env python

import xml.etree.ElementTree as ET
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser

import os 
import sys
import re

#### VARIABLES ####
inputdir = sys.argv[1]
outputfilename = "./task9.2_TrainGianMarc_1.txt"
outputfile = open(outputfilename, "w")

#### MAIN ####
# connect to CoreNLP server 
my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

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
        for e in ents :
            ent_id = e.attrib["id"]
            offs = e.attrib["charOffset"].split("-")
            entities[ent_id] = offs
            print(ent_id) 
        
        # Tokenize, tag, and parse sentence
        # TODO analyze function
        #analysis = analyze(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = sentence.findall("pair")
        for pair in pairs:
            id_e1 = pair.attrib["e1"]
            id_e2 = pair.attrib["e2"]
            # TODO check_interaction function
            #(is_ddi, ddi_type) = check_interaction(analysis, entities, id_e1, id_e2)            
            line = "|".join([sid, id_e1, id_e2, is_ddi, ddi_type])
            outputfile.write(line + "\n")

evaluate(inputdir, outputfilename)
outputfile.close()
