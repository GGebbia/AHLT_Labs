from __future__ import print_function
import xml.etree.ElementTree as ET

import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser

import os
import sys
import re
import argparse

nltk.download('punkt')

class Entity:

    def __init__(self, **kwargs):
        self.word = kwargs["text"]
        self.offset_from, self.offset_to = self.parse_offset(kwargs["charOffset"])
        self.id = kwargs["id"]

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

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def analyze(s):
    mytree = my_parser.raw_parse(s)

    last_offset_end = 0
    for head_node in mytree:
        # first key is not first word (is root)
        for key in list(head_node.nodes.keys())[1:]:
            # find first occurrence of substring token in sentence
            word = head_node.nodes[key]['word']
            offset_start = s.find(word, last_offset_end)
            offset_end = offset_start + len(word) - 1  # -1 as length 1 is same start and end
            # store last offsets
            last_offset_end = offset_end
            # add start and end to the token
            head_node.nodes[key]['start'] = offset_start
            head_node.nodes[key]['end'] = offset_end

    return head_node

def is_node_in_parent(parent, word):
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == word:
                return True
            if is_node_in_parent(node, word):
                return True
        else:
            if node == word:
                return True
    return False

def get_subtree_from_word(tree, ent):

    # Get the first word if there are more than one
    for word in ent.split():
        entity_subtree =  list(tree.subtrees(filter=lambda t: t.label() == word))
        if entity_subtree != []:
            return entity_subtree[0]

    return []

def get_dependency(analysis, e1, e2):
    #Split entities in a list of words:
    for governor, dep, dependent in analysis.triples():
        governor_word = governor[0]
        dependent_word = dependent[0]
        if (governor_word in e1 and dependent_word in e2) or (governor_word in e2 and dependent_word in e1):
            return dep
    return None
# Given a token list, collect the most relevant features to store in a list.
def extract_features(analysis, sentence, entities, e1, e2):
    features = []


    tree = analysis.tree()

    dep = get_dependency(analysis, e1, e2)
    features.append("dep={}".format(dep))
    for lemma in sentence.partition(e1)[0].split():
        features.append("lb1={}".format(lemma))
    for lemma in find_between(sentence, e1, e2).split():
        features.append("lib={}".format(lemma))
    for lemma in sentence.partition(e2)[2].split():
        features.append("la2={}".format(lemma))



    e1_tree = get_subtree_from_word(tree, e1)
    e2_tree = get_subtree_from_word(tree, e2)
    if is_node_in_parent(e1_tree, e2):
        features.append("2under1")
    elif is_node_in_parent(e2_tree, e1):
        features.append("1under2")


    return " ".join(features)



# Store as a list of dictionaries the word, the offset interval and the label (drug, group, brand, drug_n,...) of each entity in the sentence.
def get_entities(child):
    entities = []
    for ent in child.findall('entity'):
        entity = Entity(**ent.attrib)
        entities.append(entity)
    return entities

def get_entity_by_id(entities, ent_id):
    for entity in entities:
        if entity.id == ent_id:
            return entity.word
    return None

def extract_entities(sentence):
    # load sentence entities into a dictionary
    entities = {}
    ents = sentence.findall("entity")
    for e in ents:
        ent_id = e.attrib["id"]
        offs = e.attrib["charOffset"].split("-")
        entities[ent_id] = offs
    return entities


### MAIN ###

parser = argparse.ArgumentParser(description=
                                 """
        Compute the feature extractor of a given dataset containing XML Files.\n
        Usage: \n\npython3 feature_extractor.py --dir data/Train --type <feats|megam> --out ./extracted_features/Train\n\n
        If feats is selected returns the complete feature extractor with more detailed information about each sample.\n
        If megam is selected returns only the features of each sample.\n
        """
                                 )
parser.add_argument('--type', type=str, choices=["feats", "megam"], help='Two option of extraction: feats or megam')
parser.add_argument('--dir', type=str, help='Directory where are located the XML files')
parser.add_argument('--out', type=str, help='Output directory to save the extracted features.')
args = parser.parse_args()

inputdir = args.dir
outputfilename = os.path.join(args.out, args.type) + ".dat"

if not os.path.exists(args.out):
    os.makedirs(args.out)

outputfile = open(outputfilename, "w")

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

        entities = get_entities(sentence)
        # Tokenize, tag, and parse sentence
        if not stext:
            continue
        analysis = analyze(stext)
        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = sentence.findall("pair")
        for pair in pairs:
            id_e1 = pair.attrib["e1"]
            id_e2 = pair.attrib["e2"]
            if pair.attrib["ddi"] == "true":

                try:
                    ddi_type = pair.attrib["type"]
                except:
                    ddi_type = "null"

            else:
                ddi_type = "null"

            e1 = get_entity_by_id(entities, id_e1)
            e2 = get_entity_by_id(entities, id_e2)

            features = extract_features(analysis, stext, entities, e1, e2)

            if args.type == "feats":
                line = " ".join([sid, id_e1, id_e2, ddi_type, features])
            elif args.type == "megam":
                line = " ".join([ddi_type, features])
            else:
                print("Incorrect feature extraction type\n")
                sys.exit(1)
            outputfile.write(line + "\n")


outputfile.close()
