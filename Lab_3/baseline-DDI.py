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
            # first key is not first word (is root)
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
            offset_end = offset_start + len(word) - 1  # -1 as length 1 is same start and end
            # store last offsets
            last_offset_end = offset_end
            # add start and end to the token
            head_node.nodes[key]['start'] = offset_start
            head_node.nodes[key]['end'] = offset_end

    return head_node


def get_entity_node_key(entity, analysis):
    for key in sorted(analysis.nodes, key=lambda key: int(key)):
        try:
            if analysis.nodes[key]['start'] == int(entity[0]):
                return key
        except KeyError:
            pass
    return 0


def isNodeInParent(parent, word):
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == word:
                return True
            if isNodeInParent(node, word):
                return True
        else:
            if node == word:
                return True
    return False


# receives DependencyGraph with all sentence, list of entities and the ids of the 2 entities to be checked
def check_interaction(analysis, entities, e1, e2):
    result = "0"
    interaction = "null"

    e1_node_key = get_entity_node_key(entities[e1], analysis)
    e2_node_key = get_entity_node_key(entities[e2], analysis)
    e1_word = analysis.nodes[e1_node_key]['word']
    e2_word = analysis.nodes[e2_node_key]['word']

    # if not found assume no interaction
    if e1_node_key == 0 or e2_node_key == 0:
        return result, interaction

    for key in sorted(analysis.nodes, key=lambda key: int(key)):
        try:
            current_word = analysis.nodes[key]['word']
            tree = analysis.tree()
            if current_word in int_clue_words:
                for subtree in tree.subtrees():
                    if subtree.label() in int_clue_words:
                        if isNodeInParent(subtree, e1_word) and isNodeInParent(subtree, e2_word):
                            result = "1"
                            interaction = "int"

            elif current_word in mechanism_clue_words:
                for subtree in tree.subtrees():
                    if subtree.label() in mechanism_clue_words:
                        if isNodeInParent(subtree, e1_word) and isNodeInParent(subtree, e2_word):
                            result = "1"
                            interaction = "mechanism"

            elif current_word in effect_clue_words:
                for subtree in tree.subtrees():
                    if subtree.label() in effect_clue_words:
                        if isNodeInParent(subtree, e1_word) and isNodeInParent(subtree, e2_word):
                            result = "1"
                            interaction = "effect"

            elif current_word in advise_clue_words:
                next_word = analysis.nodes[key + 1]['word']
                if next_word == "not" and analysis.nodes[key + 2]['word'] == "be" \
                        and analysis.nodes[key + 3]['tag'] == "VBN" \
                        or next_word == "be" and analysis.nodes[key + 2]['tag'] == "VBN":

                    result = "1"
                    interaction = "advise"

        except KeyError:
            pass

    return result, interaction


# receives data dir and filename for the results to evaluate
def evaluate(inputdir, outputfile):
    os.system("java -jar eval/evaluateDDI.jar " + inputdir + " " + outputfile)


#### VARIABLES ####
inputdir = sys.argv[1]
outputfilename = "./task9.2_TrainGianMarc_1.txt"
outputfile = open(outputfilename, "w")

#### RULE VARIABLES ####
effect_clue_words = {"administer", "potentiate", "prevent"}
mechanism_clue_words = {"reduce", "increase", "decrease"}
int_clue_words = {"interact", "interaction"}
advise_clue_words = {"may", "might", "should"}

# connect to CoreNLP server
my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

#### MAIN ####

# TODO mirar ordre!

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
            (is_ddi, ddi_type) = check_interaction(analysis, entities, id_e1, id_e2)
            line = "|".join([sid, id_e1, id_e2, is_ddi, ddi_type])
            outputfile.write(line + "\n")

evaluate(inputdir, outputfilename)
outputfile.close()
