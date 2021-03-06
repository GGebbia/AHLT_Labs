from __future__ import print_function
import xml.etree.ElementTree as ET

import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser

import os
import sys
import re
import argparse

nltk.download('punkt')

def analyze(s):
    mytree = my_parser.raw_parse(s)

    last_offset_end = 0
    for head_node in mytree:
        # first key is not first word (is root)
        for key in sorted(head_node.nodes, key=lambda key: int(key)):
            # first key is not first word (is root)
            if key == 0:
                continue
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

# Custom class entity
class Entity:
    def __init__(self, **kwargs):
        self.word = kwargs["text"]
        self.offset_from, self.offset_to = self.parse_offset(kwargs["charOffset"])
        self.id = kwargs["id"]

    def parse_offset(self, offset):
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

def get_entity_node_key(entity, analysis):
    for key in sorted(analysis.nodes, key=lambda key: int(key)):
        try:
            if analysis.nodes[key]['start'] == entity.offset_from:
                return key
            if analysis.nodes[key]['end'] == entity.offset_to:
                return key
            if analysis.nodes[key]['word'] == entity.word:
                return key
        except KeyError:
            pass
    return 0
# Return path of each entity to common parent
def separately_paths_to_common_parent(list1, list2):
    h_1 = [el[0] for el in list1]
    h_2 = [el[0] for el in list2]
    h_intersection = [value for value in h_1 if value in h_2]

    if h_intersection != []:
        common_parent_node_key = h_intersection[0]
    else:
        common_parent_node_key = 0
    path_1 = sorted([el for el in list1 if el[0] >= common_parent_node_key])
    path_2 = sorted([el for el in list2 if el[0] >= common_parent_node_key])

    return path_1, path_2

def _get_heads_and_relations(entity_key, analysis):
    node = analysis.nodes[entity_key]

    head = node["head"]
    rel = node["rel"]
    if head != 0:
        heads_rels.append((head,rel))
        _get_heads_and_relations(head,analysis)
    return heads_rels
# Return heads of entity
def get_heads_and_relations(entity_key, analysis):
    global heads_rels
    heads_rels = []
    heads_rels = _get_heads_and_relations(entity_key, analysis)
    return heads_rels

def count_no_not_in_sentence(sentence):
    counter = 0
    for word in nltk.word_tokenize(sentence):
        if word == 'not' or word == 'no' or word[-3:] == "n't":
            counter += 1
        return counter

def mechanism_words_present(sentence):
    for word in nltk.word_tokenize(sentence):
        if word in mechanism_clue_words:
            return True
    return False

def effect_words_present(sentence):
    for word in nltk.word_tokenize(sentence):
        if word in effect_clue_words:
            return True
    return False

def int_words_present(sentence):
    for word in nltk.word_tokenize(sentence):
        if word in int_clue_words:
            return True
    return False

def special_words_count(sentence):
    counter = 0
    for word in nltk.word_tokenize(sentence.lower()):
        if word in special_words:
            counter += 1
    return counter

def special_words_pos_count(sentence, e1_offset_1, e1_offset_2, e2_offset_1, e2_offset_2):
    first_phrase = sentence[0:e1_offset_1].split()
    mid_phrase = sentence[e1_offset_2 + 1:e2_offset_1].split()
    end_phrase = sentence[e2_offset_2 + 1:].split()
    before_first_count = len(set(first_phrase) & special_words)
    middle_count = len(set(mid_phrase) & special_words)
    after_second_count = len(set(end_phrase) & special_words)
    return before_first_count, middle_count, after_second_count

def negative_words_count(sentence, e1_offset_1, e1_offset_2, e2_offset_1, e2_offset_2):
    first_phrase = sentence[0:e1_offset_1].split()
    mid_phrase = sentence[e1_offset_2 + 1:e2_offset_1].split()
    end_phrase = sentence[e2_offset_2 + 1:].split()
    before_first_count = len(set(first_phrase) & negative_words)
    middle_count = len(set(mid_phrase) & negative_words)
    after_second_count = len(set(end_phrase) & negative_words)
    return before_first_count, middle_count, after_second_count

def count_key_words_present(sentence):
    key_words = ['coadministration', 'concomitant', 'concomitantly']
    count = 0
    for word in key_words:
        if word in nltk.word_tokenize(sentence.lower()):
            count += 1
    return count

def words_sep_by_and(sentence, e1_word, e2_word):
    sentence = sentence.replace(e1_word, 'e1')
    sentence = sentence.replace(e2_word, 'e2')
    words = nltk.word_tokenize(sentence)
    try:
        if abs(words.index('e1') - words.index('e2')) < 3:
            if words[words.index('e1') + 1] == 'and':
                return True
    except ValueError:
        pass
    return False

def words_sep_by_comma(sentence, e1_word, e2_word):
    sentence = sentence.replace(e1_word, 'e1')
    sentence = sentence.replace(e2_word, 'e2')
    words = nltk.word_tokenize(sentence)
    try:
        if abs(words.index('e1') - words.index('e2')) < 3:
            if words[words.index('e1') + 1] == ',':
                return True
    except ValueError:
        pass
    return False

# Check if key phrase present
def key_phrase(sentence, e1_word, e2_word):
    sentence = sentence.replace(e1_word, 'drug1')
    sentence = sentence.replace(e2_word, 'drug2')
    key_phrases = ['concurrent administration of drug1 and drug2', 'drug1 concurrently with drug2',
            'co-administration of drug1 and drug2', 'coadministration of drug1 and drug2',
               'concurrent use of drug1 and drug2',]
    for phrase in key_phrases:
        if phrase in sentence.lower():
            return True
    return False

# Counts how many words are between the two entities
def count_words_between(sentence, e1_offset_end, e2_offset_begin):
    words_between = sentence[e1_offset_end+1:e2_offset_begin].split()
    return len(words_between)

# True if a special word is in the direct path between e1 and e2
def special_word_in_path(path1, path2):
    for head, _ in path1:
        if analysis.nodes[head]["word"] in special_words:
            return True

    for head, _ in path2:
        if analysis.nodes[head]["word"] in special_words:
            return True
    return False

# Given a token list, collect the most relevant features to store in a list.
def extract_features(analysis, sentence, entities, id_e1, id_e2):
    features = []
    # Get entities with custom class
    e1 = get_entity_by_id(entities, id_e1)
    e2 = get_entity_by_id(entities, id_e2)
    key_e1 = get_entity_node_key(e1,analysis)
    key_e2 = get_entity_node_key(e2,analysis)

    if key_e1 == 0 or key_e2 == 0:
        return ""

    # get the corresponding word (token) of each entity from the dependency graph
    e1_word = analysis.nodes[key_e1]['word']
    e2_word = analysis.nodes[key_e2]['word']

    for key in sorted(analysis.nodes, key=lambda key: int(key)):
        try:
            # get current word of current node
            current_word = analysis.nodes[key]['word']
            # Check if interaction is advise
            if current_word in advise_clue_words:
                next_word = analysis.nodes[key + 1]['word']
                if next_word == "not" and analysis.nodes[key + 2]['word'] == "be" \
                        and analysis.nodes[key + 3]['tag'] == "VBN" \
                        or next_word == "be" and analysis.nodes[key + 2]['tag'] == "VBN":
                    features.append("advise_guess")
        except KeyError:
            pass

    # Lemmas in between
    for lemma in find_between(sentence, str(e1.offset_to), str(e2.offset_from)).split():
        features.append("lib{}".format(lemma.replace("=","eq")))

    # Paths to common parent, tags features
    heads_rels_e1 = get_heads_and_relations(key_e1, analysis)
    heads_rels_e2 = get_heads_and_relations(key_e2, analysis)
    path_to_cp_e1, path_to_cp_e2 = separately_paths_to_common_parent(heads_rels_e1, heads_rels_e2)
    for item in path_to_cp_e2:
        features.append("cpe2{}".format(item[1]))
    for item in path_to_cp_e1:
        features.append("cpe1{}".format(item[1]))

    # Paths to common parent, lemma features
    for (head, _) in path_to_cp_e2:
        word = analysis.nodes[head]["word"]
        features.append("cpe2w{}".format(word))
    for (head, _) in path_to_cp_e1:
        word = analysis.nodes[head]["word"]
        features.append("cpe1w{}".format(word))

    if key_e1 in [el[0] for el in path_to_cp_e2]:
        features.append("1under2")
    elif key_e2 in [el[0] for el in path_to_cp_e1]:
        features.append("2under1")
    # Number of no, nots etc.
    count_nots = count_no_not_in_sentence(sentence)
    features.append("count_not={}".format(count_nots))
    # Key words for each type present
    if mechanism_words_present(sentence):
        features.append("mechanism_guess")
    if effect_words_present(sentence):
        features.append("effect_guess")
    if int_words_present(sentence):
        features.append("int_guess")

    # Number of special words
    special_w = special_words_count(sentence)
    features.append("special_w={}".format(special_w))
    # Special words in each position
    special_w_pos_b, special_w_pos_mid, special_w_pos_end = special_words_pos_count(sentence, e1.offset_from, e1.offset_to, e2.offset_from, e2.offset_to)
    features.append("special_w_pos_begin={}".format(special_w_pos_b))
    features.append("special_w_pos_mid={}".format(special_w_pos_mid))
    features.append("special_w_pos_end={}".format(special_w_pos_end))
    # Num of negative words in each pos
    neg_words_count_b, neg_words_count_mid, neg_words_count_end = negative_words_count(sentence, e1.offset_from, e1.offset_to, e2.offset_from, e2.offset_to)
    features.append("neg_word_begin={}".format(neg_words_count_b))
    features.append("neg_word_mid={}".format(neg_words_count_mid))
    features.append("neg_word_end={}".format(neg_words_count_end))
    # Num of key words
    key_words_count = count_key_words_present(sentence)
    features.append("num_keywords={}".format(key_words_count))
    # Lemmas between two entities
    words_between = count_words_between(sentence, e1.offset_to, e2.offset_from)
    features.append("words_between={}".format(words_between))
    if words_sep_by_and(sentence, e1_word, e2_word):
        features.append("separated_by_and")
    if words_sep_by_comma(sentence, e1_word, e2_word):
        features.append("separated_by_comma")
    if key_phrase(sentence, e1_word, e2_word):
        features.append("key_phrase")
    # Count how many drugs are in between the two entities
    count_drugs = 0
    for entity in entities:
        if (e1.offset_to < entity.offset_from) and (entity.offset_to < e2.offset_from):
            count_drugs += 1
    features.append("count_drugs={}".format(count_drugs))

    if any(rel == "nsubj" for head, rel in path_to_cp_e1) and any(rel == "dobj" for head, rel in path_to_cp_e2):
        features.append("sub_obj")

    if analysis.nodes[key_e1]["head"] == analysis.nodes[key_e2]["head"]:
        features.append("both_head")

    special_word = special_word_in_path(path_to_cp_e1, path_to_cp_e2)
    if special_word:
        features.append("special_word")
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
            return entity
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

effect_clue_words = {"administer", "potentiate", "prevent", "antagonize", "antagonized"}
mechanism_clue_words = {"reduce", "increase", "decrease"}
int_clue_words = {"interact", "interaction"}
advise_clue_words = {"may", "might", "should"}
special_words = []
with open("special_words.txt") as f:
    for line in f:
        special_words.append(line)
special_words = set(special_words)
negative_words = ['No', 'not', 'neither', 'without', 'lack', 'fail',
                  'unable', 'abrogate', 'absence', 'prevent',
                  'unlikely', 'unchanged', 'rarely']
negative_words = set(negative_words)

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

            features = extract_features(analysis, stext, entities, id_e1, id_e2)

            if args.type == "feats":
                line = " ".join([sid, id_e1, id_e2, ddi_type, features])
            elif args.type == "megam":
                line = " ".join([ddi_type, features])
            else:
                print("Incorrect feature extraction type\n")
                sys.exit(1)
            outputfile.write(line + "\n")


outputfile.close()
