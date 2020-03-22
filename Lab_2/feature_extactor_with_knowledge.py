from __future__ import print_function
import xml.etree.ElementTree as ET

import nltk
from nltk.tokenize import WhitespaceTokenizer

import os
import sys
import re
import argparse

nltk.download('punkt')


class Token:
    def __init__(self, word, offset_from, offset_to):
        self.word = word
        self.offset_from = int(offset_from)
        self.offset_to = int(offset_to)
        self.type = "O"  # Initialize all tokens with type O as non-drugs non-brands and non-groups

    def word_iscapitalized(self):
        return self.word[0].isupper()

    def word_isupper(self):
        return self.word.isupper()


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


# Read a string sentence and tokenize it by words with the corresponding offset. words ending with comma or dot are splitted onto two tokenized words. Return a tokenized list of sentence.

def tokenize(sentence):
    span_generator = WhitespaceTokenizer().span_tokenize(sentence)
    tokens = [(sentence[span[0]: span[1]], span[0], span[1] - 1) for span in span_generator]

    new_tokens = []
    for i, token in enumerate(tokens):
        word, offset_from, offset_to = token

        if (len(word) > 1) and (word.endswith(',') or word.endswith('.') or word.endswith(':') or word.endswith(';')):
            punct = word[-1]
            punct_offset_from = offset_to
            punct_offset_to = offset_to

            word = word[:-1]
            offset_to -= 1

            new_tokens.append(Token(word, offset_from, offset_to))
            new_tokens.append(Token(punct, punct_offset_from, punct_offset_to))

        elif (len(word) > 1) and word[0] == '(' and (word[0:2] != '(+' or word[0:2] != '(-'):
            punct = word[0]
            punct_offset_from = offset_from
            punct_offset_to = offset_from

            word = word[1:]
            offset_from += 1

            new_tokens.append(Token(punct, punct_offset_from, punct_offset_to))
            new_tokens.append(Token(word, offset_from, offset_to))
        else:
            new_tokens.append(Token(word, offset_from, offset_to))

    return new_tokens


# Given a token list, collect the most relevant features to store in a list.
def extract_features(tokens):
    features = []
    next_is_brand_txt = 0
    next_is_drug_n_txt = 0
    next_is_group_txt = 0
    all_words_capitalized = all(token.word_iscapitalized() or len(token.word) < 2 for token in tokens)

    for i, token in enumerate(tokens, 0):
        word_features = []

        # COMMON FEATURES
        form = token.word
        word_features.append("form={}".format(form))
        suf4 = token.word[-4:]
        word_features.append("suf4={}".format(suf4))
        max_pref_len = 4
        if len(token.word) < max_pref_len:
            max_pref_len = len(token.word)
        pref4 = token.word[0:max_pref_len]
        word_features.append("pref4={}".format(pref4))
        if i != 0:
            prev = tokens[i - 1].word
            word_features.append("prevsuf4={}".format(prev[-4:]))
            max_pref_len = 4
            if len(prev) < max_pref_len:
                max_pref_len = len(prev)
            prevpref4 = prev[0:max_pref_len]
            word_features.append("prevpref4={}".format(prevpref4))
            if prev[0].isupper():
                word_features.append("prevupper")
        else:
            prev = "_BoS_"
        word_features.append("prev={}".format(prev))

        # todo try next and previous prefix and sufix
        try:
            _next = tokens[i + 1].word
            word_features.append("next={}".format(_next))
            max_pref_len = 4
            if len(_next) < max_pref_len:
                max_pref_len = len(_next)
            nextpref4 = _next[0:max_pref_len]
            word_features.append("nextpref4={}".format(nextpref4))
            word_features.append("nextsuf4={}".format(_next[-4:]))
        except:
            pass

        # SPECIFIC FEATURES
        if token.word_iscapitalized():
            word_features.append("capitalized")

        if token.word_isupper():
            word_features.append("isupper")

        if all_words_capitalized:
            word_features.append("all_capitalized")

        for drug_n in drug_n_list:
            if drug_n.lower() in token.word.lower():
                word_features.append("drug_n_list")
                break

        if next_is_group_txt > 0:
            word_features.append("group_txt")
            next_is_brand_txt -= 1
        for line in group_names_txt:
            entire_line = []
            number_of_words_in_line = 0
            same_line = False
            line_words = line.split()
            # check each word in line for exact match
            for word_index, line_word in enumerate(line_words):
                try:
                    word_to_check = tokens[i + word_index].word
                except:
                    break
                if word_to_check.lower() == line_word.lower():
                    entire_line.append(word_to_check)
                    number_of_words_in_line += 1
                    same_line = True
                else:
                    same_line = False
                    break
            if same_line:
                word_features.append("group_txt")
                next_is_group_txt = word_index
                break

        if next_is_brand_txt > 0:
            word_features.append("brand_txt")
            next_is_brand_txt -= 1
        for line in brand_names_txt:
            entire_line = []
            number_of_words_in_line = 0
            same_line = False
            line_words = line.split()
            # check each word in line for exact match
            for word_index, line_word in enumerate(line_words):
                try:
                    word_to_check = tokens[i + word_index].word
                except:
                    break
                if word_to_check.lower() == line_word.lower():
                    entire_line.append(word_to_check)
                    number_of_words_in_line += 1
                    same_line = True
                else:
                    same_line = False
                    break
            if same_line:
                word_features.append("brand_txt")
                next_is_brand_txt = word_index
                break

        if next_is_drug_n_txt > 0:
            word_features.append("drug_n_txt")
            next_is_drug_n_txt -= 1
        for line in drug_n_list_txt:
            entire_line = []
            number_of_words_in_line = 0
            same_line = False
            line_words = line.split()
            # check each word in line for exact match
            for word_index, line_word in enumerate(line_words):
                try:
                    word_to_check = tokens[i + word_index].word
                except:
                    break
                if word_to_check.lower() == line_word.lower():
                    entire_line.append(word_to_check)
                    number_of_words_in_line += 1
                    same_line = True
                else:
                    same_line = False
                    break
            if same_line:
                word_features.append("drug_n_txt")
                next_is_drug_n_txt = word_index
                break

        features.append(word_features)

    return features


def detect_label(token, entities):
    for entity in entities:
        # If the two offsets are equal, then it corresponds to the same word and type.
        if token.offset_from == entity.offset_from and token.offset_to == entity.offset_to:
            token.type = "B-" + entity.type
            return
        # If the token offset interval is inside the entity offset interval, then it is a first or the continuation of a type sequence.
        elif entity.offset_from <= token.offset_from and token.offset_to <= entity.offset_to:
            if entity.offset_from == token.offset_from:
                token.type = "B-" + entity.type
            else:
                token.type = "I-" + entity.type

            return
        else:
            token.type = "O"


# Write the feature extraction together with the label class and offsets into a file to future
def output_features(sid, tokens, entities, features, flag="feats"):
    if flag == "feats":

        for i, token in enumerate(tokens):
            detect_label(token, entities)
            line = [sid, token.word, str(token.offset_from), str(token.offset_to), token.type] + features[i]
            outputfile.write(" ".join(line) + "\n")
        outputfile.write("\n")
    elif flag == "megam":
        for i, token in enumerate(tokens):
            detect_label(token, entities)
            line = [token.type] + features[i]
            outputfile.write(" ".join(line) + "\n")
        outputfile.write("\n")
    else:
        print("Incorrect feature extraction type\n")
        sys.exit(1)


# Store as a list of dictionaries the word, the offset interval and the label (drug, group, brand, drug_n,...) of each entity in the sentence.
def get_entities(child):
    entities = []
    for ent in child.findall('entity'):
        entity = Entity(**ent.attrib)
        entities.append(entity)
    return entities


### MAIN ###

parser = argparse.ArgumentParser(description=
                                 """
        Compute the feature extractor of a given dataset containing XML Files.\n
        Usage: \n\npython3 feature_extractor.py --dir data/Train --type <feats|megam>\n\n
        If feats is selected returns the complete feature extractor with more detailed information about each sample.\n
        If megam is selected returns only the features of each sample.\n
        """
                                 )
parser.add_argument('--type', type=str, choices=["feats", "megam"], help='Two option of extraction: feats or megam')
parser.add_argument('--dir', type=str, help='Directory where are located the XML files')

args = parser.parse_args()

inputdir = args.dir
outputfilename = inputdir.replace("/", "_") + "_%s.dat" % args.type
outputfile = open(outputfilename, "w")

# EXTERNAL KNOWLEDGE
drug_n_list = ["angiotensins", "angiotensin", "DPCPX", "FBAL", "5-FU", "trichlorfon", "coumaphos", "18-MC",
               "Flavoridin",
               "5-oxo-desethylzaleplon", "As(V)", "arsenate", "SN38", "PTX", "palytoxin", "dehydroaripiprazole",
               "misonidazole",
               "endotoxin", "Sedatives", "picrotoxin", "amizyl", "(+)", "(-)", "phenibut", "phenazepam", "picrotoxin",
               "contortrostatin",
               "iron", "PCP", "carboxytolbutamide", "dmPGE2", "heroin", "jacalin", "MPTP", "InsP(3)", "NN", "ibogaine",
               "MHD",
               "thimerosal", "Arecoline", "TML", "18-Methoxycoronaridine", "MHD"]

# some brands extracted from chebi db
brand_names_txt = set()
with open("brand_names_dB.txt", "r") as f:
    for line in f:
        brand_names_txt.add(line)

# some groups extracted resources db
group_names_txt = set()
with open("groups_DrugBank.txt", "r") as f:
    for line in f:
        group_names_txt.add(line)


# some drug_n extracted from chebi db
drug_n_list_txt = set()
with open("compounds_dB.txt", "r") as f:
    for line in f:
        drug_n_list_txt.add(line)

for filename in os.listdir(inputdir):
    file_path = os.path.join(inputdir, filename)
    tree = ET.parse(file_path)
    root = tree.getroot()

    for child in root:
        (sid, sentence) = (child.attrib["id"], child.attrib["text"])
        tokens = tokenize(sentence)
        features = extract_features(tokens)
        entities = get_entities(child)
        output_features(sid, tokens, entities, features, flag=args.type)
outputfile.close()
