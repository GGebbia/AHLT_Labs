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
        self.type = "O"                                # Initialize all tokens with type O as non-drugs non-brands and non-groups

    def word_iscapitalized(self):
        return self.word[0].isupper()


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
        if (len(word) > 1) and ( word.endswith(',') or word.endswith('.') ):
            punct = word[-1]
            punct_offset_from = offset_to
            punct_offset_to = offset_to

            word = word[:-1]
            offset_to -= 1

            new_tokens.append(Token(word, offset_from, offset_to))
            new_tokens.append(Token(punct, punct_offset_from, punct_offset_to))
        else:
            new_tokens.append(Token(word, offset_from, offset_to))

    return new_tokens

# Given a token list, collect the most relevant features to store in a list.
def extract_features(tokens):
    features = []

    for i, token in enumerate(tokens, 0):
        word_features = []

        # COMMON FEATURES
        form = token.word
        word_features.append("form={}".format(form))
        suf4 = token.word[-4:]
        word_features.append("suf4={}".format(suf4))
        if i != 0:
            prev = tokens[i-1].word
        else:
            prev = "_BoS_"
        word_features.append("prev={}".format(prev))

        try:
            _next = tokens[i+1].word
            word_features.append("next={}".format(_next))
        except:
            pass

        # SPECIFIC FEATURES
        if token.word_iscapitalized():
            word_features.append("capitalized")

        features.append(word_features)

    return features

def detect_label(token, entities):

    for entity in entities:
        # If the two offsets are equal, then it corresponds to the same word and type.
        if token.offset_from == entity.offset_from and token.offset_to == entity.offset_to:
            token.type = "B-" + entity.type
        # If the token offset interval is inside the entity offset interval, then it is a first or the continuation of a type sequence.
        elif entity.offset_from <= token.offset_from and token.offset_to <= entity.offset_to:
            if entity.offset_from == token.offset_from:
                token.type = "B-" + entity.type
            else:
                token.type = "I-" + entity.type
        else:
            token.type = "O"

# Write the feature extraction together with the label class and offsets into a file to future
def output_features(sid, tokens, entities, features, flag="feats"):


    if flag == "feats":

        for i,token in enumerate(tokens):
            detect_label(token, entities)
            line = [sid, token.word, str(token.offset_from), str(token.offset_to), token.type] + features[i]
            outputfile.write(" ".join(line) + "\n")
    elif flag == "megam":
        for i, token in enumerate(tokens):
            detect_label(token, entities)
            line = [token.type] + features[i]
            outputfile.write(" ".join(line) + "\n")
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
parser.add_argument('--type', type=str, choices=["feats","megam"], help='Two option of extraction: feats or megam')
parser.add_argument('--dir', type=str, help='Directory where are located the XML files')

args = parser.parse_args()

inputdir = args.dir
outputfilename = "%s.dat" % args.type
outputfile = open(outputfilename, "w")

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

