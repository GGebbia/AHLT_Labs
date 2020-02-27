from __future__ import print_function
import xml.etree.ElementTree as ET

import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords

import os
import sys
import re

nltk.download('stopwords')
nltk.download('punkt')

stopWords = set(stopwords.words('english'))


# Replace punctuation symbols by space in order to maintain the offset.
def remove_punctuation(sentence):
    sentence = re.sub(r'\([^)]*\)', ' ', sentence)
    sentence = re.sub("?!", " ", sentence)
    return sentence


def tokenize(sentence):
    span_generator = WhitespaceTokenizer().span_tokenize(sentence)
    return [(sentence[span[0]: span[1]], span[0], span[1] - 1) for span in span_generator]


def strip_word_end(word):
    punctuation = [",", ":", ";", ")", "!", "?"]
    if any(word.endswith(punct) for punct in punctuation):
        word = word[:-1]
        return True, word
    else:
        return False, word


# TODO afegir knowledge de english words (the, in etc. + drugs i brands que sabem de test)
# TODO mirar com detectar brands

def extract_entities(tokenized_list):
    entities_list = []
    last_type = None
    word_stripped = False
    list_length = len(tokenized_list) - 1
    # we use skip word when we want to store next token (vitamine D),
    # when we reach vitamine, we take D also, so when we take the next token (D), we skip it
    skip_word = False
    global suffixes_list

    for i, token in enumerate(tokenized_list):
        word, offset_from, offset_to = token
        d = {}

        if skip_word:
            skip_word = False
            continue

        if word.lower() in stopWords:
            last_type = None
            continue

        # remove . from last word
        if i == list_length and word.endswith("."):
            word = word[:-1]
            offset_to -= 1
            # this should only happen if last word is a single ".""
        if not word:
            continue

        # strip symbols from end of the word
        continue_stripping_word = True

        while continue_stripping_word:
            continue_stripping_word, word = strip_word_end(word)
            if continue_stripping_word:
                offset_to-= 1
                word_stripped = True
        if not word:
            continue

        # If word contains any of the suffixes in the list, then mark it as drug
        if any(suffix in word for suffix in suffixes_list):
            d["name"] = word
            d["type"] = "drug"
            d["offset"] = "{}-{}".format(offset_from, offset_to)
            last_type = "drug"

        # If word is vitamin, we store with the second letter
        elif word == "vitamin":
            post_word, _, post_offset_to = tokenized_list[i + 1]
            continue_stripping_word = True
            while continue_stripping_word:
                continue_stripping_word, post_word = strip_word_end(post_word)
                if continue_stripping_word:
                    post_offset_to -= 1
                    word_stripped = True
            d["name"] = "{} {}".format(word, post_word)
            d["type"] = "group"
            d["offset"] = "{}-{}".format(offset_from, post_offset_to)
            last_type = "group"
            skip_word = True

        # todo canviar a brand o droga depenent de sufix
        elif (word[0].isupper() and offset_from != 0) or (word.isupper()):
            d["name"] = word
            d["type"] = "brand"  # Posar pes per fer probabilitat 2/3 si es drug, 1/3 si es brand
            d["offset"] = "{}-{}".format(offset_from, offset_to)
            last_type = "brand"

        # TODO: si la paraula seguent tambe te el mateix tipus, merge
        elif word.lower() == "acid":
            prev_word, prev_offset_from, _ = tokenized_list[i - 1]

            # Remove drug or brand if it was added since the next word is acid
            if last_type != None:
                entitites_list.pop(-1)
            d["name"] = prev_word + " acid"
            d["type"] = "drug"
            d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
            last_type = "drug"

        else:
            last_type = None

        # If this word was ending with , or : next word not mergerd with this one
        if word_stripped:
            last_type = None
            word_stripped = False

        if d.keys(): entities_list.append(d)
    return entities_list


def output_entities(sid, entities, outputfile):
    for ent in entities:
        line = '|'.join([sid, ent["offset"], ent["name"], ent["type"]])
        outputfile.write(line + "\n")


def evaluate(inputdir, outputfile):
    os.system("java -jar eval/evaluateNER.jar " + inputdir + " " + outputfile)


### VARIABLES
inputdir = sys.argv[1]
outputfilename = "./task9.1_GianMarc_1.txt"
outputfile = open(outputfilename, "w")
suffixes_list = [line.strip() for line in open("sufixes_devel.txt", "r")]

for filename in os.listdir(inputdir):
    file_path = os.path.join(inputdir, filename)
    tree = ET.parse(file_path)
    root = tree.getroot()

    for child in root:
        (sid, sentence) = (child.attrib["id"], child.attrib["text"])
        # sentence = remove_punctuation(sentence)
        tokenized_list = tokenize(sentence)
        entities = extract_entities(tokenized_list)
        output_entities(sid, entities, outputfile)
evaluate(inputdir, outputfilename)
outputfile.close()
