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

# Replace punctuation symbols by space in order to maintain the offset.
def remove_punctuation(sentence):
    sentence = re.sub(r'\([^)]*\)', ' ', sentence)
    sentence = re.sub("?!", " ", sentence)
    return sentence

def tokenize(sentence):
    span_generator = WhitespaceTokenizer().span_tokenize(sentence)
    return [(sentence[span[0]: span[1]], span[0], span[1]-1) for span in span_generator]

# TODO afegir knowledge de english words (the, in etc. + drugs i brands que sabem de test)
# TODO mirar com detectar brands
# TODO les vitamines posa "vitamine D" i detectem D com a drug, mirar si paraula anterior es vitamine i actual és una lletra i no contar com a drug
def extract_entities(tokenized_list):
    entities_list = []
    last_type = None
    word_stripped = False
    list_length = len(tokenized_list)-1
    for i, token in enumerate(tokenized_list):
        word, offset_from, offset_to = token
        d = {}

        # remove . from last word
        if i == list_length and word.endswith("."):
            word = word[:-1]
        # strip , and : from word
        elif any(punct in word for punct in punctuation):
            print(word)
            word = word[:-1]
            word_stripped = True

        # this should only happen if last word is a single ".""
        if not word or len(word)<=2:
            continue

        # If word contains any of the suffixes in the list, then mark it as drug
        if any(suffix in word for suffix in suffixes_list):
            d["name"] = word
            d["type"] = "drug"
            d["offset"] = "{}-{}".format(offset_from, offset_to)
            last_type = "drug"

        # If word is vitamin, we store with the second letter
        if word == "vitamin":
            post_word, _, post_offset_to = tokenized_list[i+1]
            d["name"] = "{} {}".format(word, post_word)
            d["type"] = "drug"
            d["offset"] = "{}-{}".format(offset_from, post_offset_to)
            last_type = "drug"

        if (word[0].isupper() and offset_from != 0) or (word.isupper()):
            d["name"] = word
            d["type"] = "drug" # Posar pes per fer probabilitat 2/3 si es drug, 1/3 si es brand
            d["offset"] = "{}-{}".format(offset_from, offset_to)
            last_type = "drug"

        #TODO: si la paraula seguent tambe te el mateix tipus, merge
        elif word.lower() == "acid":
            prev_word, prev_offset_from, _ = tokenized_list[i-1]

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
    os.system("java -jar eval/evaluateNER.jar "+ inputdir + " " + outputfile)

### VARIABLES
inputdir = sys.argv[1]
outputfilename="./output.txt"
outputfile = open(outputfilename, "w")
suffixes_list = [line.strip() for line in open("sufixes_devel.txt","r")]
punctuation = [",", ":", ";", ")", "!", "?"]
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
