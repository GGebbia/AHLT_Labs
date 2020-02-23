from __future__ import print_function
import xml.etree.ElementTree as ET

import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords

import os
import sys

nltk.download('stopwords')
nltk.download('punkt')

def tokenize(sentence, span_generator):
    return [(sentence[span[0]: span[1]], span[0], span[1]-1) for span in span_generator]

def extract_entities(tokenized_list):
    entities_list = []
    last_type = None
    for i, token in enumerate(tokenized_list):
        word, offset_from, offset_to = token
        d = {}

        if (word[0].isupper() and offset_from != 0) or (word.isupper()) or (word=="Spermine"):
            d["name"] = word
            d["type"] = "drug" # Posar pes per fer probabilitat 2/3 si es drug, 1/3 si es brand
            d["offset"] = "{}-{}".format(offset_from, offset_to)
            last_type = "drug"


        if last_type != None:
            pass

        #TODO: si la paraula seguent tambe te el mateix tipus, merge
        elif word.lower() == "acid":
            prev_word, prev_offset_from, _ = tokenized_list[i-1]

            # Remove drug or brand if it was added since the next word is acid
            if last_type != None:
                entitites_list.pop(-1)

            d["name"] = prev_word + " acid"
            d["type"] = "drug"
            d["offset"] = "{}-{}".format(prev_offset_from, offset_to)

        last_type = None
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

tree = ET.parse("./data/Devel/2981704.xml")
root = tree.getroot()

for child in root:
    (sid, text) = (child.attrib["id"], child.attrib["text"])
    span_generator = WhitespaceTokenizer().span_tokenize(text)
    tokenized_list = tokenize(text, span_generator)
    entities = extract_entities(tokenized_list)
    output_entities(sid, entities, outputfile)

evaluate(inputdir, outputfilename)
outputfile.close()
