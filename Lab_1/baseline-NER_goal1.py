from __future__ import print_function
import xml.etree.ElementTree as ET

import nltk
from nltk.tokenize import WhitespaceTokenizer

import os
import sys
import re

nltk.download('punkt')

stopWords = set(stopwords.words('english'))

# Tokenize words
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

            new_tokens.append((word, offset_from, offset_to))
            new_tokens.append((punct, punct_offset_from, punct_offset_to))
        else:
            new_tokens.append((word, offset_from, offset_to))

    return new_tokens

# If a word has some of these symbols at the end we remove it and return the stripped word and tell if we stripped it
# so to take it into account when computing the offsets
def strip_word_end(word):
    punctuation = [",", ":", ";", ")", "!", "?"]
    if any(word.endswith(punct) for punct in punctuation):
        word = word[:-1]
        return True, word
    else:
        return False, word

def extract_entities(tokenized_list):
    # where we will store all the entities detected
    entities_list = []
    # we use the info of the last detected type to merge two equal subsequent types
    last_type = None
    # we might remove some punctuation characters, this bool tells us if we did that or not
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
            last_type = None
            continue
        elif len(word) < 2:
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
        if not word or any(unwanted_word.lower() in word.lower() for unwanted_word in unwanted_word_list):
            last_type = None
            continue

        # check if word is in some of the groups we extracted manually from the training dataset
        elif any(group_name in word.lower() for group_name in group_name_list):
            if last_type is not None:
                prev_word, prev_offset_from, _ = tokenized_list[i - 1]
                # Remove drug or brand if it was added since the next word is acid
                entities_list.pop(-1)
                d["name"] = prev_word + " " + word
                d["type"] = "group"
                d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
                last_type = "group"
            else:
                d["name"] = word
                d["type"] = "group"
                d["offset"] = "{}-{}".format(offset_from, offset_to)
                last_type = "group"
        # check if word is in some of the drug_n we extracted manually from the training dataset
        elif any(drug_n_word.lower() in word.lower() for drug_n_word in drug_n_list):
            if last_type is not None:
                prev_word, prev_offset_from, _ = tokenized_list[i - 1]
                # Remove drug or brand if it was added since the next word is acid
                entities_list.pop(-1)
                d["name"] = prev_word + " " + word
                d["type"] = "drug_n"
                d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
                last_type = "drug_n"
            else:
                d["name"] = word
                d["type"] = "drug_n"
                d["offset"] = "{}-{}".format(offset_from, offset_to)
                last_type = "drug_n"

        # check if word is in some of the drugs we extracted manually from the training dataset
        elif any(drug_word.lower() in word.lower() for drug_word in drug_list):
            if last_type is not None:
                prev_word, prev_offset_from, _ = tokenized_list[i - 1]
                # Remove drug or brand if it was added since the next word is acid
                entities_list.pop(-1)
                d["name"] = prev_word + " " + word
                d["type"] = "drug"
                d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
                last_type = "drug"
            else:
                d["name"] = word
                d["type"] = "drug"
                d["offset"] = "{}-{}".format(offset_from, offset_to)
                last_type = "drug"

        # we observed that words that acid usually has the structure : X acid, being X of some type, so we check the
        # previous type and merge them if are subsequent
        elif word.lower() == "acid":
            if last_type != None:
                entities_list.pop(-1)
                prev_word, prev_offset_from, _ = tokenized_list[i - 1]
                # Remove drug or brand if it was added since the next word is acid
                d["name"] = prev_word + " acid"
                d["type"] = last_type
                d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
                last_type = last_type

        # If word is vitamin, we store with the second letter
        elif word.lower() == "vitamin":
            if i != list_length:
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
            else:
                d["name"] = word
                d["type"] = "group"
                d["offset"] = "{}-{}".format(offset_from, offset_to)
                last_type = "group"

        # from the train dataset we observed some common sufixes and prefixes among drugs and groups. The groups end with s as they
        # are plural, so check if the word ends has a plural suffix, if so it probably is a group
        elif any(suffix in word.lower() for suffix in suffixes_plural_list):
            if last_type == "group":
                prev_word, prev_offset_from, _ = tokenized_list[i - 1]
                 # Remove drug or brand if it was added since the next word is acid
                entities_list.pop(-1)
                d["name"] = prev_word + " " + word
                d["type"] = "group"
                d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
                last_type = "group"

            else:
                d["name"] = word
                d["type"] = "group"
                d["offset"] = "{}-{}".format(offset_from, offset_to)
                last_type = "group"

       # If word contains any of the suffixes in the list we wrote form the training db, then mark it as drug
        elif any(suffix in word for suffix in suffixes_list):
            if last_type == "drug":
                prev_word, prev_offset_from, _ = tokenized_list[i - 1]
                # Remove drug or brand if it was added since the next word is acid
                entities_list.pop(-1)
                d["name"] = prev_word + " " + word
                d["type"] = "drug"
                d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
                last_type = "drug"
            else:
                d["name"] = word
                d["type"] = "drug"
                d["offset"] = "{}-{}".format(offset_from, offset_to)
                last_type = "drug"

        # if first letter of the word is capital and not at the beginning of sentence,
        # and we have not classified it yet, it problaly is a brand
        elif (word[0].isupper() and offset_from != 0):
            if last_type == "brand":
                prev_word, prev_offset_from, _ = tokenized_list[i - 1]
                # Remove drug or brand if it was added since the next word is acid
                entities_list.pop(-1)
                d["name"] = prev_word + " " + word
                d["type"] = "brand"
                d["offset"] = "{}-{}".format(prev_offset_from, offset_to)
                last_type = "brand"
            else:
                d["name"] = word
                d["type"] = "brand"
                d["offset"] = "{}-{}".format(offset_from, offset_to)
                last_type = "brand"

        # if we haven't classified the word yet probably is not an entity, don't store it and set as NONE
        else:
            last_type = None

        # If this word was ending with , or : next word not merged with this one, so set last type to none
        if word_stripped:
            last_type = None
            word_stripped = False

        # if we set the word to the dictionary, append it to the list with all the other already detected entities
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
outputfilename = "./task9.1_TrainGianMarc_3.txt"
outputfile = open(outputfilename, "w")

# suffixes observed from the training db, which are for drugs
suffixes_list = [line.strip() for line in open("sufixes_no_knowledge.txt", "r")]
# same suffixes but in plural, they are used for groups
suffixes_plural_list = [line.strip() for line in open("sufixes_plural_no_knowledge.txt", "r")]
# some manually anotated drug_n from de train dB. Drug_n are usually hard to detect so we used this method to ensure we detect these.
drug_n_list = ["angiotensins", "angiotensin", "DPCPX", "FBAL", "5-FU", "trichlorfon", "coumaphos", "18-MC", "Flavoridin",
               "5-oxo-desethylzaleplon", "As(V)", "arsenate","SN38", "PTX", "palytoxin", "dehydroaripiprazole","misonidazole",
               "endotoxin", "Sedatives", "picrotoxin", "amizyl", "phenibut", "phenazepam", "picrotoxin", "contortrostatin",
               "iron", "PCP", "carboxytolbutamide", "dmPGE2", "heroin", "jacalin", "MPTP", "InsP(3)", "NN", "ibogaine", "MHD",
               "thimerosal", "Arecoline", "TML", "18-Methoxycoronaridine", "MHD"]
# some groups from the train db we were not detecting with rules
group_name_list = ["Antacids", "beta", "alpha", "anti", "NSAID", "NSAIDs", "anticoagulant", "TCA", "TCAs", "polymyxins", "coumarin", "Androgens",
                   "diuretic", "diuretics", "Digitalis", "nitrosourea", "hypoglycemic", "agents", "barbiturates", "Corticosteroids",
                   "cortico-steroids", "systemic", "solvent", "Drugs", "surfactant", "bronchodilators", "preparations", "inhibitors"]
# some drugs from the train db we were not detecting with rules
drug_list = ["1,25(OH)2D3", "etodolac", "Rifabutin", "chloroquine", "CCNU", "CYP2D6", "CYP3A4", "MTX", "CYP2C9", "corticosteroid", "dapsone", "anakinra"]
# some words we were detecting as entities during training which should be ignored
unwanted_word_list = ["CYP3A", "3A", "P450", "Table", "environment", "identification", "provided", "Guidelines", "risk", "ironically", "manner", "cannot"]

for filename in os.listdir(inputdir):
    file_path = os.path.join(inputdir, filename)
    tree = ET.parse(file_path)
    root = tree.getroot()

    for child in root:
        (sid, sentence) = (child.attrib["id"], child.attrib["text"])
        tokenized_list = tokenize(sentence)
        entities = extract_entities(tokenized_list)
        output_entities(sid, entities, outputfile)
evaluate(inputdir, outputfilename)
outputfile.close()
