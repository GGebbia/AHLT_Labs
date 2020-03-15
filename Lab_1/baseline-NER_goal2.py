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

# stopWords which we will not classify as entities
stopWords = set(stopwords.words('english'))

# Replace punctuation symbols by space in order to maintain the offset.
def remove_punctuation(sentence):
    sentence = re.sub(r'\([^)]*\)', ' ', sentence)
    sentence = re.sub("?!", " ", sentence)
    return sentence

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
    skip_word_times = 0
    global suffixes_list

    for i, token in enumerate(tokenized_list):
        word, offset_from, offset_to = token
        d = {}

        if skip_word:
            if skip_word_times != 0:
                skip_word_times -= 1
                continue
            else:
                skip_word = False
                last_type = None
                continue
        elif len(word) < 2:
            last_type = None
            continue

        # if word is a stopword, set type to none and take next word
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
        if not word or any(unwanted_word.lower() in word.lower() for unwanted_word in unwanted_word_list):
            last_type = None
            continue

        # Check each line of some of the drug names on drugs.com , if they exactly match, classify it!
        for line in drug_names_txt:
            entire_line = []
            number_of_words_in_line = 0
            same_line = False
            last_offset_to = 0
            line_words = line.split()
            for word_index, line_word in enumerate(line_words):
                if i + word_index > list_length:
                    break
                word_to_check, _, last_offset_to = tokenized_list[i + word_index]
                if word_to_check.lower() == line_word.lower():
                    entire_line.append(word_to_check)
                    number_of_words_in_line += 1
                    same_line = True
                else:
                    same_line = False
                    break
            if same_line:
                d["name"] = " ".join(entire_line)
                d["type"] = "drug"
                d["offset"] = "{}-{}".format(offset_from, last_offset_to)
                last_type = "drug"
                skip_word = True
                skip_word_times = number_of_words_in_line
                break

        if skip_word:
            if d.keys(): entities_list.append(d)
            word_stripped = False
            continue

        # check if word is in some of the groups we extracted manually from the training dataset
        if any(group_name.lower() in word.lower() for group_name in group_name_list):
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
        elif any(drug_n_word.lower() in word.lower() for drug_n_word in drug_n_list)\
                or "(+)" in word or "(-)" in word:
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

        # from the train dataset we observed some common sufixes and prefixzes among drugs and groups. The groups end with s as they
        # are plural, so check if the word ends with a plural suffix or begins with, if so it probably is a group
        elif any(word.endswith(suffix.lower()) or word.startswith(suffix.lower()) for suffix in suffixes_plural_list):
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

        # from the train dataset we observed some common sufixes and prefixzes among drugs.
        # check if the word ends with a suffix or begins with a prefix, if so it probably is a drug
        elif any(word.endswith(suffix.lower()) or word.startswith(suffix.lower()) for suffix in
                     suffixes_list):
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
        else:
            # last type is none as we will be checking exact match in entire lines, so no merging
            last_type = None
            # Check each line of some of the brand names on chebi db , if they exactly match, classify it!
            for line in brand_names_txt:
                entire_line = []
                number_of_words_in_line = 0
                same_line = False
                last_offset_to = 0
                line_words = line.split()
                # check each word in line for exact match
                for word_index, line_word in enumerate(line_words):
                    if i + word_index > list_length:
                        break
                    word_to_check, _, last_offset_to = tokenized_list[i + word_index]
                    if word_to_check.lower() == line_word.lower():
                        entire_line.append(word_to_check)
                        number_of_words_in_line += 1
                        same_line = True
                    else:
                        same_line = False
                        break
                if same_line:
                    d["name"] = " ".join(entire_line)
                    d["type"] = "brand"
                    d["offset"] = "{}-{}".format(offset_from, last_offset_to)
                    last_type = "brand"
                    skip_word = True
                    skip_word_times = number_of_words_in_line
                    break
            # if found word, break loop
            if skip_word:
                if d.keys(): entities_list.append(d)
                word_stripped = False
                continue
            # Check each line of some of the group names on chebi db , if they exactly match, classify it!
            for line in groups_list_txt:
                entire_line = []
                number_of_words_in_line = 0
                same_line = False
                last_offset_to = 0
                line_words = line.split()
                # check each word in line for exact match
                for word_index, line_word in enumerate(line_words):
                    if i + word_index > list_length:
                        break
                    word_to_check, _, last_offset_to = tokenized_list[i + word_index]
                    if word_to_check.lower() == line_word.lower():
                        entire_line.append(word_to_check)
                        number_of_words_in_line += 1
                        same_line = True
                    else:
                        same_line = False
                        break
                if same_line:
                    d["name"] = " ".join(entire_line)
                    d["type"] = "group"
                    d["offset"] = "{}-{}".format(offset_from, last_offset_to)
                    last_type = "group"
                    skip_word = True
                    skip_word_times = number_of_words_in_line
                    break
            # if found word, break loop
            if skip_word:
                if d.keys(): entities_list.append(d)
                word_stripped = False
                continue

            # Check each line of some of the drug_n names on chebi db , if they exactly match, classify it!
            for line in drug_n_list_txt:
                entire_line = []
                number_of_words_in_line = 0
                same_line = False
                last_offset_to = 0
                line_words = line.split()
                for word_index, line_word in enumerate(line_words):
                    if i + word_index > list_length:
                        break
                    word_to_check, _, last_offset_to = tokenized_list[i + word_index]
                    if word_to_check.lower() == line_word.lower():
                        entire_line.append(word_to_check)
                        number_of_words_in_line += 1
                        same_line = True
                    else:
                        same_line = False
                        break
                if same_line:
                    d["name"] = " ".join(entire_line)
                    d["type"] = "drug_n"
                    d["offset"] = "{}-{}".format(offset_from, last_offset_to)
                    last_type = "drug_n"
                    skip_word = True
                    skip_word_times = number_of_words_in_line
                    break
        # if found word, break loop
        if skip_word:
            if d.keys(): entities_list.append(d)
            word_stripped = False
            continue

        # If this word was ending with , or : next word not mergerd with this one
        if word_stripped:
            last_type = None
            word_stripped = False
        # if agents was not merged with any words it is never an entity (observed from training db), so remove it
        if d.keys() and d["name"].lower() == "agents":
            d = {}
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
outputfilename = "./task9.1_GianMarc_3.txt"
outputfile = open(outputfilename, "w")


groups_list_txt = []
# some groups extracted from chebi db
with open("groups.txt", "r") as f:
    for line in f:
        groups_list_txt.append(line)

# some drug_n extracted from chebi db
drug_n_list_txt = set()
with open("compounds_dB.txt", "r") as f:
    for line in f:
        drug_n_list_txt.add(line)

# some brands extracted from chebi db
brand_names_txt = set()
with open("brand_names_dB.txt", "r") as f:
    for line in f:
        brand_names_txt.add(line)

# some drugs taken from drugs.com
drug_names_txt = set()
with open("drug_names_dB.txt", "r") as f:
    for line in f:
        drug_names_txt.add(line)

# suffixes for drugs
suffixes_list = [line.strip() for line in open("sufixes_external_knowledge.txt", "r")]
# same suffixes but in plural, they are used for groups
suffixes_plural_list = [line.strip() for line in open("sufixes_plural_external_knowledge.txt", "r")]
# some manually anotated drug_n from de train dB. Drug_n are usually hard to detect so we used this method to ensure we detect these.
drug_n_list = ["angiotensins", "angiotensin", "DPCPX", "FBAL", "5-FU", "trichlorfon", "coumaphos", "18-MC", "Flavoridin",
               "5-oxo-desethylzaleplon", "As(V)", "arsenate","SN38", "PTX", "palytoxin", "dehydroaripiprazole","misonidazole",
               "endotoxin", "Sedatives", "picrotoxin", "amizyl", "phenibut", "phenazepam", "picrotoxin", "contortrostatin",
               "iron", "PCP", "carboxytolbutamide", "dmPGE2", "heroin", "jacalin", "MPTP", "InsP(3)", "NN", "ibogaine", "MHD",
               "thimerosal", "Arecoline", "TML", "18-Methoxycoronaridine", "MHD"]
# some groups from the train db we were not detecting with rules
group_name_list = ["Antacids", "alpha", "anti", "NSAID", "NSAIDs", "anticoagulant", "TCA", "TCAs", "polymyxins", "coumarin", "Androgens",
                   "diuretic", "diuretics", "Digitalis", "nitrosourea", "hypoglycemic", "agents", "barbiturates", "Corticosteroids",
                   "cortico-steroids", "systemic", "solvent", "surfactant", "channel", "bronchodilators", "preparations", "inhibitors", ]
# some drugs from the train db we were not detecting with rules
drug_list = ["1,25(OH)2D3", "etodolac", "Rifabutin", "chloroquine", "CCNU", "CYP2D6", "CYP3A4", "MTX", "CYP2C9", "corticosteroid", "dapsone", "anakinra", "sodium"]
# some words we were detecting as entities during training which should be ignored
unwanted_word_list = ["CYP3A", "3A", "P450", "Table", "environment", "identification", "provided", "Guidelines", "risk", "ironically", "manner", "cannot"]

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
