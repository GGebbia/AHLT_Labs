#!/usr/bin/env python

import xml.etree.ElementTree as ET
import os
import sys
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser

# TODO see if is used
nltk.download('punkt')

def analysis_shortened_and_masked(analysis, id_e1, id_e2, entities):

    entities_masked = []
    entities_texts = []
    # Words to mask
    for entity in entities:
        entity_text = entity.attrib['text']
        ent_id = entity.attrib['id']
        if ent_id == id_e1:
            entity1 = entity_text.lower()
        elif ent_id == id_e2:
            entity2 = entity_text.lower()
        else:
            entities_texts.append(entity_text.lower())

    # Iterate each word in sentence
    for head_node in analysis:
        for key in sorted(head_node.nodes, key=lambda key: int(key)):
            # first key is not first word (is root)
            if key == 0:
                continue
            current_word = head_node.nodes[key]['word'].lower()

            if entity1 == current_word:
                word = "<DRUG1>"
                lemma = "<DRUG1>"
                pos = "<DRUG1>"
                tag = "<DRUG1>"
            elif entity2 == current_word:
                word = "<DRUG2>"
                lemma = "<DRUG2>"
                pos = "<DRUG2>"
                tag = "<DRUG2>"
            elif current_word in entities_texts:
                word = "<DRUG_OTHER>"
                lemma = "<DRUG_OTHER>"
                pos = "<DRUG_OTHER>"
                tag = "<DRUG_OTHER>"
            else:
                word = head_node.nodes[key]['word']
                lemma = head_node.nodes[key]['lemma']
                pos = head_node.nodes[key]['address']
                tag = head_node.nodes[key]['tag']
            entities_masked.append([word, lemma, pos, tag])

    return entities_masked

def load_data(datadir):
    #TODO
    # returns dataset as a list of examples: each example correspond to a drug pair in a sentence and contains:
    # sent id, e1id, e2id, ground truth class, list of sentence tokens
    # MASK TARGET DRUGS!
    # process each file in directory

    # connect to CoreNLP server
    my_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    dataset_examples = []
    for filename in os.listdir(datadir):
        # parse XML file, obtaining a DOM tree
        file_path = os.path.join(datadir, filename)
        tree = ET.parse(file_path)
        sentences = tree.findall("sentence")
        for sentence in sentences:
            (sid, stext) = (sentence.attrib["id"], sentence.attrib["text"])
            if not stext:
                continue

            entities = sentence.findall("entity")
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
                # If the parsing is performed before the for loop it gets corrupted after first it
                analysis = my_parser.raw_parse(stext)
                analysis_masked_entities = analysis_shortened_and_masked(analysis, id_e1, id_e2, entities)
                example = [sid, id_e1, id_e2, ddi_type, analysis_masked_entities]
                dataset_examples.append(example)

    return dataset_examples


def learn (traindir, validationdir, modelname):
    '''
    learns a NN model using traindir as training data , and validationdir
    as validation data . Saves learnt model in a file named modelname
    '''
    # load train and validation data in a suitable form
    traindata = load_data(traindir)
    valdata = load_data(validationdir)

    # create indexes from training data
    max_len = 100
    idx = create_indexs(traindata, max_len)

    # build network
    model = build_network(idx)

    # encode datasets
    Xtrain = encode_words(traindata,idx)
    Ytrain = encode_tags(traindata, idx)
    Xval = encode_words(valdata, idx)
    Yval = encode_tags(valdata, idx)

    # train model
    model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval))

    # save model and indexs , for later use in prediction
    save_model_and_indexs(model, idx, modelname)


### MAIN ###
data_loaded = load_data("data/Train")