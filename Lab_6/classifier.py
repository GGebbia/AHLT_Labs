#!/usr/bin/env python
import os
import sys

import pickle
import numpy as np

from nltk.parse.corenlp import CoreNLPDependencyParser
import xml.etree.ElementTree as ET

from keras.models import Model
from keras.models import load_model

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
            pos = head_node.nodes[key]['address']

            if entity1 == current_word:
                word = "<DRUG1>"
                lemma = "<DRUG1>"
                tag = "<DRUG1>"
            elif entity2 == current_word:
                word = "<DRUG2>"
                lemma = "<DRUG2>"
                tag = "<DRUG2>"
            elif current_word in entities_texts:
                word = "<DRUG_OTHER>"
                lemma = "<DRUG_OTHER>"
                tag = "<DRUG_OTHER>"
            else:
                word = head_node.nodes[key]['word']
                lemma = head_node.nodes[key]['lemma']
                tag = head_node.nodes[key]['tag']
            entities_masked.append([word, lemma, pos, tag])

    return entities_masked

def load_data(datadir):
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

def encode_words(dataset, idx):
    encoded_words_all = []
    encoded_lemmas_all = []
    encoded_tags_all = []
    pos_all = []
    # Iterate through sentences
    for data in dataset:
        sentence_words_encoded = []
        sentence_lemmas_encoded = []
        sentence_pos = []
        sentence_tags_encoded = []
        # Encode words on each sentence
        for analysis in data[4]:
            word = analysis[0]
            lemma = analysis[1]
            pos = analysis[2]
            tag = analysis[3]
            try:
                word_encoded = idx['words'][word]
            except KeyError:
                word_encoded = idx['words']['<UNK>']
            try:
                lemma_encoded = idx['lemmas'][lemma]
            except KeyError:
                lemma_encoded = idx['lemmas']['<UNK>']
            try:
                tag_encoded = idx['tags'][tag]
            except KeyError:
                tag_encoded = idx['tags']['<UNK>']
            sentence_words_encoded.append(word_encoded)
            sentence_lemmas_encoded.append(lemma_encoded)
            sentence_pos.append(pos)
            sentence_tags_encoded.append(tag_encoded)
            if len(sentence_words_encoded) == idx['maxlen']:
                break
        # Apply padding if needed
        while len(sentence_words_encoded) < idx['maxlen']:
            sentence_words_encoded.append(idx['words']['<PAD>'])
            sentence_lemmas_encoded.append(idx['lemmas']['<PAD>'])
            sentence_pos.append(0)
            sentence_tags_encoded.append(idx['tags']['<PAD>'])
        encoded_words_all.append(np.array(sentence_words_encoded, dtype=np.int32))
        encoded_lemmas_all.append(sentence_lemmas_encoded)
        pos_all.append(sentence_pos)
        encoded_tags_all.append(sentence_tags_encoded)
    # Return the 4 lists generated
    return np.array(encoded_words_all, dtype=np.int32), encoded_lemmas_all, pos_all, encoded_tags_all

def load_model_and_indexs(filename):
    model = load_model(filename + '.h5')
    indexs = pickle.load(open(filename + '.pkl', 'rb'))
    return model, indexs

def output_interactions(dataset, preds, outfilename):
    outputfile = open(outfilename, "w")
    for index, data in enumerate(dataset):
        if preds[index] == "null":
            interaction = "0"
        else:
            interaction = "1"
        line = [data[0], data[1], data[2], interaction, preds[index]]
        outputfile.write("|".join(line) + "\n")
    outputfile.close()

def evaluation(inputdir, outputfile):
    """
    Receives an input directory and the outputfile to evaluate the predicted labels with the evaluateNER.jar program.
    """
    os.system("java -jar eval/evaluateDDI.jar " + inputdir + " " + outputfile)

def predict(modelname, datadir, outfile):
    '''
    Loads a NN model from file ’modelname ’ and uses it to extract drugs
    in datadir . Saves results to ’outfile ’ in the appropriate format .
    '''

    # load model and associated encoding data
    model, idx = load_model_and_indexs(modelname)
    # load data to annotate
    testdata = load_data(datadir)

    # encode dataset
    X, _, _, _ = encode_words(testdata, idx)

    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each pair
    tags_map = {v: k for k, v in idx['types'].items()}
    one_hot_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    Y = [tags_map[one_hot_map[np.argmax(y)]] for y in Y]

    # extract entities and dump them to output file
    output_interactions(testdata, Y, outfile)

    # evaluate using official evaluator
    evaluation(datadir, outfile)

predict('MODELS/myModel', 'data/Devel', 'class_predictions.txt')