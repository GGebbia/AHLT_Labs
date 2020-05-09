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

def create_indexs(loaded_dataset, max_length):
    '''
    Returns a mapping of each word seen in the data with an integer. Also a mapping of each tag (null, mechanism, advise,
    effect, int). Also returns maxlen
    It has an <UNK> token and <PAD> token that will be used for filling the encoded sentence till max_length
    No need to encode pos as it is already an integer
    '''
    all_indexes = {}
    types = ['null', 'mechanism', 'advise', 'effect', 'int']
    types_indexes = dict(enumerate(types))
    word_indexes = {"<UNK>":0, "<PAD>":1}
    lemma_indexes = {"<UNK>": 0, "<PAD>": 1}
    tag_indexes = {"<UNK>": 0, "<PAD>": 1}
    word_counter = lemma_counter = tag_counter = 2

    # connect to CoreNLP server
    for data in loaded_dataset:
        for entities in data[4]:
            word = entities[0]
            lemma =  entities[1]
            tag = entities[3]
            if word not in word_indexes:
                word_indexes[word] = word_counter
                word_counter += 1
            if lemma not in lemma_indexes:
                lemma_indexes[lemma] = lemma_counter
                lemma_counter += 1
            if tag not in tag_indexes:
                tag_indexes[tag] = tag_counter
                tag_counter += 1

    all_indexes['words'] = word_indexes
    all_indexes['lemmas'] = lemma_indexes
    all_indexes['types'] = types_indexes
    all_indexes['tags'] = tag_indexes
    all_indexes['maxlen'] = max_length
    return all_indexes


def build_network(idx):
    '''
    Input : Receives the index dictionary with the encondings of words and
    tags, and the maximum length of sentences.
    Output : Returns a compiled Keras neural network
    '''
    # sizes
    n_words = len(idx['words'])
    n_tags = len(idx['tags'])
    max_len = idx['maxlen']
    # create network layers
    inp = Input(shape=(max_len,))
    ## ... add missing layers here ... #
    # out = # final output layer
    # create and compile model
    model = Model(inp, out)
    model.compile()  # set appropriate parameters ( optimizer, loss, etc)

    return model

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
            sentence_words_encoded.append(lemma_encoded)
            sentence_pos.append(pos)
            sentence_tags_encoded.append(tag_encoded)
        # Apply padding if needed
        while len(sentence_words_encoded) < idx['maxlen']:
            sentence_words_encoded.append(idx['words']['<PAD>'])
            sentence_lemmas_encoded.append(idx['lemmas']['<PAD>'])
            sentence_pos.append(0)
            sentence_tags_encoded.append(idx['tags']['<PAD>'])
        encoded_words_all.append(sentence_words_encoded)
        encoded_lemmas_all.append(sentence_lemmas_encoded)
        pos_all.append(sentence_pos)
        encoded_tags_all.append(sentence_tags_encoded)
    # Return the 4 lists generated
    return encoded_words_all, encoded_lemmas_all, pos_all, encoded_tags_all


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
data_loaded = load_data("data/Devel")
indexs_dict = create_indexs(data_loaded, 100)
encoded_words =  encode_words(data_loaded, indexs_dict)
