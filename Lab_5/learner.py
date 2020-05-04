#!/bin/python

import xml.etree.ElementTree as ET
import os
import sys
from utils.models import Token, Entity
from utils.functions import get_entities, detect_label, tokenize

def load_data(datadir):
    # returns dataset as a dict of examples: where the keys are the identities and the value is the tokenization list of the entity.

    dataset_dict = {}
    for filename in os.listdir(datadir):
        # parse XML file, obtaining a DOM tree
        file_path = os.path.join(datadir, filename)
        tree = ET.parse(file_path)
        sentences = tree.findall("sentence")
        for sentence in sentences:
            (sid, stext) = (sentence.attrib["id"], sentence.attrib["text"])
            if not stext:
                continue
            entities = get_entities(sentence)            
            tokens = tokenize(stext)
            for token in tokens:
                detect_label(token, entities)

            dataset_dict[sid] = [token.__repr__() for token in tokens]

    return dataset_dict

def create_indexs(datadir, max_length):
    '''
    Returns a mapping of each word seen in the data with an integer. Also a mapping of each tag (null, mechanism, advise,
    effect, int). Also returns maxlen
    It has an <UNK> token and <PAD> token that will be used for filling the encoded sentence till max_length
    '''
    tags = ['null', 'mechanism', 'advise', 'effect', 'int']
    all_indexes = {}
    word_indexes = {}
    tags_indexes = dict(enumerate(tags))
    all_indexes['maxlen'] = max_length
    all_indexes['tags'] = tags_indexes


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
print(data_loaded)
