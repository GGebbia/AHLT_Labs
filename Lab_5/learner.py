#!/bin/python

import xml.etree.ElementTree as ET
import os
import sys
import numpy as np
from utils.models import Token, Entity
from utils.functions import get_entities, detect_label, tokenize

from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint

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

def create_indexs(loaded_dataset, max_length):
    '''
    Returns a mapping of each word seen in the data with an integer. Also a mapping of each tag (null, mechanism, advise,
    effect, int). Also returns maxlen
    It has an <UNK> token and <PAD> token that will be used for filling the encoded sentence till max_length
    '''
    all_indexes = {}
    tags_indexes = {'<PAD>': 0, 'O': 1, 'B-drug': 2, 'I-drug': 3, 'B-brand': 4, 'I-brand': 5, 'B-group': 6, 'I-group': 7, 'B-drug_n': 8, 'I-drug_n': 9}
    word_indexes = {"<PAD>": 0, "<UNK>": 1}
    word_counter = 2

    for _id, sentence in loaded_dataset.items():
        for entity in sentence:
            word = entity[0]
            if word not in word_indexes:
                word_indexes[word] = word_counter
                word_counter += 1

    all_indexes['words'] = word_indexes
    all_indexes['tags'] = tags_indexes
    all_indexes['maxlen'] = max_length

    return all_indexes

def build_network(idx):
    '''
    Input : Receives the index dictionary with the encodings of words and
    tags, and the maximum length of sentences.
    Output : Returns a compiled Keras neural network
    '''
    ## HYPERPARAMS ##
    batch_size = 64
    embedding_dim = 64
    n_words = len(idx['words'])
    n_outputs = len(idx['tags'])
    max_len = idx['maxlen']
    
    # Model architecture
    inputs = Input(shape = (max_len,))
    model = Embedding(input_dim = n_words, output_dim = embedding_dim, input_length = max_len)(inputs)
    model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation="relu"))(model)
    crf = CRF(n_outputs)  # CRF layer
    out = crf(model)  # output

    model = Model(inputs, out)
    # OTHER CONFIGURATION
    # adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=adam,
    #               metrics=['accuracy'])

    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    print("Model Summary")
    print(model.summary())

    return model

def encode_words(dataset, idx):
    encoded_words_all = []
    # Iterate through sentences
    for _id, sentence in dataset.items():
        sentence_words_encoded = []
        # Encode words on each sentence
        for entity in sentence:
            word = entity[0]
            try:
                word_encoded = idx['words'][word]
            except KeyError:
                word_encoded = idx['words']['<UNK>']
            sentence_words_encoded.append(word_encoded)
            if len(sentence_words_encoded) == idx['maxlen']:
                break
       
         # Apply padding if needed
        while len(sentence_words_encoded) < idx['maxlen']:
            sentence_words_encoded.append(idx['words']['<PAD>'])
        encoded_words_all.append(np.array(sentence_words_encoded, dtype=np.int32))
    
    # Return the 4 lists generated
    return np.array(encoded_words_all)

def encode_tags(dataset, idx):
    encoded_interaction_tags = []
    for _id, sentence in dataset.items():
        sentence_tags = []
        for entity in sentence:
            tag = entity[3]
            sentence_tags.append(idx['tags'][tag])

        while len(sentence_tags) < idx['maxlen']:
            sentence_tags.append(idx['tags']['<PAD>'])
        
        encoded_interaction_tags.append(sentence_tags)
    return encoded_interaction_tags


def save_model_and_indexs(model, idx, filename):
    model.save(filename + '.h5')

    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(idx, f, 0)


def learn(traindir, validationdir, modelname):
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
    model.fit(np.array(Xtrain), np.array(Ytrain)) #, validation_data=(Xval, Yval))

    # save model and indexs , for later use in prediction
    #save_model_and_indexs(model, idx, modelname)


### MAIN ###
# data_loaded = load_data("data/Train")
# idx = create_indexs(data_loaded, 100)
# encoded_words = encode_words(data_loaded, idx)
# encoded_tags = encode_tags(data_loaded, idx)
learn("data/Train", "data/Devel", "hola")
