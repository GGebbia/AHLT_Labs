#!/usr/bin/env python

import xml.etree.ElementTree as ET
import os
import sys
import nltk
import numpy as np
import pickle
from nltk.parse.corenlp import CoreNLPDependencyParser

from keras.utils import np_utils
from keras.layers import Dense, Input, Reshape, Concatenate, Flatten
from keras.layers import Conv2D, MaxPool2D, Embedding, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from numpy.random import seed
import tensorflow

from sklearn.utils import class_weight

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
    types_indexes = {'null': 0, 'mechanism': 1, 'advise': 2, 'effect': 3, 'int': 4}
    word_indexes = {"<UNK>": 0, "<PAD>": 1}
    lemma_indexes = {"<UNK>": 0, "<PAD>": 1}
    tag_indexes = {"<UNK>": 0, "<PAD>": 1}
    word_counter = lemma_counter = tag_counter = 2

    # connect to CoreNLP server
    for data in loaded_dataset:
        for entities in data[4]:
            word = entities[0]
            lemma = entities[1]
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


def encode_tags(dataset, idx):
    encoded_interaction_tags = []
    for data in dataset:
        one_hot = [0, 0, 0, 0, 0]
        if idx['types'][data[3]] == 0:
            one_hot = [0, 0, 0, 0, 1]
        if idx['types'][data[3]] == 1:
            one_hot = [0, 0, 0, 1, 0]
        if idx['types'][data[3]] == 2:
            one_hot = [0, 0, 1, 0, 0]
        if idx['types'][data[3]] == 3:
            one_hot = [0, 1, 0, 0, 0]
        if idx['types'][data[3]] == 4:
            one_hot = [1, 0, 0, 0, 0]
        encoded_interaction_tags.append(one_hot)
    return np.array(encoded_interaction_tags)

def build_network(idx):
    '''
    Input : Receives the index dictionary with the encodings of words and
    tags, and the maximum length of sentences.
    Output : Returns a compiled Keras neural network
    '''
    # TODO, not using lemmas see how can train 2 embeddings
    # TODO use tags
    ## HYPERPARAMS ###
    embedding_dim = 64
    # sizes
    n_words = len(idx['words'])
    n_outputs = len(idx['types'])
    # n_word_tags = len(idx['tags'])
    # n_lemmas = len(idx['lemmas'])
    max_len = idx['maxlen']
    # create network layers
    inputs = Input(shape=(max_len,), dtype='int32')
    X_input_words_embedding = Embedding(input_dim=n_words, output_dim=embedding_dim, input_length=max_len)(inputs)
    X_input = Reshape((max_len, embedding_dim, 1))(X_input_words_embedding)
    # X_input_lemmas_embedding = Embedding(input_dim=n_lemmas, output_dim=embedding_dim, input_length=max_len)(inputs)

    # Input size = (embedding_dim_words (64) + embedding de lemmas (64) -not being used now- + one hot de 44 tags -not used now-  + 1 pos-not used-) * 100 paraules
    ### COPIED MODEL ###
    X1 = Conv2D(128, kernel_size=(3, embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu',
                name='conv1Filter1')(X_input)
    maxpool_1 = MaxPool2D(pool_size=(48, 1), strides=(1, 1), padding='valid', name='maxpool1')(X1)

    X2 = Conv2D(128, kernel_size=(4, embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu',
                name='conv1Filter2')(X_input)
    maxpool_2 = MaxPool2D(pool_size=(47, 1), strides=(1, 1), padding='valid', name='maxpool2')(X2)

    X3 = Conv2D(128, kernel_size=(5, embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu',
                name='conv1Filter3')(X_input)
    maxpool_3 = MaxPool2D(pool_size=(46, 1), strides=(1, 1), padding='valid', name='maxpool3')(X3)

    concatenated_tensor = Concatenate(axis=1)([maxpool_1, maxpool_2, maxpool_3])

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.25)(flatten)
    output = Dense(units=n_outputs, activation='softmax', name='fully_connected_affine_layer')(dropout)

    ###
    # create and compile model
    model = Model(inputs=inputs, outputs=output, name='intent_classifier')
    print("Model Summary")
    print(model.summary())
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


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
    # TODO use the other dicts
    Xtrain, _, _, _ = encode_words(traindata, idx)
    Ytrain = encode_tags(traindata, idx)
    Xval, _, _, _ = encode_words(valdata, idx)
    Yval = encode_tags(valdata, idx)

    y_integers = np.argmax(Ytrain, axis=1)
    class_weights = class_weight.compute_class_weight('balanced'
                                                     , np.unique(y_integers)
                                                     , y_integers)
    d_class_weights = dict(enumerate(class_weights))

    mc_acc = ModelCheckpoint('best_model_acc_no_balance.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    mc_loss = ModelCheckpoint('best_model_loss_no_balance.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
    # train model
    model.fit(Xtrain, Ytrain, batch_size=256, epochs=400, validation_data=(Xval, Yval), verbose=2, callbacks=[es, mc_acc, mc_loss], class_weight = d_class_weights)

    plot_model(model, to_file='ddi_model.png')
    scores = model.evaluate(Xval, Yval, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # save model and indexs , for later use in prediction
    save_model_and_indexs(model, idx, modelname)

def save_model_and_indexs(model, idx, filename):
    model.save(filename + '.h5')

    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(idx, f, 0)

seed(1)
tensorflow.random.set_seed(2)
learn("data/Train", "data/Devel", "model_1st_attempt_devel_test")

