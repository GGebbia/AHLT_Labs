#!/bin/python
import xml.etree.ElementTree as ET
import os
import sys
import pickle
import nltk
import numpy as np
from nltk.tokenize import WhitespaceTokenizer
from .models import Token, Entity

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

# Store as a list of dictionaries the word, the offset interval and the label (drug, group, brand, drug_n,...) of each entity in the sentence.
def get_entities(sentence):
    entities = []
    for ent in sentence.findall('entity'):
        entity = Entity(**ent.attrib)
        entities.append(entity)
    return entities

def tokenize(sentence):
    span_generator = WhitespaceTokenizer().span_tokenize(sentence)
    tokens = [(sentence[span[0]: span[1]], span[0], span[1] - 1) for span in span_generator]

    new_tokens = []
    for i, token in enumerate(tokens):
        word, offset_from, offset_to = token

        if (len(word) > 1) and (word.endswith(',') or word.endswith('.') or word.endswith(':') or word.endswith(';')):
            punct = word[-1]
            punct_offset_from = offset_to
            punct_offset_to = offset_to

            word = word[:-1]
            offset_to -= 1

            new_tokens.append(Token(word, offset_from, offset_to))
            new_tokens.append(Token(punct, punct_offset_from, punct_offset_to))

        elif (len(word) > 1) and word[0] == '(' and (word[0:2] != '(+' or word[0:2] != '(-'):
            punct = word[0]
            punct_offset_from = offset_from
            punct_offset_to = offset_from

            word = word[1:]
            offset_from += 1

            new_tokens.append(Token(punct, punct_offset_from, punct_offset_to))
            new_tokens.append(Token(word, offset_from, offset_to))
        else:
            new_tokens.append(Token(word, offset_from, offset_to))

    return new_tokens

def detect_label(token, entities):
    for entity in entities:
        # If the two offsets are equal, then it corresponds to the same word and type.
        if token.offset_from == entity.offset_from and token.offset_to == entity.offset_to:
            token.type = "B-" + entity.type
            return
        # If the token offset interval is inside the entity offset interval, then it is a first or the continuation of a type sequence.
        elif entity.offset_from <= token.offset_from and token.offset_to <= entity.offset_to:
            if entity.offset_from == token.offset_from:
                token.type = "B-" + entity.type
            else:
                token.type = "I-" + entity.type

            return
        else:
            token.type = "O"

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
        encoded_words_all.append(sentence_words_encoded)
   
    # Apply padding
    encoded_words_all = pad_sequences(maxlen = idx["maxlen"], sequences = encoded_words_all, padding = "post", value = idx["words"]["<PAD>"])
    
    encoded_words_all = np.array([np.array(l) for l in encoded_words_all])
    return encoded_words_all

def encode_tags(dataset, idx):
    n_tags = len(idx["tags"])
    encoded_interaction_tags = []
    for _id, sentence in dataset.items():
        sentence_tags = []
        for entity in sentence:
            tag = entity[3]
            sentence_tags.append(idx['tags'][tag])
        
        encoded_interaction_tags.append(sentence_tags)
    
    # Apply padding
    encoded_interaction_tags = pad_sequences(maxlen = idx["maxlen"], sequences = encoded_interaction_tags, padding = "post", value = idx["tags"]["<PAD>"])
    
    encoded_interaction_tags = np.array([np.array(l) for l in encoded_interaction_tags])
    encoded_interaction_tags = np.array([to_categorical(i, n_tags) for i in encoded_interaction_tags])
    return encoded_interaction_tags

def save_model_and_indexs(model, idx, filename):
    # model.save(filename + '.h5')

    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(idx, f, 0)

def load_model_and_indexs(filename):
    model = load_model(filename + '.h5', custom_objects={'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy':crf_viterbi_accuracy})
    indexs = pickle.load(open(filename + '.pkl', 'rb'))
    return model, indexs

