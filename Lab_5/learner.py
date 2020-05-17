#!/bin/python

import xml.etree.ElementTree as ET
import os
import sys
from utils.models import Token, Entity
from utils.functions import get_entities, detect_label, tokenize
from utils.functions import load_data, create_indexs, encode_words, encode_tags, build_network, save_model_and_indexs

from keras.callbacks import ModelCheckpoint

def learn(traindir, validationdir, modelname, epochs, batch_size):
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
    checkpointer = ModelCheckpoint(filepath = '{}.h5'.format(modelname),
                       verbose = 0,
                       mode = 'auto',
                       save_best_only = True,
                       monitor='val_loss') 
    model.fit(Xtrain, Ytrain, validation_data=(Xval,Yval), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer])

    # save model and indexs , for later use in prediction
    save_model_and_indexs(model, idx, modelname)


### MAIN ###
epochs = 30
batch_size = 64
learn("data/Train", "data/Devel", "model_train_devel", epochs, batch_size)
