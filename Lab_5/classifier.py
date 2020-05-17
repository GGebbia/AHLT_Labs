#!/usr/bin/env python
import os
import sys

import pickle
import numpy as np

from nltk.parse.corenlp import CoreNLPDependencyParser
import xml.etree.ElementTree as ET

from keras.models import Model
from keras.models import load_model

from utils.functions import load_data, encode_words, encode_tags, load_model_and_indexs 

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
    os.system("java -jar eval/evaluateNER.jar " + inputdir + " " + outputfile)

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
    X = encode_words(testdata, idx)

    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each pair
    tags_map = {v: k for k, v in idx['tags'].items()}
    
    Y = [[ tags_map[np.argmax(word_pred)] for word_pred in sent_pred] for sent_pred in Y]
    print(Y)
    # one_hot_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    # Y = [tags_map[one_hot_map[np.argmax(y)]] for y in Y]

    # extract entities and dump them to output file
    #output_interactions(testdata, Y, outfile)

    # evaluate using official evaluator
    #evaluation(datadir, outfile)

## MAIN ##
predict('model_train_devel', 'data/Devel', 'class_predictions.txt')
