import os
import joblib
import argparse
import itertools
import collections
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

def split_data(features):
    """
    Receive a list of featured sentences splitted by words and split it into samples and labels.

    Parameters:
    word_features(list): List of feature words. There is an empty element between sentences in order to split each one.

    Returns:
    X_samples(list): List of feature words missing the label of the word.
    Y_labels(list): List of labels for each word.
    """

    X_samples = []
    Y_labels = []

    for feat in features:
        feat = feat.split(" ")
        Y_labels.append(feat[0])
        X_samples.append(feat[1:])

    return X_samples, Y_labels

def transform_feature_vector_to_dataset(X_train, X_test, flag="train"):

    # We need to extract the features that belong on the train dataset in order to construct the dataset array for train
    # and test.
    
    flatten_X_samples = list(itertools.chain(*X_train))
    feature_names_eq = [item.split("=")[0]for item in flatten_X_samples if "=" in item]
    feature_names_eq = list(set(feature_names_eq))
    feature_names_without_eq = [item for item, count in collections.Counter(flatten_X_samples).items() if "=" not in item][0:]
    feature_names_without_eq.remove('')
    n_features = len(feature_names_eq) + len(feature_names_without_eq)

    new_X_train = np.zeros((len(X_train), n_features))
    new_X_test = np.zeros((len(X_test), n_features))
    
    if flag == "train":
        
        for i, sample in enumerate(X_train):
            for j, feature_name in enumerate(feature_names_eq):
                for feat in sample:
                    if feature_name in feat:
                        value = feat.split("=")[1]
                        new_X_train[i,j] = value
                        
            for j, feature_name in enumerate(feature_names_without_eq, len(feature_names_eq)):
                if feature_name in sample:
                    new_X_train[i,j] = 1
                    
    elif flag == "test":
    
        for i, sample in enumerate(X_test):
            for j, feature_name in enumerate(feature_names_eq):
                for feat in sample:
                    if feature_name in feat:
                        value = feat.split("=")[1]
                        new_X_test[i,j] = value
                        
            for j, feature_name in enumerate(feature_names_without_eq, len(feature_names_eq)):
                if feature_name in sample:
                    new_X_test[i,j] = 1

    return new_X_train, new_X_test


def process_feature_vectors(train_feature_vectors, test_feature_vectors, flag):
    X_train, Y_train = split_data(train_feature_vectors)
    X_test, Y_test = split_data(test_feature_vectors)

    new_X_train, new_X_test = transform_feature_vector_to_dataset(X_train, X_test, flag=flag)
    return new_X_train, Y_train, new_X_test, Y_test


def learner(algorithm, params, X, Y):
    model = algorithm(**params)
    model.fit(X, Y)
    return model

def classifier(model, X_test):
    Y_pred = model.predict(X_test)
    Y_pred = [[value] for value in Y_pred]
    return Y_pred

def output_predicted_entities(Y_pred, filename, outputfile):
    """
    Receives the predicted list of labels by the ML model and a filename of the detailed word features and construct predicted.txt file
    with the intrinsic details of each word, id, offsets and the predicted label.

    Parameters:
    Y_pred(list): List of list of predicted labels for each word.
    filename(str): filename to read the "feats.dat" data of the testing dataset

    Returns:

    """

    Y_pred_flatten = [el for line in Y_pred for el in line]
    detailed_word_features = open(filename).read().split("\n")[:-1]

    for label, detailed_feats in zip(Y_pred_flatten, detailed_word_features):
        _id, e1_id, e2_id = detailed_feats.split(" ")[0:3]
        if label == "null":
            interaction = "0"
        else:
            interaction = "1"
        line = [_id, e1_id, e2_id, interaction, label]
        outputfile.write("|".join(line) + "\n")

def evaluate(inputdir, outputfile):
    """
    Receives an input directory and the outputfile to evaluate the predicted labels with the evaluateNER.jar program.

    Parameters:
    inputdir(str):
    outputfile(str):

    Returns:

    """

    os.system("java -jar eval/evaluateDDI.jar " + inputdir + " " + outputfile)
