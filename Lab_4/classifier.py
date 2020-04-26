from utils_ml import *
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

def main():
    
    parser = argparse.ArgumentParser(description=
            """
            Classifier of the ML model loading the feature vectors and saving the model in the output directory
            """
                                     )
    parser.add_argument('--train', type=str, help='Path to Train dataset directory to use.')
    parser.add_argument('--test', type=str, help='Path to Test dataset directory to use')
    parser.add_argument('--model', type=str, help='Path to .sav model')
    parser.add_argument('--out', type=str, help='Output directory to save the results.')
    args = parser.parse_args()


    ### MAIN VARIABLES
    train_filename = os.path.join(args.train, "megam.dat")
    test_filename = os.path.join(args.test, "megam.dat")
    fulltest_filename = os.path.join(args.test, "feats.dat")

    # _type represents the if it has internal or external knowledge
    _type = args.test.split("/")[-2]
    inputdir = os.path.join("data", args.test.split("/")[-1])

    # If outputdir does not exist, we safely create it
    outputdir = args.out
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    output_filename = os.path.join(outputdir, "predicted.txt")

    train_feat_vects = open(train_filename, "r").read().split("\n")[:-1]
    test_feat_vects = open(test_filename, "r").read().split("\n")[:-1]

    X_train, Y_train, X_test, Y_test = process_feature_vectors(train_feat_vects, test_feat_vects, flag="test")

    model = joblib.load(args.model)
    Y_pred = classifier(model, X_test)

    outputfile = open(output_filename, "w")
    output_predicted_entities(Y_pred, fulltest_filename, outputfile)
    outputfile.close()

    evaluate(inputdir, output_filename)

    
if __name__ == "__main__":
    main()